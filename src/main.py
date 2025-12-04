"""CLI interface for the Sales Enablement Pitch Generator."""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.tree import Tree

from src.acquisition.engine import ContentAcquisitionEngine
from src.models.config import SiteConfig

# Initialize CLI app
app = typer.Typer(
    name="pitch-gen",
    help="AI-powered sales pitch generator from product documentation",
    add_completion=False,
)

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Set up logging with rich formatting."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@app.command()
def extract(
    url: str = typer.Argument(..., help="Product website URL to extract content from"),
    product_id: Optional[str] = typer.Option(
        None, "--product-id", "-p", help="Product identifier for storage"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path for extracted content (JSON)"
    ),
    max_pages: int = typer.Option(
        20, "--max-pages", "-m", help="Maximum number of pages to extract"
    ),
    download_images: bool = typer.Option(
        True, "--download-images/--no-download-images", help="Download images locally"
    ),
    headless: bool = typer.Option(
        True, "--headless/--no-headless", help="Run browser in headless mode"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """
    Extract content from a product website.

    This command crawls the specified product website, handles dynamic content
    (SPAs, lazy loading, tabs), and extracts text, images, and tables.

    Example:
        pitch-gen extract https://example.com/product --product-id myproduct
    """
    setup_logging(verbose)

    console.print(Panel.fit(
        f"[bold blue]Content Acquisition Engine[/bold blue]\n"
        f"Extracting from: {url}",
        title="Sales Pitch Generator",
    ))

    # Create configuration
    config = SiteConfig.from_url(url)
    config.extraction.max_pages = max_pages
    config.extraction.download_images = download_images
    config.browser.headless = headless

    # Generate product ID if not provided
    if not product_id:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        product_id = parsed.netloc.replace(".", "_")

    async def run_extraction():
        async with ContentAcquisitionEngine(config) as engine:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Extracting content...", total=100)

                # Run extraction
                result = await engine.extract(product_id=product_id, save=True)

                progress.update(task, completed=100)

            return result

    try:
        result = asyncio.run(run_extraction())

        # Display results
        display_extraction_results(result)

        # Save to output file if specified
        if output:
            output_path = Path(output)
            with open(output_path, "w") as f:
                json.dump(result.model_dump(), f, default=str, indent=2)
            console.print(f"\n[green]Results saved to: {output_path}[/green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Extraction cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Extraction failed: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


@app.command()
def analyze(
    url: str = typer.Argument(..., help="Website URL to analyze"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """
    Analyze a website's structure without full extraction.

    Useful for understanding how a site is built before extraction.
    Detects SPA frameworks, navigation structure, and content areas.

    Example:
        pitch-gen analyze https://example.com/product
    """
    setup_logging(verbose)

    console.print(f"\n[bold]Analyzing site structure:[/bold] {url}\n")

    config = SiteConfig.from_url(url)

    async def run_analysis():
        from src.acquisition.browser import BrowserManager
        from src.acquisition.site_analyzer import SiteAnalyzer

        browser = BrowserManager(config.browser)
        analyzer = SiteAnalyzer(config)

        await browser.start()
        try:
            async with browser.get_page() as page:
                await page.goto(url, wait_until="networkidle")
                structure = await analyzer.analyze(page, url)
                recommendations = await analyzer.get_extraction_recommendations(structure)
                return structure, recommendations
        finally:
            await browser.stop()

    try:
        structure, recommendations = asyncio.run(run_analysis())

        # Display analysis results
        display_site_analysis(structure, recommendations)

    except Exception as e:
        console.print(f"\n[red]Analysis failed: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


@app.command()
def check(
    url: str = typer.Argument(..., help="Website URL to check for changes"),
    product_id: Optional[str] = typer.Option(
        None, "--product-id", "-p", help="Product identifier to compare against"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """
    Quick check if a website's content has changed.

    Compares current content against the last extraction for a product.
    Useful for scheduled monitoring.

    Example:
        pitch-gen check https://example.com/product --product-id myproduct
    """
    setup_logging(verbose)

    console.print(f"\n[bold]Checking for changes:[/bold] {url}\n")

    config = SiteConfig.from_url(url)

    async def run_check():
        async with ContentAcquisitionEngine(config) as engine:
            return await engine.quick_check(url)

    try:
        result = asyncio.run(run_check())

        # Display check results
        table = Table(title="Quick Check Results")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("URL", result["url"])
        table.add_row("Content Hash", result["content_hash"])
        table.add_row("Last Modified", result["last_modified"] or "Unknown")
        table.add_row("Checked At", result["checked_at"])

        console.print(table)

        # If product_id provided, compare with last extraction
        if product_id:
            from src.acquisition.fingerprint import ExtractionStore
            store = ExtractionStore()
            previous = store.get_latest_extraction(product_id)

            if previous:
                console.print(f"\n[bold]Comparison with last extraction:[/bold]")
                console.print(f"  Previous extraction: {previous.extraction_id}")
                console.print(f"  Previous hash: {previous.content_hash[:16]}...")
                # Note: Full comparison would require running extract_with_diff
            else:
                console.print(f"\n[yellow]No previous extraction found for {product_id}[/yellow]")

    except Exception as e:
        console.print(f"\n[red]Check failed: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


@app.command()
def list_extractions(
    product_id: Optional[str] = typer.Option(
        None, "--product-id", "-p", help="Filter by product ID"
    ),
    storage_path: str = typer.Option(
        "data/extractions", "--storage", "-s", help="Storage path"
    ),
) -> None:
    """
    List all saved extractions.

    Example:
        pitch-gen list-extractions --product-id myproduct
    """
    from pathlib import Path

    storage = Path(storage_path)
    if not storage.exists():
        console.print("[yellow]No extractions found. Storage directory doesn't exist.[/yellow]")
        return

    files = sorted(storage.glob("*.json"), reverse=True)

    if product_id:
        files = [f for f in files if f.name.startswith(product_id)]

    if not files:
        console.print("[yellow]No extractions found.[/yellow]")
        return

    table = Table(title="Saved Extractions")
    table.add_column("Extraction ID", style="cyan")
    table.add_column("Size", style="white")
    table.add_column("Created", style="green")

    for f in files[:20]:  # Show last 20
        extraction_id = f.stem
        size = f"{f.stat().st_size / 1024:.1f} KB"
        created = f.stat().st_mtime
        from datetime import datetime
        created_str = datetime.fromtimestamp(created).strftime("%Y-%m-%d %H:%M")
        table.add_row(extraction_id, size, created_str)

    console.print(table)

    if len(files) > 20:
        console.print(f"\n[dim]... and {len(files) - 20} more[/dim]")


def display_extraction_results(result) -> None:
    """Display extraction results in a formatted way."""
    console.print("\n")

    # Summary panel
    summary = f"""
[bold]Product:[/bold] {result.product_name or 'Unknown'}
[bold]URL:[/bold] {result.product_url}
[bold]Pages Extracted:[/bold] {result.total_pages_extracted}
[bold]Total Words:[/bold] {result.total_word_count:,}
[bold]Images:[/bold] {len(result.all_images)}
[bold]Tables:[/bold] {len(result.all_tables)}
[bold]Success Rate:[/bold] {result.extraction_success_rate * 100:.1f}%
[bold]Content Hash:[/bold] {result.content_hash[:16] if result.content_hash else 'N/A'}...
"""
    console.print(Panel(summary, title="Extraction Summary", border_style="green"))

    # Page tree
    if result.pages:
        tree = Tree("[bold]Extracted Pages[/bold]")
        for page in result.pages:
            page_node = tree.add(f"[cyan]{page.url}[/cyan]")
            page_node.add(f"Title: {page.title or 'No title'}")
            page_node.add(f"Words: {page.word_count}")
            page_node.add(f"Images: {len(page.images)}")
            page_node.add(f"Tables: {len(page.tables)}")
            page_node.add(f"Content Blocks: {len(page.content_blocks)}")
        console.print(tree)

    # Warnings/Errors
    if result.warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for warning in result.warnings:
            console.print(f"  - {warning}")

    if result.errors:
        console.print("\n[red]Errors:[/red]")
        for error in result.errors:
            console.print(f"  - {error}")


def display_site_analysis(structure, recommendations) -> None:
    """Display site analysis results."""
    # Site structure
    table = Table(title="Site Structure Analysis")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Base URL", structure.base_url)
    table.add_row("Is SPA", "Yes" if structure.is_spa else "No")
    table.add_row("Framework", structure.spa_framework or "Not detected")
    table.add_row("Requires JavaScript", "Yes" if structure.requires_javascript else "No")
    table.add_row("Has Cookie Banner", "Yes" if structure.has_cookie_banner else "No")
    table.add_row("Uses Lazy Loading", "Yes" if structure.uses_lazy_loading else "No")
    table.add_row("Uses Infinite Scroll", "Yes" if structure.uses_infinite_scroll else "No")
    table.add_row("Requires Interaction", "Yes" if structure.requires_interaction else "No")
    table.add_row("Analysis Confidence", f"{structure.analysis_confidence * 100:.0f}%")

    console.print(table)

    # Content selectors
    if structure.main_content_selector or structure.main_navigation_selector:
        console.print("\n[bold]Detected Selectors:[/bold]")
        if structure.main_navigation_selector:
            console.print(f"  Navigation: {structure.main_navigation_selector}")
        if structure.main_content_selector:
            console.print(f"  Main Content: {structure.main_content_selector}")
        if structure.footer_selector:
            console.print(f"  Footer: {structure.footer_selector}")

    # Discovered pages
    console.print(f"\n[bold]Discovered Pages:[/bold] {len(structure.discovered_pages)}")
    if structure.feature_pages:
        console.print(f"  Feature Pages: {len(structure.feature_pages)}")
    if structure.documentation_pages:
        console.print(f"  Documentation Pages: {len(structure.documentation_pages)}")
    if structure.pricing_pages:
        console.print(f"  Pricing Pages: {len(structure.pricing_pages)}")

    # Recommendations
    console.print("\n[bold]Extraction Recommendations:[/bold]")
    console.print(f"  Wait Strategy: {recommendations['wait_strategy']}")
    console.print(f"  Needs Scrolling: {'Yes' if recommendations['needs_scroll'] else 'No'}")
    console.print(f"  Needs Interaction: {'Yes' if recommendations['needs_interaction'] else 'No'}")
    console.print(f"  Estimated Pages: {recommendations['estimated_pages']}")

    if recommendations['warnings']:
        console.print("\n[yellow]Warnings:[/yellow]")
        for warning in recommendations['warnings']:
            console.print(f"  - {warning}")

    # Priority pages
    if recommendations['priority_pages']:
        console.print("\n[bold]Priority Pages for Extraction:[/bold]")
        for i, page in enumerate(recommendations['priority_pages'][:10], 1):
            console.print(f"  {i}. {page}")
        if len(recommendations['priority_pages']) > 10:
            console.print(f"  ... and {len(recommendations['priority_pages']) - 10} more")


@app.callback()
def main():
    """
    Sales Enablement Pitch Generator

    Generate compelling sales pitch documents from product websites.
    Uses AI to extract, understand, and transform product documentation
    into effective sales collateral.
    """
    pass


if __name__ == "__main__":
    app()

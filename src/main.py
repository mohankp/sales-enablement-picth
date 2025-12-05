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
from src.models.content import ExtractedContent

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


@app.command()
def process(
    input_file: str = typer.Argument(..., help="Path to extracted content JSON file"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path for processed content (JSON)"
    ),
    product_name: Optional[str] = typer.Option(
        None, "--product-name", "-n", help="Override product name"
    ),
    aspects: Optional[str] = typer.Option(
        None, "--aspects", "-a",
        help="Comma-separated aspects to process (features,benefits,use_cases,competitive,audience,pricing,technical,summary)"
    ),
    model: str = typer.Option(
        "sonnet", "--model", "-m", help="Model to use (haiku, sonnet, opus)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """
    Process extracted content with LLM to analyze and structure product information.

    Takes extracted content JSON and produces structured analysis including
    features, benefits, use cases, competitive positioning, audience segments,
    pricing, and technical specifications.

    Example:
        pitch-gen process extraction.json --output processed.json
        pitch-gen process extraction.json --aspects features,benefits
    """
    setup_logging(verbose)

    input_path = Path(input_file)
    if not input_path.exists():
        console.print(f"[red]Input file not found: {input_file}[/red]")
        sys.exit(1)

    console.print(Panel.fit(
        f"[bold blue]Content Processing Engine[/bold blue]\n"
        f"Processing: {input_file}",
        title="Sales Pitch Generator",
    ))

    # Load extracted content
    try:
        with open(input_path) as f:
            data = json.load(f)
        content = ExtractedContent.model_validate(data)
    except Exception as e:
        console.print(f"[red]Failed to load extraction file: {e}[/red]")
        sys.exit(1)

    # Import processing modules
    from src.processing import ContentProcessor, ProcessingConfig
    from src.llm import ModelName

    # Configure processing
    model_map = {
        "haiku": ModelName.HAIKU,
        "sonnet": ModelName.SONNET,
        "opus": ModelName.OPUS,
    }
    selected_model = model_map.get(model.lower(), ModelName.SONNET)

    config = ProcessingConfig(
        default_model=selected_model,
        analysis_model=selected_model,
        extraction_model=selected_model,
    )

    # Disable aspects not requested
    if aspects:
        aspect_list = [a.strip().lower() for a in aspects.split(",")]
        config.enable_features = "features" in aspect_list
        config.enable_benefits = "benefits" in aspect_list
        config.enable_use_cases = "use_cases" in aspect_list
        config.enable_competitive = "competitive" in aspect_list
        config.enable_audience = "audience" in aspect_list
        config.enable_pricing = "pricing" in aspect_list
        config.enable_technical = "technical" in aspect_list

    async def run_processing():
        async with ContentProcessor(config) as processor:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Processing content with LLM...", total=None)
                result = await processor.process(content, product_name=product_name)
                progress.update(task, completed=True)
            return result

    try:
        result = asyncio.run(run_processing())

        # Display results
        display_processing_results(result)

        # Save to output file if specified
        if output:
            output_path = Path(output)
            with open(output_path, "w") as f:
                json.dump(result.model_dump(), f, default=str, indent=2)
            console.print(f"\n[green]Results saved to: {output_path}[/green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Processing cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Processing failed: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


def display_processing_results(result) -> None:
    """Display processing results in a formatted way."""
    console.print("\n")

    # Summary panel
    summary = f"""
[bold]Product:[/bold] {result.product_name}
[bold]URL:[/bold] {result.product_url}
[bold]Processing ID:[/bold] {result.processing_id}
[bold]Duration:[/bold] {result.processing_duration_ms / 1000:.1f}s
[bold]Tokens Used:[/bold] {result.total_llm_tokens_used:,}
[bold]Cost:[/bold] ${result.total_llm_cost_usd:.4f}
[bold]Confidence:[/bold] {result.overall_confidence * 100:.0f}%
"""
    console.print(Panel(summary, title="Processing Summary", border_style="green"))

    # Executive Summary
    if result.summary.executive_summary:
        console.print(Panel(
            result.summary.executive_summary,
            title="Executive Summary",
            border_style="blue",
        ))

    # Key Points
    if result.summary.key_points:
        console.print("\n[bold]Key Points:[/bold]")
        for point in result.summary.key_points:
            console.print(f"  • {point}")

    # Features table
    if result.features.features:
        table = Table(title=f"Features ({len(result.features.features)} found)")
        table.add_column("Feature", style="cyan", max_width=30)
        table.add_column("Category", style="magenta")
        table.add_column("Description", style="white", max_width=50)
        table.add_column("Flagship", style="green")

        for feature in result.features.features[:10]:
            table.add_row(
                feature.name,
                feature.category.value,
                feature.description[:100] + "..." if len(feature.description) > 100 else feature.description,
                "✓" if feature.is_flagship else "",
            )

        if len(result.features.features) > 10:
            table.add_row("...", "...", f"(and {len(result.features.features) - 10} more)", "")

        console.print("\n")
        console.print(table)

    # Benefits
    if result.benefits.benefits:
        console.print(f"\n[bold]Top Benefits ({len(result.benefits.benefits)} total):[/bold]")
        for benefit in result.benefits.benefits[:5]:
            console.print(f"  • [cyan]{benefit.headline}[/cyan]")
            console.print(f"    {benefit.description[:150]}...")

    # Use Cases
    if result.use_cases.use_cases:
        console.print(f"\n[bold]Use Cases ({len(result.use_cases.use_cases)} found):[/bold]")
        for uc in result.use_cases.use_cases[:5]:
            console.print(f"  • [cyan]{uc.title}[/cyan]: {uc.problem_solved[:100]}...")

    # Competitive Differentiators
    if result.competitive_analysis.differentiators:
        console.print(f"\n[bold]Differentiators ({len(result.competitive_analysis.differentiators)} found):[/bold]")
        for diff in result.competitive_analysis.differentiators[:3]:
            console.print(f"  • [green]{diff.claim}[/green] ({diff.strength})")

    # Audience Segments
    if result.audience_analysis.segments:
        console.print(f"\n[bold]Target Audiences ({len(result.audience_analysis.segments)} segments):[/bold]")
        if result.audience_analysis.primary_audience:
            console.print(f"  Primary: [cyan]{result.audience_analysis.primary_audience}[/cyan]")
        for segment in result.audience_analysis.segments[:3]:
            console.print(f"  • {segment.name} ({segment.segment_type.value})")

    # Pricing
    if result.pricing.tiers:
        console.print(f"\n[bold]Pricing ({result.pricing.pricing_model.value}):[/bold]")
        if result.pricing.has_free_trial:
            console.print(f"  Free Trial: {result.pricing.trial_duration or 'Yes'}")
        for tier in result.pricing.tiers[:4]:
            price_str = tier.price or "Contact Sales"
            console.print(f"  • [cyan]{tier.name}[/cyan]: {price_str}")

    # Technical Specs
    if result.technical_specs.platforms_supported or result.technical_specs.integrations:
        console.print("\n[bold]Technical:[/bold]")
        if result.technical_specs.platforms_supported:
            console.print(f"  Platforms: {', '.join(result.technical_specs.platforms_supported[:5])}")
        if result.technical_specs.api_available:
            console.print(f"  API: {result.technical_specs.api_type or 'Available'}")
        if result.technical_specs.security_certifications:
            console.print(f"  Security: {', '.join(result.technical_specs.security_certifications[:3])}")

    # Warnings/Errors
    if result.warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for warning in result.warnings:
            console.print(f"  - {warning}")

    if result.errors:
        console.print("\n[red]Errors:[/red]")
        for error in result.errors:
            console.print(f"  - {error}")


@app.command()
def batch_process(
    input_dir: str = typer.Argument(..., help="Directory containing extraction JSON files"),
    output_dir: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output directory for processed files"
    ),
    pattern: str = typer.Option(
        "*.json", "--pattern", "-p", help="Glob pattern for input files"
    ),
    concurrency: int = typer.Option(
        3, "--concurrency", "-c", help="Maximum concurrent processing (1-10)"
    ),
    model: str = typer.Option(
        "sonnet", "--model", "-m", help="Model to use (haiku, sonnet, opus)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """
    Process multiple extraction files in batch.

    Efficiently processes multiple extracted content files with
    concurrent processing, caching, and retry logic.

    Example:
        pitch-gen batch-process ./extractions --output ./processed --concurrency 5
    """
    setup_logging(verbose)

    input_path = Path(input_dir)
    if not input_path.exists():
        console.print(f"[red]Input directory not found: {input_dir}[/red]")
        sys.exit(1)

    # Find input files
    input_files = list(input_path.glob(pattern))
    if not input_files:
        console.print(f"[yellow]No files matching '{pattern}' found in {input_dir}[/yellow]")
        sys.exit(0)

    console.print(Panel.fit(
        f"[bold blue]Batch Processing Engine[/bold blue]\n"
        f"Processing {len(input_files)} files from: {input_dir}",
        title="Sales Pitch Generator",
    ))

    # Import processing modules
    from src.processing import BatchProcessor, BatchConfig, ProcessingConfig
    from src.llm import ModelName

    # Configure
    model_map = {
        "haiku": ModelName.HAIKU,
        "sonnet": ModelName.SONNET,
        "opus": ModelName.OPUS,
    }
    selected_model = model_map.get(model.lower(), ModelName.SONNET)

    processing_config = ProcessingConfig(
        default_model=selected_model,
        analysis_model=selected_model,
        extraction_model=selected_model,
    )

    batch_config = BatchConfig(
        max_concurrent_items=min(max(1, concurrency), 10),
        processing_config=processing_config,
    )

    # Setup output directory
    output_path = Path(output_dir) if output_dir else input_path / "processed"
    output_path.mkdir(parents=True, exist_ok=True)

    # Progress tracking
    def progress_callback(completed: int, total: int, current: Optional[str]) -> None:
        if current:
            console.print(f"  [{completed}/{total}] Processing: {current}")

    async def run_batch():
        async with BatchProcessor(batch_config, progress_callback=progress_callback) as processor:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"Processing {len(input_files)} files...",
                    total=len(input_files),
                )

                result = await processor.process_from_files(input_files)

                progress.update(task, completed=len(input_files))

            return result

    try:
        result = asyncio.run(run_batch())

        # Display results summary
        display_batch_results(result)

        # Save individual results
        for item_result in result.get_successful_items():
            if item_result.result:
                output_file = output_path / f"{item_result.item_id}_processed.json"
                with open(output_file, "w") as f:
                    json.dump(item_result.result.model_dump(), f, default=str, indent=2)

        console.print(f"\n[green]Results saved to: {output_path}[/green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Batch processing cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Batch processing failed: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


def display_batch_results(result) -> None:
    """Display batch processing results."""
    console.print("\n")

    # Summary panel
    summary = f"""
[bold]Total Items:[/bold] {result.total_items}
[bold]Successful:[/bold] {result.successful}
[bold]Failed:[/bold] {result.failed}
[bold]Success Rate:[/bold] {result.success_rate * 100:.1f}%
[bold]Total Time:[/bold] {result.total_time_ms / 1000:.1f}s
[bold]Total Tokens:[/bold] {result.total_tokens:,}
[bold]Total Cost:[/bold] ${result.total_cost_usd:.4f}
"""
    console.print(Panel(summary, title="Batch Processing Summary", border_style="green"))

    # Show failed items if any
    failed = result.get_failed_items()
    if failed:
        console.print("\n[red]Failed Items:[/red]")
        for item in failed:
            console.print(f"  - {item.item_id}: {item.error}")

    # Show successful items
    successful = result.get_successful_items()
    if successful and len(successful) <= 20:
        console.print("\n[green]Successful Items:[/green]")
        for item in successful:
            console.print(f"  - {item.item_id} ({item.processing_time_ms:.0f}ms)")


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

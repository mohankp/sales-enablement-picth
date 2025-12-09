"""CLI interface for the Sales Enablement Pitch Generator."""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import typer

# Load environment variables from .env file
load_dotenv()
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
        "sonnet", "--model", "-m", help="Model to use (haiku, sonnet, opus for Anthropic; pro, flash for Gemini)"
    ),
    provider: str = typer.Option(
        "anthropic", "--provider", "-p", help="LLM provider to use (anthropic, gemini)"
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
        pitch-gen process extraction.json --provider gemini --model pro
    """
    setup_logging(verbose)

    input_path = Path(input_file)
    if not input_path.exists():
        console.print(f"[red]Input file not found: {input_file}[/red]")
        sys.exit(1)

    console.print(Panel.fit(
        f"[bold blue]Content Processing Engine[/bold blue]\n"
        f"Processing: {input_file}\n"
        f"Provider: {provider}",
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
    from src.llm import ModelName, ProviderType

    # Determine provider
    provider_map = {
        "anthropic": ProviderType.ANTHROPIC,
        "gemini": ProviderType.GEMINI,
    }
    selected_provider = provider_map.get(provider.lower(), ProviderType.ANTHROPIC)

    # Configure processing based on provider
    if selected_provider == ProviderType.GEMINI:
        # Map model names for Gemini
        gemini_model_map = {
            "pro": "gemini-3-pro-preview",  # Gemini 3 Pro (most intelligent)
            "flash": "gemini-2.5-flash",
            "fast": "gemini-2.5-flash",
            "lite": "gemini-2.5-flash-lite",
            "2.5-pro": "gemini-2.5-pro",
        }
        gemini_model = gemini_model_map.get(model.lower(), "gemini-3-pro-preview")
        config = ProcessingConfig(
            provider=selected_provider,
            gemini_default_model=gemini_model,
            gemini_analysis_model=gemini_model,
            gemini_extraction_model=gemini_model,
        )
    else:
        # Anthropic models
        model_map = {
            "haiku": ModelName.HAIKU,
            "sonnet": ModelName.SONNET,
            "opus": ModelName.OPUS,
        }
        selected_model = model_map.get(model.lower(), ModelName.SONNET)
        config = ProcessingConfig(
            provider=selected_provider,
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
        "*.json", "--pattern", "-pt", help="Glob pattern for input files"
    ),
    concurrency: int = typer.Option(
        3, "--concurrency", "-c", help="Maximum concurrent processing (1-10)"
    ),
    model: str = typer.Option(
        "sonnet", "--model", "-m", help="Model to use (haiku, sonnet, opus for Anthropic; pro, flash for Gemini)"
    ),
    provider: str = typer.Option(
        "anthropic", "--provider", "-p", help="LLM provider to use (anthropic, gemini)"
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
        pitch-gen batch-process ./extractions --provider gemini --model flash
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
        f"Processing {len(input_files)} files from: {input_dir}\n"
        f"Provider: {provider}",
        title="Sales Pitch Generator",
    ))

    # Import processing modules
    from src.processing import BatchProcessor, BatchConfig, ProcessingConfig
    from src.llm import ModelName, ProviderType

    # Determine provider
    provider_map = {
        "anthropic": ProviderType.ANTHROPIC,
        "gemini": ProviderType.GEMINI,
    }
    selected_provider = provider_map.get(provider.lower(), ProviderType.ANTHROPIC)

    # Configure based on provider
    if selected_provider == ProviderType.GEMINI:
        gemini_model_map = {
            "pro": "gemini-3-pro-preview",  # Gemini 3 Pro (most intelligent)
            "flash": "gemini-2.5-flash",
            "fast": "gemini-2.5-flash",
            "lite": "gemini-2.5-flash-lite",
            "2.5-pro": "gemini-2.5-pro",
        }
        gemini_model = gemini_model_map.get(model.lower(), "gemini-3-pro-preview")
        processing_config = ProcessingConfig(
            provider=selected_provider,
            gemini_default_model=gemini_model,
            gemini_analysis_model=gemini_model,
            gemini_extraction_model=gemini_model,
        )
    else:
        model_map = {
            "haiku": ModelName.HAIKU,
            "sonnet": ModelName.SONNET,
            "opus": ModelName.OPUS,
        }
        selected_model = model_map.get(model.lower(), ModelName.SONNET)
        processing_config = ProcessingConfig(
            provider=selected_provider,
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


@app.command()
def generate(
    input_file: str = typer.Argument(..., help="Path to processed content JSON file"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path for generated pitch (JSON)"
    ),
    output_format: str = typer.Option(
        "json", "--format", "-f", help="Output format: json, markdown, or text"
    ),
    tone: str = typer.Option(
        "professional", "--tone", "-t",
        help="Pitch tone: professional, conversational, technical, executive, enthusiastic, consultative"
    ),
    length: str = typer.Option(
        "standard", "--length", "-l",
        help="Pitch length: elevator, short, standard, detailed, comprehensive"
    ),
    audience: Optional[str] = typer.Option(
        None, "--audience", "-a", help="Target audience for customization"
    ),
    include_pricing: bool = typer.Option(
        True, "--pricing/--no-pricing", help="Include pricing section"
    ),
    include_technical: bool = typer.Option(
        False, "--technical/--no-technical", help="Include technical details section"
    ),
    model: str = typer.Option(
        "sonnet", "--model", "-m", help="Model to use (haiku, sonnet, opus for Anthropic; pro, flash for Gemini)"
    ),
    provider: str = typer.Option(
        "anthropic", "--provider", "-p", help="LLM provider to use (anthropic, gemini)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """
    Generate a sales pitch from processed content.

    Takes processed content JSON and generates a complete sales pitch
    with customizable tone, length, and audience targeting.

    Example:
        pitch-gen generate processed.json --output pitch.json --tone executive
        pitch-gen generate processed.json --format markdown --length detailed
        pitch-gen generate processed.json --provider gemini --model pro
    """
    setup_logging(verbose)

    input_path = Path(input_file)
    if not input_path.exists():
        console.print(f"[red]Input file not found: {input_file}[/red]")
        sys.exit(1)

    console.print(Panel.fit(
        f"[bold blue]Pitch Generation Engine[/bold blue]\n"
        f"Generating from: {input_file}\n"
        f"Provider: {provider}",
        title="Sales Pitch Generator",
    ))

    # Load processed content
    try:
        with open(input_path) as f:
            data = json.load(f)
        from src.models.processed import ProcessedContent
        processed_content = ProcessedContent.model_validate(data)
    except Exception as e:
        console.print(f"[red]Failed to load processed content: {e}[/red]")
        sys.exit(1)

    # Import generation modules
    from src.generation import PitchGenerator, GenerationConfig
    from src.models.pitch import PitchConfig, PitchTone, PitchLength
    from src.llm import LLMConfig, ModelSettings, ModelName, ProviderType

    # Map CLI options to enums
    tone_map = {
        "professional": PitchTone.PROFESSIONAL,
        "conversational": PitchTone.CONVERSATIONAL,
        "technical": PitchTone.TECHNICAL,
        "executive": PitchTone.EXECUTIVE,
        "enthusiastic": PitchTone.ENTHUSIASTIC,
        "consultative": PitchTone.CONSULTATIVE,
    }
    length_map = {
        "elevator": PitchLength.ELEVATOR,
        "short": PitchLength.SHORT,
        "standard": PitchLength.STANDARD,
        "detailed": PitchLength.DETAILED,
        "comprehensive": PitchLength.COMPREHENSIVE,
    }

    # Determine provider
    provider_map = {
        "anthropic": ProviderType.ANTHROPIC,
        "gemini": ProviderType.GEMINI,
    }
    selected_provider = provider_map.get(provider.lower(), ProviderType.ANTHROPIC)

    selected_tone = tone_map.get(tone.lower(), PitchTone.PROFESSIONAL)
    selected_length = length_map.get(length.lower(), PitchLength.STANDARD)

    # Configure pitch
    pitch_config = PitchConfig(
        tone=selected_tone,
        length=selected_length,
        target_audience=audience,
        include_pricing=include_pricing,
        include_technical=include_technical,
    )

    # Configure generation based on provider
    llm_config = LLMConfig()

    if selected_provider == ProviderType.GEMINI:
        gemini_model_map = {
            "pro": "gemini-3-pro-preview",  # Gemini 3 Pro (most intelligent)
            "flash": "gemini-2.5-flash",
            "fast": "gemini-2.5-flash",
            "lite": "gemini-2.5-flash-lite",
            "2.5-pro": "gemini-2.5-pro",
        }
        gemini_model = gemini_model_map.get(model.lower(), "gemini-3-pro-preview")
        gen_config = GenerationConfig(
            provider=selected_provider,
            llm_config=llm_config,
            gemini_model=gemini_model,
            verbose=verbose,
        )
    else:
        model_map = {
            "haiku": ModelName.HAIKU,
            "sonnet": ModelName.SONNET,
            "opus": ModelName.OPUS,
        }
        selected_model = model_map.get(model.lower(), ModelName.SONNET)
        model_settings = ModelSettings(model=selected_model)
        gen_config = GenerationConfig(
            provider=selected_provider,
            llm_config=llm_config,
            model_settings=model_settings,
            verbose=verbose,
        )

    async def run_generation():
        async with PitchGenerator(gen_config) as generator:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Generating pitch...", total=None)
                result = await generator.generate(processed_content, pitch_config)
                progress.update(task, completed=True)
            return result

    try:
        result = asyncio.run(run_generation())

        # Display results
        display_generation_results(result)

        # Save output
        if output or output_format != "json":
            output_path = Path(output) if output else input_path.with_suffix(f".pitch.{output_format}")

            if output_format == "json":
                with open(output_path, "w") as f:
                    json.dump(result.pitch.model_dump(), f, default=str, indent=2)
            elif output_format == "markdown":
                with open(output_path, "w") as f:
                    f.write(result.pitch.get_full_content())
            elif output_format == "text":
                with open(output_path, "w") as f:
                    # Plain text version
                    f.write(f"{result.pitch.title}\n")
                    f.write("=" * len(result.pitch.title) + "\n\n")
                    if result.pitch.subtitle:
                        f.write(f"{result.pitch.subtitle}\n\n")
                    f.write(f"{result.pitch.executive_summary}\n\n")
                    for section in sorted(result.pitch.sections, key=lambda s: s.order):
                        f.write(f"{section.title}\n")
                        f.write("-" * len(section.title) + "\n")
                        f.write(f"{section.content}\n\n")
                        if section.key_points:
                            for point in section.key_points:
                                f.write(f"  • {point}\n")
                            f.write("\n")

            console.print(f"\n[green]Pitch saved to: {output_path}[/green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Generation cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Generation failed: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


def display_generation_results(result) -> None:
    """Display pitch generation results."""
    console.print("\n")
    pitch = result.pitch

    # Summary panel
    summary = f"""
[bold]Product:[/bold] {pitch.product_name}
[bold]Pitch ID:[/bold] {pitch.pitch_id}
[bold]Tone:[/bold] {pitch.config.tone.value}
[bold]Length:[/bold] {pitch.config.length.value}
[bold]Sections:[/bold] {len(pitch.sections)}
[bold]Word Count:[/bold] {pitch.word_count():,}
[bold]Duration:[/bold] ~{result.total_duration_ms / 1000:.1f}s to generate
[bold]Est. Presentation:[/bold] ~{pitch.estimated_duration_minutes():.1f} minutes
[bold]Tokens Used:[/bold] {result.total_tokens_used:,}
[bold]Cost:[/bold] ${result.total_cost_usd:.4f}
[bold]Confidence:[/bold] {pitch.overall_confidence * 100:.0f}%
"""
    console.print(Panel(summary, title="Generation Summary", border_style="green"))

    # One-liner and elevator pitch
    if pitch.one_liner:
        console.print(Panel(
            pitch.one_liner,
            title="One-Liner",
            border_style="cyan",
        ))

    if pitch.elevator_pitch:
        console.print(Panel(
            pitch.elevator_pitch,
            title="Elevator Pitch (30 seconds)",
            border_style="blue",
        ))

    # Key messages
    if pitch.key_messages:
        console.print("\n[bold]Key Messages:[/bold]")
        for i, msg in enumerate(pitch.key_messages, 1):
            console.print(f"  {i}. {msg}")

    # Sections overview
    console.print("\n[bold]Pitch Sections:[/bold]")
    for section in sorted(pitch.sections, key=lambda s: s.order):
        console.print(f"  [{section.order}] [cyan]{section.title}[/cyan] ({section.section_type.value})")
        if section.key_points:
            for point in section.key_points[:2]:
                console.print(f"      • {point[:80]}...")

    # Feature highlights
    if pitch.feature_highlights:
        console.print(f"\n[bold]Feature Highlights ({len(pitch.feature_highlights)}):[/bold]")
        for feature in pitch.feature_highlights[:5]:
            console.print(f"  • [cyan]{feature.name}[/cyan]: {feature.headline[:60]}...")

    # Benefit statements
    if pitch.benefit_statements:
        console.print(f"\n[bold]Benefit Statements ({len(pitch.benefit_statements)}):[/bold]")
        for benefit in pitch.benefit_statements[:5]:
            console.print(f"  • [green]{benefit.headline}[/green]")

    # Objection handling
    if pitch.common_objections:
        console.print(f"\n[bold]Objection Responses ({len(pitch.common_objections)}):[/bold]")
        for objection in list(pitch.common_objections.keys())[:3]:
            console.print(f"  • [yellow]\"{objection}\"[/yellow]")

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
def export_pptx(
    input_file: str = typer.Argument(..., help="Path to generated pitch JSON file"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output PPTX file path"
    ),
    theme: str = typer.Option(
        "corporate_blue", "--theme", "-t",
        help="Color theme: corporate_blue, modern_dark, clean_light, professional_green, executive_gray"
    ),
    include_notes: bool = typer.Option(
        True, "--notes/--no-notes", help="Include speaker notes"
    ),
    include_images: bool = typer.Option(
        True, "--images/--no-images", help="Include visual assets"
    ),
    slide_numbers: bool = typer.Option(
        True, "--slide-numbers/--no-slide-numbers", help="Add slide numbers"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """
    Export a generated pitch to PowerPoint (PPTX) format.

    Creates a professional presentation with title slide, content slides,
    elevator pitch, and closing slide. Supports multiple color themes.

    Example:
        pitch-gen export-pptx pitch.json --output presentation.pptx
        pitch-gen export-pptx pitch.json --theme modern_dark --no-notes
    """
    setup_logging(verbose)

    input_path = Path(input_file)
    if not input_path.exists():
        console.print(f"[red]Input file not found: {input_file}[/red]")
        sys.exit(1)

    console.print(Panel.fit(
        f"[bold blue]PPTX Export[/bold blue]\n"
        f"Exporting from: {input_file}",
        title="Sales Pitch Generator",
    ))

    # Load pitch
    try:
        with open(input_path) as f:
            data = json.load(f)
        from src.models.pitch import Pitch
        pitch = Pitch.model_validate(data)
    except Exception as e:
        console.print(f"[red]Failed to load pitch file: {e}[/red]")
        sys.exit(1)

    # Import composer
    from src.generation.composers import PPTXComposer, PPTXConfig
    from src.generation.composers.base import ThemeColor

    # Map theme
    theme_map = {
        "corporate_blue": ThemeColor.CORPORATE_BLUE,
        "modern_dark": ThemeColor.MODERN_DARK,
        "clean_light": ThemeColor.CLEAN_LIGHT,
        "professional_green": ThemeColor.PROFESSIONAL_GREEN,
        "executive_gray": ThemeColor.EXECUTIVE_GRAY,
    }
    selected_theme = theme_map.get(theme.lower(), ThemeColor.CORPORATE_BLUE)

    # Configure
    config = PPTXConfig(
        theme=selected_theme,
        include_speaker_notes=include_notes,
        include_visual_assets=include_images,
        add_slide_numbers=slide_numbers,
    )

    try:
        composer = PPTXComposer(config)
        output_path = Path(output) if output else input_path.with_suffix(".pptx")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating PPTX...", total=None)
            result = composer.compose(pitch, output_path)
            progress.update(task, completed=True)

        if result.success:
            console.print(f"\n[green]PPTX saved to: {result.output_path}[/green]")
            console.print(f"  Slides: {result.page_count}")
            console.print(f"  Size: {result.file_size_bytes / 1024:.1f} KB")
            console.print(f"  Theme: {selected_theme.value}")

            if result.warnings:
                console.print("\n[yellow]Warnings:[/yellow]")
                for warning in result.warnings:
                    console.print(f"  - {warning}")
        else:
            console.print(f"\n[red]Export failed:[/red]")
            for error in result.errors:
                console.print(f"  - {error}")
            sys.exit(1)

    except Exception as e:
        console.print(f"\n[red]Export failed: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


@app.command()
def export_pdf(
    input_file: str = typer.Argument(..., help="Path to generated pitch JSON file"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output PDF file path"
    ),
    theme: str = typer.Option(
        "corporate_blue", "--theme", "-t",
        help="Color theme: corporate_blue, modern_dark, clean_light, professional_green, executive_gray"
    ),
    page_size: str = typer.Option(
        "letter", "--page-size", "-p", help="Page size: letter or a4"
    ),
    include_toc: bool = typer.Option(
        True, "--toc/--no-toc", help="Include table of contents"
    ),
    include_images: bool = typer.Option(
        True, "--images/--no-images", help="Include visual assets"
    ),
    justified: bool = typer.Option(
        True, "--justified/--no-justified", help="Use justified text alignment"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """
    Export a generated pitch to PDF format.

    Creates a professional document with title page, table of contents,
    all pitch sections, feature highlights, benefits, and call to action.

    Example:
        pitch-gen export-pdf pitch.json --output document.pdf
        pitch-gen export-pdf pitch.json --theme executive_gray --page-size a4
    """
    setup_logging(verbose)

    input_path = Path(input_file)
    if not input_path.exists():
        console.print(f"[red]Input file not found: {input_file}[/red]")
        sys.exit(1)

    console.print(Panel.fit(
        f"[bold blue]PDF Export[/bold blue]\n"
        f"Exporting from: {input_file}",
        title="Sales Pitch Generator",
    ))

    # Load pitch
    try:
        with open(input_path) as f:
            data = json.load(f)
        from src.models.pitch import Pitch
        pitch = Pitch.model_validate(data)
    except Exception as e:
        console.print(f"[red]Failed to load pitch file: {e}[/red]")
        sys.exit(1)

    # Import composer
    from src.generation.composers import PDFComposer, PDFConfig
    from src.generation.composers.base import ThemeColor

    # Map theme
    theme_map = {
        "corporate_blue": ThemeColor.CORPORATE_BLUE,
        "modern_dark": ThemeColor.MODERN_DARK,
        "clean_light": ThemeColor.CLEAN_LIGHT,
        "professional_green": ThemeColor.PROFESSIONAL_GREEN,
        "executive_gray": ThemeColor.EXECUTIVE_GRAY,
    }
    selected_theme = theme_map.get(theme.lower(), ThemeColor.CORPORATE_BLUE)

    # Configure
    config = PDFConfig(
        theme=selected_theme,
        page_size=page_size.lower(),
        include_table_of_contents=include_toc,
        include_visual_assets=include_images,
        justified_text=justified,
    )

    try:
        composer = PDFComposer(config)
        output_path = Path(output) if output else input_path.with_suffix(".pdf")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating PDF...", total=None)
            result = composer.compose(pitch, output_path)
            progress.update(task, completed=True)

        if result.success:
            console.print(f"\n[green]PDF saved to: {result.output_path}[/green]")
            console.print(f"  Pages: {result.page_count}")
            console.print(f"  Size: {result.file_size_bytes / 1024:.1f} KB")
            console.print(f"  Theme: {selected_theme.value}")

            if result.warnings:
                console.print("\n[yellow]Warnings:[/yellow]")
                for warning in result.warnings:
                    console.print(f"  - {warning}")
        else:
            console.print(f"\n[red]Export failed:[/red]")
            for error in result.errors:
                console.print(f"  - {error}")
            sys.exit(1)

    except Exception as e:
        console.print(f"\n[red]Export failed: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


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

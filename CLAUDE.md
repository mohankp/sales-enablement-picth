# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sales Enablement Pitch Generator - An AI-powered tool that extracts product documentation from websites and generates compelling sales pitch documents. The system handles dynamic websites (SPAs), supports incremental updates, and allows conversational refinement of generated pitches.

## Architecture

The project follows a layered architecture:

```
src/
├── acquisition/     # Web scraping & content extraction
│   ├── browser.py       # Playwright browser management
│   ├── engine.py        # Main extraction orchestrator
│   ├── spa_handler.py   # SPA-specific handling
│   ├── site_analyzer.py # Site structure detection
│   ├── fingerprint.py   # Change detection
│   └── extractors/      # Content-specific extractors
│       ├── text.py
│       ├── images.py
│       └── tables.py
├── models/          # Pydantic data models
│   ├── content.py   # Content models (PageContent, ContentBlock, etc.)
│   ├── processed.py # Processed content models (features, benefits, etc.)
│   └── config.py    # Configuration models
├── llm/             # LLM integration layer
│   ├── client.py    # Anthropic API client with retry/streaming
│   ├── config.py    # Model settings and pricing
│   ├── prompts.py   # Prompt template management
│   └── parser.py    # Structured output parsing
├── processing/      # Content understanding (LLM-powered)
│   ├── processor.py # Main content processor orchestrator
│   └── chunker.py   # Content chunking strategies
├── generation/      # Pitch generation [TODO]
├── update/          # Incremental updates [TODO]
├── refinement/      # Conversational refinement [TODO]
└── main.py          # CLI entry point
```

## Environment Setup

**Activate the virtual environment:**
```bash
source bin/activate  # On macOS/Linux
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Install Playwright browsers:**
```bash
playwright install chromium
```

## CLI Commands

The CLI is available via the `pitch-gen` command (or `python -m src.main`):

### Extract content from a website
```bash
python -m src.main extract https://example.com/product --product-id myproduct
```

Options:
- `--product-id, -p`: Identifier for storing/comparing extractions
- `--output, -o`: Output JSON file path
- `--max-pages, -m`: Maximum pages to extract (default: 20)
- `--no-download-images`: Skip image downloading
- `--no-headless`: Show browser window
- `--verbose, -v`: Enable debug logging

### Analyze a website structure
```bash
python -m src.main analyze https://example.com/product
```

### Check for content changes
```bash
python -m src.main check https://example.com/product --product-id myproduct
```

### List saved extractions
```bash
python -m src.main list-extractions --product-id myproduct
```

### Process extracted content with LLM
```bash
python -m src.main process extraction.json --output processed.json
```

Options:
- `--output, -o`: Output JSON file path for processed content
- `--product-name, -n`: Override product name
- `--aspects, -a`: Comma-separated aspects to process (features,benefits,use_cases,competitive,audience,pricing,technical,summary)
- `--model, -m`: Model to use (haiku, sonnet, opus)
- `--verbose, -v`: Enable debug logging

### Batch process multiple extractions
```bash
python -m src.main batch-process ./extractions --output ./processed --concurrency 5
```

Options:
- `--output, -o`: Output directory for processed files
- `--pattern, -p`: Glob pattern for input files (default: *.json)
- `--concurrency, -c`: Maximum concurrent processing (1-10)
- `--model, -m`: Model to use (haiku, sonnet, opus)
- `--verbose, -v`: Enable debug logging

## Development

### Running Tests
```bash
pytest tests/
```

### Key Dependencies
- **Playwright**: Browser automation for SPA handling
- **BeautifulSoup4/lxml**: HTML parsing
- **Pydantic**: Data validation and models
- **Typer/Rich**: CLI interface
- **aiohttp**: Async HTTP for image downloads
- **anthropic**: Claude API client
- **tenacity**: Retry logic for API calls

### Code Patterns

1. **Async-first**: Most operations are async for performance
2. **Configuration objects**: Use Pydantic models for all config
3. **Context managers**: Browser and engine use `async with` pattern
4. **Fingerprinting**: All content is hashed for change detection

### Example Usage in Code

**Content Extraction:**
```python
from src.acquisition.engine import ContentAcquisitionEngine
from src.models.config import SiteConfig

config = SiteConfig.from_url("https://example.com/product")

async with ContentAcquisitionEngine(config) as engine:
    content = await engine.extract(product_id="myproduct")
    print(f"Extracted {content.total_word_count} words")
```

**Content Processing:**
```python
from src.processing import ContentProcessor, ProcessingConfig

config = ProcessingConfig()

async with ContentProcessor(config) as processor:
    result = await processor.process(extracted_content)
    print(f"Found {len(result.features.features)} features")
    print(f"Executive summary: {result.summary.executive_summary}")
```

## Project Status

### Completed (Phase 1)
- [x] Content Acquisition Engine
  - Browser management with Playwright
  - SPA detection and handling
  - Site structure analysis
  - Text extraction with semantic classification
  - Image extraction and download
  - Table extraction
  - Content fingerprinting for change detection
  - CLI interface

### Completed (Phase 2)
- [x] LLM Integration Layer
  - Anthropic API client with async support
  - Retry logic and rate limiting
  - Streaming support
  - Cost tracking
  - Prompt template system with versioning
  - Structured output parsing (JSON, Pydantic models)

- [x] Content Processing Layer
  - ContentProcessor orchestrator
  - Content chunking strategies (fixed, semantic, hybrid)
  - Feature extraction and categorization
  - Benefit identification
  - Use case extraction
  - Competitive differentiator analysis
  - Audience segmentation
  - Pricing information extraction
  - Technical specifications extraction
  - Multi-level summarization
  - CLI `process` command
  - LLM result caching (memory + disk with TTL)
  - Batch processing with concurrency control
  - CLI `batch-process` command

### TODO
- [ ] Pitch Generation Engine
- [ ] Incremental Update System
- [ ] Refinement Engine (conversational)
- [ ] Output composers (PPTX, PDF)

## Key Design Decisions

1. **Quality over speed**: Multi-pass extraction with verification
2. **Change tracking**: Every content piece is fingerprinted
3. **SPA support**: Comprehensive handling of React/Vue/Angular apps
4. **Modular extractors**: Separate handlers for text, images, tables
5. **Graceful degradation**: Handles partial failures without crashing

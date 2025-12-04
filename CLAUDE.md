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
│   └── config.py    # Configuration models
├── processing/      # Content understanding (LLM) [TODO]
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

### Code Patterns

1. **Async-first**: Most operations are async for performance
2. **Configuration objects**: Use Pydantic models for all config
3. **Context managers**: Browser and engine use `async with` pattern
4. **Fingerprinting**: All content is hashed for change detection

### Example Usage in Code
```python
from src.acquisition.engine import ContentAcquisitionEngine
from src.models.config import SiteConfig

config = SiteConfig.from_url("https://example.com/product")

async with ContentAcquisitionEngine(config) as engine:
    content = await engine.extract(product_id="myproduct")
    print(f"Extracted {content.total_word_count} words")
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

### TODO
- [ ] Content Processing Layer (LLM integration)
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

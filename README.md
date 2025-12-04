# Sales Enablement Pitch Generator

An AI-powered tool that extracts product documentation from websites and generates compelling sales pitch documents. The system handles dynamic websites (SPAs), supports incremental updates, and allows conversational refinement of generated pitches.

## Features

### Phase 1: Content Acquisition (✅ Complete)

- **Browser Automation**: Playwright-based browser management with stealth mode
- **Site Structure Analysis**: Intelligent detection of navigation, tabs, and content areas
- **Hybrid Path Matching**: Handles both directory-based and file-based URL structures (e.g., `.aspx`, `.php`)
- **SPA Support**: Comprehensive handling for React, Vue, Angular, Next.js, Nuxt, and Svelte applications
- **Content Extraction**:
  - Text extraction with semantic classification (headings, paragraphs, features, benefits)
  - Image extraction and download with metadata
  - Table extraction with structured data preservation
- **Change Detection**: Content fingerprinting for incremental updates
- **CLI Interface**: Easy-to-use command-line tool for extraction and analysis

### Upcoming Phases

- [ ] **Content Processing** (LLM integration)
- [ ] **Pitch Generation Engine**
- [ ] **Incremental Update System**
- [ ] **Refinement Engine** (conversational)
- [ ] **Output Composers** (PPTX, PDF)

## Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/mohankp/sales-enablement-picth.git
cd sales-enablement-picth

# Create virtual environment
python -m venv .

# Activate virtual environment
source bin/activate  # On macOS/Linux
# or
.\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium
```

## Usage

### Quick Start

```bash
# Analyze a website structure
python -m src.main analyze https://example.com/product

# Extract content from a website
python -m src.main extract https://example.com/product --product-id myproduct
```

### Available Commands

#### Extract Content

```bash
python -m src.main extract <URL> [OPTIONS]

Options:
  -p, --product-id TEXT    Identifier for storing/comparing extractions
  -o, --output TEXT        Output JSON file path
  -m, --max-pages INTEGER  Maximum pages to extract (default: 20)
  --no-download-images     Skip image downloading
  --no-headless            Show browser window (useful for debugging)
  -v, --verbose            Enable debug logging
```

#### Analyze Website Structure

```bash
python -m src.main analyze <URL>
```

#### Check for Changes

```bash
python -m src.main check <URL> --product-id <ID>
```

#### List Extractions

```bash
python -m src.main list-extractions --product-id <ID>
```

### Examples

```bash
# Extract with verbose output and limit to 10 pages
python -m src.main extract https://example.com/product \
  -p myproduct \
  -m 10 \
  -v

# Extract and save to specific file
python -m src.main extract https://example.com/product \
  -p myproduct \
  -o output/myproduct.json

# Show browser window while extracting (debugging)
python -m src.main extract https://example.com/product \
  -p myproduct \
  --no-headless
```

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
│   ├── content.py   # Content models
│   └── config.py    # Configuration models
├── processing/      # Content understanding (LLM) [TODO]
├── generation/      # Pitch generation [TODO]
├── update/          # Incremental updates [TODO]
├── refinement/      # Conversational refinement [TODO]
└── main.py          # CLI entry point
```

## Key Design Decisions

1. **Quality over speed**: Multi-pass extraction with verification
2. **Change tracking**: Every content piece is fingerprinted
3. **SPA support**: Comprehensive handling of React/Vue/Angular apps
4. **Modular extractors**: Separate handlers for text, images, tables
5. **Graceful degradation**: Handles partial failures without crashing
6. **Hybrid path matching**: Supports both directory and file-based URL structures

## Development

### Running Tests

```bash
pytest tests/
```

### Code Structure

- **Async-first**: Most operations are async for performance
- **Configuration objects**: Use Pydantic models for all config
- **Context managers**: Browser and engine use `async with` pattern
- **Fingerprinting**: All content is hashed for change detection

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Detailed system architecture
- [CLAUDE.md](CLAUDE.md) - Development guidance for AI assistants
- [CLI_COMMANDS.md](CLI_COMMANDS.md) - Complete CLI reference

## Contributing

Contributions are welcome! Please ensure:

1. All tests pass: `pytest tests/`
2. Code follows existing patterns
3. New features include tests
4. Documentation is updated

## License

[Add your license here]

## Acknowledgments

Built with:
- [Playwright](https://playwright.dev/) - Browser automation
- [Pydantic](https://docs.pydantic.dev/) - Data validation
- [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) - HTML parsing
- [Typer](https://typer.tiangolo.com/) - CLI framework
- [Rich](https://rich.readthedocs.io/) - Terminal formatting

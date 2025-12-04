# CLI Commands Reference

## Setup

```bash
# Activate virtual environment first
source bin/activate
```

## Available Commands

### Extract Content

Extract content from a website:

```bash
python -m src.main extract <URL> [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--product-id` | `-p` | Identifier for storing/comparing extractions |
| `--output` | `-o` | Output JSON file path |
| `--max-pages` | `-m` | Maximum pages to extract (default: 20) |
| `--no-download-images` | | Skip image downloading |
| `--no-headless` | | Show browser window (useful for debugging) |
| `--verbose` | `-v` | Enable debug logging |

**Examples:**

```bash
# Basic extraction
python -m src.main extract https://in.adp.com/what-we-offer/products/adp-securtime.aspx --product-id adp-securtime

# Extract with verbose output and limit to 10 pages
python -m src.main extract https://in.adp.com/what-we-offer/products/adp-securtime.aspx \
  -p adp-securtime \
  -m 10 \
  -v

# Extract and save to specific file
python -m src.main extract https://in.adp.com/what-we-offer/products/adp-securtime.aspx \
  -p adp-securtime \
  -o output/adp-securtime.json

# Show browser window while extracting (debugging)
python -m src.main extract https://in.adp.com/what-we-offer/products/adp-securtime.aspx \
  -p adp-securtime \
  --no-headless
```

### Analyze Website Structure

Analyze a website structure without full extraction:

```bash
python -m src.main analyze <URL>
```

**Example:**

```bash
python -m src.main analyze https://in.adp.com/what-we-offer/products/adp-securtime.aspx
```

### Check for Content Changes

Check if site content has changed since last extraction:

```bash
python -m src.main check <URL> --product-id <ID>
```

**Example:**

```bash
python -m src.main check https://in.adp.com/what-we-offer/products/adp-securtime.aspx --product-id adp-securtime
```

### List Saved Extractions

List all saved extractions for a product:

```bash
python -m src.main list-extractions --product-id <ID>
```

**Example:**

```bash
python -m src.main list-extractions --product-id adp-securtime
```

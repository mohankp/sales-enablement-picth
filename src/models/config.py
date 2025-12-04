"""Configuration models for the content acquisition engine."""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, HttpUrl


class WaitStrategy(str, Enum):
    """Strategy for waiting for page content to load."""

    # Basic strategies
    LOAD = "load"  # Wait for load event
    DOMCONTENTLOADED = "domcontentloaded"  # Wait for DOMContentLoaded
    NETWORKIDLE = "networkidle"  # Wait for network to be idle

    # Advanced strategies
    SELECTOR = "selector"  # Wait for specific selector
    FUNCTION = "function"  # Wait for custom JS function
    TIMEOUT = "timeout"  # Just wait for a fixed time


class SPAConfig(BaseModel):
    """Configuration for handling Single Page Applications."""

    # Detection
    detect_framework: bool = True
    framework_hint: Optional[str] = None  # react, vue, angular, next, nuxt

    # Wait strategies
    primary_wait_strategy: WaitStrategy = WaitStrategy.NETWORKIDLE
    secondary_wait_strategy: Optional[WaitStrategy] = WaitStrategy.SELECTOR
    content_ready_selector: Optional[str] = None  # Selector indicating content loaded
    content_ready_function: Optional[str] = None  # JS function returning boolean

    # Timing
    initial_wait_ms: int = 1000  # Initial wait before checking
    max_wait_ms: int = 30000  # Maximum wait time
    network_idle_timeout_ms: int = 500  # Time with no network activity
    poll_interval_ms: int = 100  # How often to check conditions

    # Hydration handling
    wait_for_hydration: bool = True
    hydration_indicators: list[str] = Field(
        default_factory=lambda: [
            "[data-reactroot]",
            "[data-v-app]",
            "[ng-version]",
            "#__next",
            "#__nuxt",
        ]
    )

    # Dynamic content handling
    scroll_to_load: bool = True
    max_scroll_iterations: int = 10
    scroll_pause_ms: int = 500

    # Tab/accordion handling
    expand_tabs: bool = True
    expand_accordions: bool = True
    click_delay_ms: int = 300


class BrowserConfig(BaseModel):
    """Configuration for the Playwright browser instance."""

    # Browser selection
    browser_type: str = "chromium"  # chromium, firefox, webkit
    headless: bool = True
    slow_mo: int = 0  # Slow down operations by this many ms

    # Viewport
    viewport_width: int = 1920
    viewport_height: int = 1080
    device_scale_factor: float = 1.0
    is_mobile: bool = False

    # User agent
    user_agent: Optional[str] = None
    locale: str = "en-US"
    timezone_id: str = "America/New_York"

    # Network
    ignore_https_errors: bool = True
    block_resources: list[str] = Field(
        default_factory=lambda: []  # e.g., ["image", "font"] to speed up
    )
    extra_http_headers: dict[str, str] = Field(default_factory=dict)

    # Storage
    storage_state: Optional[str] = None  # Path to saved auth state
    accept_downloads: bool = True
    downloads_path: Optional[str] = None

    # Timeouts
    navigation_timeout_ms: int = 60000
    default_timeout_ms: int = 30000

    # Anti-detection
    stealth_mode: bool = True  # Apply anti-bot detection measures

    # Resources
    java_script_enabled: bool = True
    bypass_csp: bool = False  # Bypass Content Security Policy

    # Proxy
    proxy_server: Optional[str] = None
    proxy_username: Optional[str] = None
    proxy_password: Optional[str] = None


class ExtractionConfig(BaseModel):
    """Configuration for content extraction behavior."""

    # Content selection
    include_navigation: bool = False
    include_footer: bool = False
    include_sidebar: bool = True
    include_comments: bool = False
    include_ads: bool = False

    # Media handling
    download_images: bool = True
    download_videos: bool = False  # Usually just get metadata
    max_image_size_mb: float = 10.0
    min_image_width: int = 50  # Skip tiny images (icons, etc.)
    min_image_height: int = 50
    image_formats: list[str] = Field(
        default_factory=lambda: ["jpg", "jpeg", "png", "webp", "gif", "svg"]
    )

    # Text processing
    extract_structured_data: bool = True  # JSON-LD, microdata, etc.
    preserve_formatting: bool = True
    extract_code_blocks: bool = True
    max_text_length: int = 1000000  # Max chars per page

    # Link handling
    follow_internal_links: bool = False  # For multi-page extraction
    max_depth: int = 2  # How deep to follow links
    max_pages: int = 50  # Maximum pages to extract
    exclude_patterns: list[str] = Field(
        default_factory=lambda: [
            r"/blog/",
            r"/news/",
            r"/careers/",
            r"/legal/",
            r"/privacy",
            r"/terms",
        ]
    )
    include_patterns: list[str] = Field(
        default_factory=lambda: [
            r"/features",
            r"/product",
            r"/solutions",
            r"/pricing",
            r"/docs",
            r"/documentation",
        ]
    )

    # Quality thresholds
    min_content_length: int = 100  # Minimum chars to consider a valid extraction
    confidence_threshold: float = 0.7  # Minimum confidence for content blocks

    # Retry behavior
    max_retries: int = 3
    retry_delay_ms: int = 1000
    retry_on_empty: bool = True

    # Cookie/consent handling
    dismiss_cookie_banners: bool = True
    cookie_banner_selectors: list[str] = Field(
        default_factory=lambda: [
            "[class*='cookie'] button",
            "[class*='consent'] button",
            "[id*='cookie'] button",
            "[id*='consent'] button",
            ".cc-btn",
            "#onetrust-accept-btn-handler",
            "[data-testid*='cookie'] button",
        ]
    )
    cookie_dismiss_text: list[str] = Field(
        default_factory=lambda: [
            "accept",
            "agree",
            "ok",
            "got it",
            "allow",
            "continue",
        ]
    )


class SiteConfig(BaseModel):
    """
    Complete configuration for extracting from a specific site.

    Combines all configuration aspects for a targeted extraction.
    """

    # Target
    url: str
    additional_urls: list[str] = Field(default_factory=list)
    name: Optional[str] = None  # Friendly name for the product/site

    # Component configs
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    spa: SPAConfig = Field(default_factory=SPAConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)

    # Site-specific selectors (overrides auto-detection)
    custom_selectors: dict[str, str] = Field(default_factory=dict)
    # Example: {"main_content": "article.main", "features": ".feature-list"}

    # Site-specific JavaScript to execute
    pre_extraction_scripts: list[str] = Field(default_factory=list)
    post_load_scripts: list[str] = Field(default_factory=list)

    # Authentication (if needed)
    requires_auth: bool = False
    auth_url: Optional[str] = None
    auth_username: Optional[str] = None
    auth_password: Optional[str] = None
    auth_selectors: dict[str, str] = Field(default_factory=dict)

    # Rate limiting
    requests_per_second: float = 2.0
    delay_between_pages_ms: int = 500

    # Caching
    use_cache: bool = True
    cache_ttl_hours: int = 24

    @classmethod
    def from_url(cls, url: str, **kwargs: Any) -> "SiteConfig":
        """Create a basic config from just a URL."""
        return cls(url=url, **kwargs)

    @classmethod
    def for_spa(cls, url: str, content_selector: str, **kwargs: Any) -> "SiteConfig":
        """Create a config optimized for SPA sites."""
        spa_config = SPAConfig(
            content_ready_selector=content_selector,
            wait_for_hydration=True,
            expand_tabs=True,
            expand_accordions=True,
        )
        return cls(url=url, spa=spa_config, **kwargs)

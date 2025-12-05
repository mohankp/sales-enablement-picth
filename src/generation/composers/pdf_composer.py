"""PDF composer for generating documents from pitch content."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Table,
    TableStyle,
    Image,
    ListFlowable,
    ListItem,
    KeepTogether,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from .base import BaseComposer, ComposerConfig, ComposerResult, ThemeColor
from ...models.pitch import Pitch, PitchSection, SectionType


@dataclass
class PDFConfig(ComposerConfig):
    """Configuration for PDF composer."""

    # Page settings
    page_size: str = "letter"  # "letter" or "a4"

    # Margins (inches)
    margin_left: float = 1.0
    margin_right: float = 1.0
    margin_top: float = 1.0
    margin_bottom: float = 1.0

    # Font settings
    title_font_size: int = 28
    heading_font_size: int = 18
    subheading_font_size: int = 14
    body_font_size: int = 11

    # Content settings
    include_table_of_contents: bool = True
    include_executive_summary: bool = True
    include_elevator_pitch: bool = True
    include_key_messages: bool = True
    include_feature_highlights: bool = True
    include_benefit_statements: bool = True
    include_competitive_points: bool = True
    include_objection_handling: bool = True

    # Style settings
    justified_text: bool = True
    section_page_breaks: bool = False


class PDFComposer(BaseComposer):
    """Composer for generating PDF documents from pitch content."""

    def __init__(self, config: Optional[PDFConfig] = None):
        super().__init__(config or PDFConfig())
        self.config: PDFConfig = self.config  # type: ignore
        self._styles: dict = {}
        self._page_count = 0

    def compose(self, pitch: Pitch, output_path: Optional[Path] = None) -> ComposerResult:
        """Generate a PDF document from a pitch."""
        warnings: list[str] = []
        errors: list[str] = []

        try:
            # Resolve output path
            path = self._resolve_output_path(output_path, ".pdf")

            # Get page size
            page_size = A4 if self.config.page_size.lower() == "a4" else letter

            # Create document
            doc = SimpleDocTemplate(
                str(path),
                pagesize=page_size,
                leftMargin=self.config.margin_left * inch,
                rightMargin=self.config.margin_right * inch,
                topMargin=self.config.margin_top * inch,
                bottomMargin=self.config.margin_bottom * inch,
            )

            # Initialize styles
            self._init_styles()

            # Build content
            story = []

            # Title page
            story.extend(self._create_title_page(pitch))
            story.append(PageBreak())

            # Table of contents (if enabled)
            if self.config.include_table_of_contents:
                story.extend(self._create_table_of_contents(pitch))
                story.append(PageBreak())

            # Executive summary
            if self.config.include_executive_summary:
                story.extend(self._create_executive_summary(pitch))
                story.append(Spacer(1, 0.3 * inch))

            # Elevator pitch
            if self.config.include_elevator_pitch and pitch.elevator_pitch:
                story.extend(self._create_elevator_pitch_section(pitch))
                story.append(Spacer(1, 0.3 * inch))

            # Main sections
            for section in sorted(pitch.sections, key=lambda s: s.order):
                story.extend(self._create_section(section, warnings))
                if self.config.section_page_breaks:
                    story.append(PageBreak())
                else:
                    story.append(Spacer(1, 0.4 * inch))

            # Key messages
            if self.config.include_key_messages and pitch.key_messages:
                story.extend(self._create_key_messages(pitch))
                story.append(Spacer(1, 0.3 * inch))

            # Feature highlights
            if self.config.include_feature_highlights and pitch.feature_highlights:
                story.extend(self._create_feature_highlights(pitch))
                story.append(Spacer(1, 0.3 * inch))

            # Benefit statements
            if self.config.include_benefit_statements and pitch.benefit_statements:
                story.extend(self._create_benefit_statements(pitch))
                story.append(Spacer(1, 0.3 * inch))

            # Competitive points
            if self.config.include_competitive_points and pitch.competitive_points:
                story.extend(self._create_competitive_points(pitch))
                story.append(Spacer(1, 0.3 * inch))

            # Objection handling
            if self.config.include_objection_handling and pitch.common_objections:
                story.extend(self._create_objection_handling(pitch))
                story.append(Spacer(1, 0.3 * inch))

            # Call to action
            if pitch.call_to_action:
                story.extend(self._create_call_to_action(pitch))

            # Build PDF
            doc.build(story, onFirstPage=self._add_page_number, onLaterPages=self._add_page_number)

            # Get file size and page count
            file_size = path.stat().st_size

            return ComposerResult(
                success=True,
                output_path=path,
                file_size_bytes=file_size,
                page_count=doc.page,
                warnings=warnings,
                metadata={
                    "format": "pdf",
                    "theme": self.config.theme.value,
                    "page_size": self.config.page_size,
                },
            )

        except Exception as e:
            errors.append(f"Failed to generate PDF: {str(e)}")
            return ComposerResult(
                success=False,
                errors=errors,
                warnings=warnings,
            )

    def _init_styles(self) -> None:
        """Initialize paragraph styles based on theme."""
        palette = self.config.get_palette()
        base_styles = getSampleStyleSheet()

        # Convert hex colors to reportlab colors
        primary_color = colors.HexColor(palette["primary"])
        accent_color = colors.HexColor(palette["accent"])
        text_color = colors.HexColor(palette["text"])
        text_light_color = colors.HexColor(palette["text_light"])

        alignment = TA_JUSTIFY if self.config.justified_text else TA_LEFT

        self._styles = {
            "title": ParagraphStyle(
                "Title",
                parent=base_styles["Title"],
                fontSize=self.config.title_font_size,
                textColor=primary_color,
                alignment=TA_CENTER,
                spaceAfter=12,
            ),
            "subtitle": ParagraphStyle(
                "Subtitle",
                parent=base_styles["Normal"],
                fontSize=self.config.subheading_font_size,
                textColor=text_light_color,
                alignment=TA_CENTER,
                spaceAfter=24,
            ),
            "heading1": ParagraphStyle(
                "Heading1",
                parent=base_styles["Heading1"],
                fontSize=self.config.heading_font_size,
                textColor=primary_color,
                spaceBefore=16,
                spaceAfter=12,
            ),
            "heading2": ParagraphStyle(
                "Heading2",
                parent=base_styles["Heading2"],
                fontSize=self.config.subheading_font_size,
                textColor=primary_color,
                spaceBefore=12,
                spaceAfter=8,
            ),
            "body": ParagraphStyle(
                "Body",
                parent=base_styles["Normal"],
                fontSize=self.config.body_font_size,
                textColor=text_color,
                alignment=alignment,
                spaceAfter=8,
                leading=14,
            ),
            "body_light": ParagraphStyle(
                "BodyLight",
                parent=base_styles["Normal"],
                fontSize=self.config.body_font_size - 1,
                textColor=text_light_color,
                alignment=alignment,
                spaceAfter=6,
            ),
            "bullet": ParagraphStyle(
                "Bullet",
                parent=base_styles["Normal"],
                fontSize=self.config.body_font_size,
                textColor=text_color,
                leftIndent=20,
                spaceAfter=4,
            ),
            "quote": ParagraphStyle(
                "Quote",
                parent=base_styles["Normal"],
                fontSize=self.config.body_font_size + 1,
                textColor=text_color,
                leftIndent=30,
                rightIndent=30,
                fontName="Times-Italic",
                alignment=TA_CENTER,
                spaceAfter=12,
            ),
            "highlight": ParagraphStyle(
                "Highlight",
                parent=base_styles["Normal"],
                fontSize=self.config.body_font_size,
                textColor=accent_color,
                fontName="Helvetica-Bold",
                spaceBefore=6,
                spaceAfter=4,
            ),
            "toc_entry": ParagraphStyle(
                "TOCEntry",
                parent=base_styles["Normal"],
                fontSize=self.config.body_font_size,
                textColor=text_color,
                leftIndent=20,
                spaceAfter=8,
            ),
        }

    def _create_title_page(self, pitch: Pitch) -> list:
        """Create the title page."""
        elements = []

        # Add vertical space
        elements.append(Spacer(1, 2 * inch))

        # Title
        title_text = pitch.title
        if len(title_text) > 100:
            title_text = title_text[:97] + "..."
        elements.append(Paragraph(title_text, self._styles["title"]))

        # Subtitle
        if pitch.subtitle:
            elements.append(Paragraph(pitch.subtitle, self._styles["subtitle"]))

        elements.append(Spacer(1, 0.5 * inch))

        # Product name and URL
        elements.append(Paragraph(pitch.product_name, self._styles["heading2"]))
        elements.append(
            Paragraph(
                f'<link href="{pitch.product_url}">{pitch.product_url}</link>',
                self._styles["body_light"],
            )
        )

        elements.append(Spacer(1, 1 * inch))

        # One-liner
        if pitch.one_liner:
            elements.append(Paragraph(f'"{pitch.one_liner}"', self._styles["quote"]))

        # Metadata at bottom
        elements.append(Spacer(1, 2 * inch))
        meta_text = f"Generated: {pitch.generated_at.strftime('%Y-%m-%d')}"
        if pitch.config.target_audience:
            meta_text += f" | Audience: {pitch.config.target_audience}"
        meta_text += f" | Tone: {pitch.config.tone.value.title()}"
        elements.append(Paragraph(meta_text, self._styles["body_light"]))

        return elements

    def _create_table_of_contents(self, pitch: Pitch) -> list:
        """Create a table of contents."""
        elements = []

        elements.append(Paragraph("Table of Contents", self._styles["heading1"]))
        elements.append(Spacer(1, 0.2 * inch))

        # Standard sections
        toc_items = []
        if self.config.include_executive_summary:
            toc_items.append("Executive Summary")
        if self.config.include_elevator_pitch and pitch.elevator_pitch:
            toc_items.append("The 30-Second Pitch")

        # Main sections
        for section in sorted(pitch.sections, key=lambda s: s.order):
            toc_items.append(section.title)

        # Additional sections
        if self.config.include_key_messages and pitch.key_messages:
            toc_items.append("Key Messages")
        if self.config.include_feature_highlights and pitch.feature_highlights:
            toc_items.append("Feature Highlights")
        if self.config.include_benefit_statements and pitch.benefit_statements:
            toc_items.append("Benefit Statements")
        if self.config.include_competitive_points and pitch.competitive_points:
            toc_items.append("Competitive Advantages")
        if self.config.include_objection_handling and pitch.common_objections:
            toc_items.append("Objection Handling")
        if pitch.call_to_action:
            toc_items.append("Next Steps")

        for i, item in enumerate(toc_items, 1):
            elements.append(
                Paragraph(f"{i}. {item}", self._styles["toc_entry"])
            )

        return elements

    def _create_executive_summary(self, pitch: Pitch) -> list:
        """Create the executive summary section."""
        elements = []

        elements.append(Paragraph("Executive Summary", self._styles["heading1"]))
        elements.append(Paragraph(pitch.executive_summary, self._styles["body"]))

        return elements

    def _create_elevator_pitch_section(self, pitch: Pitch) -> list:
        """Create the elevator pitch section."""
        elements = []

        elements.append(Paragraph("The 30-Second Pitch", self._styles["heading1"]))
        elements.append(Paragraph(f'"{pitch.elevator_pitch}"', self._styles["quote"]))

        return elements

    def _create_section(self, section: PitchSection, warnings: list[str]) -> list:
        """Create a content section."""
        elements = []

        # Section title
        elements.append(Paragraph(section.title, self._styles["heading1"]))

        # Main content
        if section.content:
            # Split into paragraphs if content is long
            paragraphs = section.content.split("\n\n")
            for para in paragraphs:
                if para.strip():
                    elements.append(Paragraph(para.strip(), self._styles["body"]))

        # Key points as bullet list
        if section.key_points:
            elements.append(Spacer(1, 0.15 * inch))
            bullet_items = []
            for point in section.key_points:
                # Truncate very long points
                if len(point) > 500:
                    point = point[:497] + "..."
                bullet_items.append(
                    ListItem(Paragraph(point, self._styles["bullet"]))
                )

            elements.append(
                ListFlowable(
                    bullet_items,
                    bulletType="bullet",
                    start="•",
                    leftIndent=15,
                    bulletOffsetY=-2,
                )
            )

        # Visual assets (images only for PDF)
        if self.config.include_visual_assets and section.visual_assets:
            for va in section.visual_assets:
                if va.asset_type == "image" and va.local_path:
                    img_path = self._load_image_if_exists(va.local_path)
                    if img_path:
                        try:
                            elements.append(Spacer(1, 0.2 * inch))
                            img = Image(str(img_path), width=4 * inch, height=3 * inch)
                            img.hAlign = "CENTER"
                            elements.append(img)
                            if va.caption:
                                elements.append(
                                    Paragraph(va.caption, self._styles["body_light"])
                                )
                        except Exception as e:
                            warnings.append(f"Could not add image to PDF: {e}")

        return elements

    def _create_key_messages(self, pitch: Pitch) -> list:
        """Create the key messages section."""
        elements = []

        elements.append(Paragraph("Key Messages", self._styles["heading1"]))
        elements.append(
            Paragraph(
                "Remember these core messages when discussing the product:",
                self._styles["body_light"],
            )
        )

        bullet_items = []
        for msg in pitch.key_messages:
            bullet_items.append(
                ListItem(Paragraph(f"<b>{msg}</b>", self._styles["bullet"]))
            )

        elements.append(
            ListFlowable(
                bullet_items,
                bulletType="bullet",
                start="•",
                leftIndent=15,
            )
        )

        return elements

    def _create_feature_highlights(self, pitch: Pitch) -> list:
        """Create the feature highlights section."""
        elements = []

        elements.append(Paragraph("Feature Highlights", self._styles["heading1"]))

        for feature in pitch.feature_highlights:
            # Feature name as subheading
            elements.append(Paragraph(feature.name, self._styles["heading2"]))

            # Headline
            elements.append(
                Paragraph(f"<i>{feature.headline}</i>", self._styles["body"])
            )

            # Description
            elements.append(Paragraph(feature.description, self._styles["body"]))

            # Benefit
            elements.append(
                Paragraph(
                    f"<b>Benefit:</b> {feature.benefit}",
                    self._styles["body"],
                )
            )

            # Proof point if available
            if feature.proof_point:
                elements.append(
                    Paragraph(
                        f"<b>Evidence:</b> {feature.proof_point}",
                        self._styles["body_light"],
                    )
                )

            elements.append(Spacer(1, 0.15 * inch))

        return elements

    def _create_benefit_statements(self, pitch: Pitch) -> list:
        """Create the benefit statements section."""
        elements = []

        elements.append(Paragraph("Benefit Statements", self._styles["heading1"]))

        for benefit in pitch.benefit_statements:
            elements.append(Paragraph(benefit.headline, self._styles["highlight"]))
            elements.append(Paragraph(benefit.description, self._styles["body"]))

            meta_parts = []
            if benefit.supporting_feature:
                meta_parts.append(f"Feature: {benefit.supporting_feature}")
            if benefit.target_audience:
                meta_parts.append(f"Audience: {benefit.target_audience}")

            if meta_parts:
                elements.append(
                    Paragraph(" | ".join(meta_parts), self._styles["body_light"])
                )

            elements.append(Spacer(1, 0.1 * inch))

        return elements

    def _create_competitive_points(self, pitch: Pitch) -> list:
        """Create the competitive advantages section."""
        elements = []

        elements.append(Paragraph("Competitive Advantages", self._styles["heading1"]))

        for point in pitch.competitive_points:
            elements.append(Paragraph(point.claim, self._styles["highlight"]))
            elements.append(Paragraph(point.explanation, self._styles["body"]))

            if point.compared_to:
                elements.append(
                    Paragraph(
                        f"<i>Compared to: {point.compared_to}</i>",
                        self._styles["body_light"],
                    )
                )

            elements.append(Spacer(1, 0.1 * inch))

        return elements

    def _create_objection_handling(self, pitch: Pitch) -> list:
        """Create the objection handling section."""
        elements = []

        elements.append(Paragraph("Objection Handling", self._styles["heading1"]))
        elements.append(
            Paragraph(
                "Common objections and recommended responses:",
                self._styles["body_light"],
            )
        )
        elements.append(Spacer(1, 0.15 * inch))

        for objection, response in pitch.common_objections.items():
            elements.append(
                Paragraph(f'<b>Objection:</b> "{objection}"', self._styles["body"])
            )
            elements.append(
                Paragraph(f"<b>Response:</b> {response}", self._styles["body"])
            )
            elements.append(Spacer(1, 0.15 * inch))

        return elements

    def _create_call_to_action(self, pitch: Pitch) -> list:
        """Create the call to action section."""
        elements = []
        cta = pitch.call_to_action

        elements.append(Paragraph("Next Steps", self._styles["heading1"]))

        # Primary CTA
        elements.append(Paragraph(cta.primary_cta, self._styles["highlight"]))

        # Secondary CTA
        if cta.secondary_cta:
            elements.append(Paragraph(cta.secondary_cta, self._styles["body"]))

        # Urgency statement
        if cta.urgency_statement:
            elements.append(
                Paragraph(f"<i>{cta.urgency_statement}</i>", self._styles["body"])
            )

        # Next steps list
        if cta.next_steps:
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(Paragraph("Action Items:", self._styles["heading2"]))

            bullet_items = []
            for i, step in enumerate(cta.next_steps, 1):
                bullet_items.append(
                    ListItem(Paragraph(f"{step}", self._styles["bullet"]))
                )

            elements.append(
                ListFlowable(
                    bullet_items,
                    bulletType="1",
                    leftIndent=15,
                )
            )

        return elements

    def _add_page_number(self, canvas, doc) -> None:
        """Add page number to the canvas."""
        palette = self.config.get_palette()
        canvas.saveState()
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(colors.HexColor(palette["text_light"]))
        page_num = canvas.getPageNumber()
        text = f"Page {page_num}"
        canvas.drawRightString(
            doc.pagesize[0] - self.config.margin_right * inch,
            self.config.margin_bottom * inch / 2,
            text,
        )
        canvas.restoreState()

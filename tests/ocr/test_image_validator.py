import io
import pytest
import fitz
from PIL import Image
from app.modules.ocr.infra.validation.image_validator import ImageValidator, DocumentType


def create_digital_pdf_bytes() -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=300, height=400)
    page.insert_text((50, 50), "Este es un PDF digital con texto seleccionable.")
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


def create_scanned_pdf_bytes(img_width: int = 100, img_height: int = 100, color: str = "white") -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=300, height=400)  # Physical dimensions: 300x400 pt
    
    # Create an image to insert (no selectable text)
    img = Image.new("RGB", (img_width, img_height), color=color)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    
    page.insert_image(page.rect, stream=img_bytes.getvalue())
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


@pytest.mark.asyncio
async def test_validate_digital_pdf_passes_automatically():
    validator = ImageValidator()
    pdf_bytes = create_digital_pdf_bytes()
    
    # A digital PDF (has selectable text) should bypass all image validation checks and pass
    result = await validator.validate_poliza_pdf(pdf_bytes)
    
    assert result.is_valid is True
    assert len(result.errors) == 0


@pytest.mark.asyncio
async def test_validate_scanned_pdf_scales_dpi_and_ignores_overexposure():
    validator = ImageValidator()
    # Create a pure white scanned PDF. If it rendered at 200 DPI, it would be 833x1111 px
    # (which fails the 1000px minimum width requirement for policies).
    # With dynamic DPI, it should scale up the DPI to meet min resolution (1000x700).
    # Also, it is white, but since it is a PDF, it should ignore the overexposure check.
    pdf_bytes = create_scanned_pdf_bytes(img_width=200, img_height=200, color="white")
    
    result = await validator.validate_poliza_pdf(pdf_bytes)
    
    # It might still fail sharpness check since it's a solid plain color image,
    # but it should NOT fail the resolution or overexposure/brightness checks!
    # Let's inspect the errors to confirm resolution and brightness are not listed.
    res_errors = [e for e in result.errors if "Resolucion" in e or "sobreexpuesta" in e]
    assert len(res_errors) == 0


@pytest.mark.asyncio
async def test_validate_image_retains_overexposure_check():
    validator = ImageValidator()
    
    # An actual image (not a PDF) that is pure white should fail the overexposure check
    img = Image.new("RGB", (1200, 800), color="white")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    
    result = await validator.validate_ine_image(img_bytes.getvalue(), "image/jpeg")
    
    assert result.is_valid is False
    # Should fail because of overexposure (pure white has brightness ~255 > 220)
    assert any("sobreexpuesta" in e for e in result.errors)

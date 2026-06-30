from dataclasses import dataclass
from datetime import datetime


@dataclass
class OCRDocument:
    id: str
    filename: str
    text: str
    page_count: int
    created_at: datetime

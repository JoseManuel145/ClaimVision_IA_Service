from pydantic import BaseModel


class OcrResponse(BaseModel):
    id: str
    filename: str
    text: str
    page_count: int
    created_at: str


class OcrHistoryItem(BaseModel):
    id: str
    filename: str
    page_count: int
    created_at: str


class OcrHistoryResponse(BaseModel):
    data: list[OcrHistoryItem]
    total: int
    page: int
    limit: int

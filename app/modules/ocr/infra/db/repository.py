from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from app.modules.ocr.domain.models import OCRDocument
from app.modules.ocr.infra.db.tables import OCRDocumentTable


class PostgresOCRDocumentRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def save(self, doc: OCRDocument) -> OCRDocument:
        row = OCRDocumentTable(
            id=doc.id,
            filename=doc.filename,
            text=doc.text,
            page_count=doc.page_count,
            created_at=doc.created_at,
        )
        self._session.add(row)
        await self._session.commit()
        return doc

    async def list_paginated(
        self, page: int, limit: int
    ) -> tuple[list[OCRDocument], int]:
        offset = (page - 1) * limit
        total_q = select(func.count(OCRDocumentTable.id))
        total_result = await self._session.execute(total_q)
        total = total_result.scalar_one()

        q = (
            select(OCRDocumentTable)
            .order_by(OCRDocumentTable.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self._session.execute(q)
        rows = result.scalars().all()

        docs = [
            OCRDocument(
                id=r.id,
                filename=r.filename,
                text=r.text,
                page_count=r.page_count,
                created_at=r.created_at,
            )
            for r in rows
        ]
        return docs, total

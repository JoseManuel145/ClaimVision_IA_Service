from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from app.modules.nosupervised.domain.models import Inference
from app.modules.nosupervised.domain.ports import InferenceRepository
from app.modules.nosupervised.infra.db.tables import InferenceTable


class PostgresInferenceRepository(InferenceRepository):
    def __init__(self, session: AsyncSession):
        self._session = session

    async def save(self, inference: Inference) -> Inference:
        row = InferenceTable(
            id=inference.id,
            filename=inference.filename,
            cluster_id=inference.cluster_id,
            tipo_dano=inference.tipo_dano,
            severidad=inference.severidad,
            confianza=inference.confianza,
            distancia_centroide=inference.distancia_centroide,
            created_at=inference.created_at,
        )
        self._session.add(row)
        await self._session.commit()
        return inference

    async def list_paginated(
        self, page: int, limit: int
    ) -> tuple[list[Inference], int]:
        offset = (page - 1) * limit
        total_q = select(func.count(InferenceTable.id))
        total_result = await self._session.execute(total_q)
        total = total_result.scalar_one()

        q = (
            select(InferenceTable)
            .order_by(InferenceTable.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self._session.execute(q)
        rows = result.scalars().all()

        inferences = [
            Inference(
                id=r.id,
                filename=r.filename,
                cluster_id=r.cluster_id,
                tipo_dano=r.tipo_dano,
                severidad=r.severidad,
                confianza=r.confianza,
                distancia_centroide=r.distancia_centroide,
                created_at=r.created_at,
            )
            for r in rows
        ]
        return inferences, total

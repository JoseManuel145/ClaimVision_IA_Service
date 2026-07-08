import uuid
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from app.modules.nlp.domain.models import VozTranscripcion, DamageEntity
from app.modules.nlp.infra.db.tables import NlpTranscripcionTable, NlpDamageEntityTable


class PostgresVozRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def save(self, transcripcion: VozTranscripcion) -> VozTranscripcion:
        row = NlpTranscripcionTable(
            id=transcripcion.id,
            filename=transcripcion.filename,
            texto=transcripcion.texto,
            duracion_seg=transcripcion.duracion_seg,
            created_at=transcripcion.created_at,
        )
        self._session.add(row)
        for ent in transcripcion.entidades:
            ent_row = NlpDamageEntityTable(
                id=str(uuid.uuid4()),
                transcripcion_id=transcripcion.id,
                tipo_dano=ent.tipo_dano,
                severidad=ent.severidad,
                parte_afectada=ent.parte_afectada,
                sintoma=ent.sintoma,
                confianza=ent.confianza,
            )
            self._session.add(ent_row)
        await self._session.commit()
        return transcripcion

    async def list_paginated(
        self, page: int, limit: int
    ) -> tuple[list[VozTranscripcion], int]:
        offset = (page - 1) * limit
        total_q = select(func.count(NlpTranscripcionTable.id))
        total_result = await self._session.execute(total_q)
        total = total_result.scalar_one()

        q = (
            select(NlpTranscripcionTable)
            .order_by(NlpTranscripcionTable.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self._session.execute(q)
        rows = result.scalars().all()

        docs = []
        for r in rows:
            entidades = await self._get_entities(r.id)
            docs.append(
                VozTranscripcion(
                    id=r.id,
                    filename=r.filename,
                    texto=r.texto,
                    duracion_seg=r.duracion_seg,
                    entidades=entidades,
                    created_at=r.created_at,
                )
            )
        return docs, total

    async def get_by_id(self, id: str) -> VozTranscripcion | None:
        q = select(NlpTranscripcionTable).where(NlpTranscripcionTable.id == id)
        result = await self._session.execute(q)
        row = result.scalar_one_or_none()
        if row is None:
            return None
        entidades = await self._get_entities(row.id)
        return VozTranscripcion(
            id=row.id,
            filename=row.filename,
            texto=row.texto,
            duracion_seg=row.duracion_seg,
            entidades=entidades,
            created_at=row.created_at,
        )

    async def _get_entities(self, transcripcion_id: str) -> list[DamageEntity]:
        q = select(NlpDamageEntityTable).where(
            NlpDamageEntityTable.transcripcion_id == transcripcion_id
        )
        result = await self._session.execute(q)
        rows = result.scalars().all()
        return [
            DamageEntity(
                tipo_dano=r.tipo_dano,
                severidad=r.severidad,
                parte_afectada=r.parte_afectada,
                sintoma=r.sintoma,
                confianza=r.confianza,
            )
            for r in rows
        ]

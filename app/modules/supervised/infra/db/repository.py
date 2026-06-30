from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from app.modules.supervised.domain.models import V2Prediction, RetrainJob
from app.modules.supervised.infra.db.tables import V2PredictionTable, V2RetrainJobTable


class PostgresV2PredictionRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def save(self, pred: V2Prediction) -> V2Prediction:
        row = V2PredictionTable(
            id=pred.id,
            filename=pred.filename,
            class_id=pred.class_id,
            tipo_dano=pred.tipo_dano,
            severidad=pred.severidad,
            confianza=pred.confianza,
            prob_dist=pred.prob_dist,
            created_at=pred.created_at,
        )
        self._session.add(row)
        await self._session.commit()
        return pred

    async def list_paginated(
        self, page: int, limit: int
    ) -> tuple[list[V2Prediction], int]:
        offset = (page - 1) * limit
        total_q = select(func.count(V2PredictionTable.id))
        total_result = await self._session.execute(total_q)
        total = total_result.scalar_one()

        q = (
            select(V2PredictionTable)
            .order_by(V2PredictionTable.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self._session.execute(q)
        rows = result.scalars().all()

        preds = [
            V2Prediction(
                id=r.id,
                filename=r.filename,
                class_id=r.class_id,
                tipo_dano=r.tipo_dano,
                severidad=r.severidad,
                confianza=r.confianza,
                prob_dist=r.prob_dist,
                created_at=r.created_at,
            )
            for r in rows
        ]
        return preds, total


class PostgresRetrainJobRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def save(self, job: RetrainJob) -> RetrainJob:
        row = V2RetrainJobTable(
            id=job.id,
            status=job.status,
            total_epochs=job.total_epochs,
            current_epoch=job.current_epoch,
            best_accuracy=job.best_accuracy,
            loss_history=job.loss_history,
            error=job.error,
            created_at=job.created_at,
            completed_at=job.completed_at,
        )
        self._session.add(row)
        await self._session.commit()
        return job

    async def get_by_id(self, job_id: str) -> Optional[RetrainJob]:
        q = select(V2RetrainJobTable).where(V2RetrainJobTable.id == job_id)
        result = await self._session.execute(q)
        row = result.scalar_one_or_none()
        if not row:
            return None
        return RetrainJob(
            id=row.id,
            status=row.status,
            total_epochs=row.total_epochs,
            current_epoch=row.current_epoch,
            best_accuracy=row.best_accuracy,
            loss_history=row.loss_history,
            error=row.error,
            created_at=row.created_at,
            completed_at=row.completed_at,
        )

    async def update(self, job: RetrainJob) -> RetrainJob:
        q = select(V2RetrainJobTable).where(V2RetrainJobTable.id == job.id)
        result = await self._session.execute(q)
        row = result.scalar_one_or_none()
        if not row:
            return job
        row.status = job.status
        row.current_epoch = job.current_epoch
        row.best_accuracy = job.best_accuracy
        row.loss_history = job.loss_history
        row.error = job.error
        row.completed_at = job.completed_at
        await self._session.commit()
        return job

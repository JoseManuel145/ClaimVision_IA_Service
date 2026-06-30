import asyncio
import json
import shutil
import tempfile
from uuid import uuid4
from datetime import datetime, timezone
from pathlib import Path

from app.modules.supervised.domain.models import RetrainJob
from app.modules.supervised.domain.ports import ClassifierService, RetrainJobRepository


class V2RetrainUseCase:
    def __init__(
        self,
        classifier: ClassifierService,
        job_repo: RetrainJobRepository,
    ):
        self._classifier = classifier
        self._job_repo = job_repo

    async def start_retrain(
        self,
        labels: list[dict],
        files: list[tuple[str, bytes]],
        epochs: int = 40,
        lr: float = 0.001,
    ) -> RetrainJob:
        job = RetrainJob(
            id=str(uuid4()),
            status="pending",
            total_epochs=epochs,
            current_epoch=0,
            best_accuracy=0.0,
            loss_history=[],
            error=None,
            created_at=datetime.now(timezone.utc),
            completed_at=None,
        )
        await self._job_repo.save(job)

        data_dir = tempfile.mkdtemp(prefix="claimvision_retrain_")
        labels_path = Path(data_dir) / "labels.json"
        with open(labels_path, "w") as f:
            json.dump(labels, f)
        for fname, content in files:
            with open(Path(data_dir) / fname, "wb") as f:
                f.write(content)

        job.status = "training"
        await self._job_repo.update(job)

        loop = asyncio.get_event_loop()

        def on_epoch_end(job_id: str, epoch: int, acc: float, loss: float, is_best: bool):
            async def _update():
                current_job = await self._job_repo.get_by_id(job_id)
                if current_job:
                    current_job.current_epoch = epoch
                    current_job.loss_history.append(round(loss, 6))
                    if is_best:
                        current_job.best_accuracy = round(acc, 4)
                    await self._job_repo.update(current_job)
            asyncio.run_coroutine_threadsafe(_update(), loop)

        def _train():
            try:
                self._classifier.retrain(data_dir, epochs, lr, job.id, on_epoch_end)
                async def _complete():
                    done_job = await self._job_repo.get_by_id(job.id)
                    if done_job:
                        done_job.status = "completed"
                        done_job.completed_at = datetime.now(timezone.utc)
                        await self._job_repo.update(done_job)
                asyncio.run_coroutine_threadsafe(_complete(), loop)
            except Exception as e:
                async def _fail():
                    failed_job = await self._job_repo.get_by_id(job.id)
                    if failed_job:
                        failed_job.status = "failed"
                        failed_job.error = str(e)
                        await self._job_repo.update(failed_job)
                asyncio.run_coroutine_threadsafe(_fail(), loop)
            finally:
                shutil.rmtree(data_dir, ignore_errors=True)

        loop.run_in_executor(None, _train)

        return job

    async def get_job_status(self, job_id: str) -> RetrainJob | None:
        return await self._job_repo.get_by_id(job_id)

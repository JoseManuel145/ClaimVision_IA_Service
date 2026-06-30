import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import init_db
from app.modules.nosupervised.presentation.routes import router as v1_router
from app.modules.ocr.presentation.routes import router as ocr_router
from app.modules.supervised.presentation.routes import router as v2_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(v1_router, prefix="/api/v1")
app.include_router(ocr_router, prefix="/api/v1")
app.include_router(v2_router, prefix="/api/v2")


@app.get("/", tags=["Root"])
async def root():
    return {"service": settings.APP_TITLE, "version": settings.APP_VERSION, "status": "running"}


if __name__ == "__main__":
    print(f"Documentación: http://127.0.0.1:8000/docs")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

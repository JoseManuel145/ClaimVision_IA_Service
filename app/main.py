import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db
from app.presentation.routes import router


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

app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root():
    return {"service": settings.APP_TITLE, "version": settings.APP_VERSION, "status": "running"}


if __name__ == "__main__":
    print(f"Documentación: http://127.0.0.1:8000/docs")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

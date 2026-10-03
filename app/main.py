from fastapi import FastAPI

from app.api.v1.router import api_router
from app.core.config import settings

app = FastAPI(
    title=settings.APP_NAME,
    description="Backend API for an AI-powered English grammar learning application.",
    version=settings.APP_VERSION,
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/")
def read_root():
    return {
        "message": settings.APP_NAME,
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
    }

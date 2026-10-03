from fastapi import APIRouter

from app.api.v1 import ai, grammar

api_router = APIRouter()

api_router.include_router(ai.router, prefix="/ai", tags=["AI"])
api_router.include_router(grammar.router, prefix="/grammar", tags=["Grammar"])

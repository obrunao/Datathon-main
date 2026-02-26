"""FastAPI: ponto de entrada da aplicacao."""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import router
from src.monitoring import setup_logging

setup_logging()
logger = logging.getLogger("passos_magicos.api")

app = FastAPI(
    title="Datathon Passos Magicos",
    description="API para predicao de risco de defasagem escolar",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
def startup():
    logger.info("API iniciada com sucesso")

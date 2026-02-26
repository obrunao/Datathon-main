
from fastapi import FastAPI
from app.routes import router

app = FastAPI(title="Datathon Passos Mágicos")
app.include_router(router)

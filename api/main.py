from fastapi import FastAPI
from api.routes.jobs import router as jobs_router
from api.services.storage_service import ensure_storage_dirs
from database.db import Base, engine

app = FastAPI(
    title="Women Health",
    version="1.0.0",
    description="Plataforma para registro e acompanhamento de atendimentos, com captura de vídeo (cirurgia/consulta/fisioterapia), gravação de áudio e geração automática de relatório em PDF."
)


@app.on_event("startup")
def startup():
    ensure_storage_dirs()
    Base.metadata.create_all(bind=engine)


app.include_router(jobs_router)

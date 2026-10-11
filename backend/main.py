from fastapi import FastAPI
from backend.api.routers import data_router, prep_router, models_router, eval_router, xai_router

app = FastAPI(
    title="API - Prédiction Réadmission Diabète",
    description="Backend IA pour le projet Data Mining ENSI",
    version="1.0.0"
)

app.include_router(data_router)
app.include_router(prep_router) 
app.include_router(models_router)
app.include_router(eval_router)
app.include_router(xai_router) 
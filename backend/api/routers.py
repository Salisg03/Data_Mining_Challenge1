from fastapi import APIRouter
from pydantic import BaseModel
from backend.services import eda_service, data_loader_service, preprocessing_service, modeling_service, evaluation_service, xai_service
import numpy as np

# 1. Router Data
data_router = APIRouter(prefix="/api/v1/data", tags=["Data Exploration"])

@data_router.get("/sample")
def get_data_sample():
    df = data_loader_service.get_raw_data()
    df_clean = df.replace({np.nan: None}) 
    return df_clean.head(50).to_dict(orient="records")

@data_router.get("/missing-values")
def get_missing_values():
    return eda_service.get_missing_values_stats()

@data_router.get("/target-distribution")
def get_target():
    return eda_service.get_target_distribution()

@data_router.get("/types")
def get_types():
    return eda_service.get_data_types()

@data_router.get("/stats")
def get_stats():
    return eda_service.get_basic_stats()

@data_router.get("/correlations")
def get_correlations():
    return eda_service.get_numerical_correlations()

@data_router.get("/outliers")
def get_outliers():
    return eda_service.get_outliers_info()

@data_router.get("/bias-analysis")
def get_bias():
    return eda_service.get_bias_analysis()

# 2. Router Preparation
class PrepRequest(BaseModel):
    test_size: float

prep_router = APIRouter(prefix="/api/v1/prep", tags=["Data Preparation"])

@prep_router.post("/process")
def run_preprocessing(req: PrepRequest):
    return preprocessing_service.process_and_split_data(req.test_size)

# 3. Router Modélisation
class TrainRequest(BaseModel):
    model_name: str
    params: dict

models_router = APIRouter(prefix="/api/v1/models", tags=["Modeling"])

@models_router.post("/train")
def train_selected_model(req: TrainRequest):
    return modeling_service.train_model(req.model_name, req.params)

# 4. Router Evaluation
class ThresholdRequest(BaseModel):
    model_name: str
    threshold: float

eval_router = APIRouter(prefix="/api/v1/eval", tags=["Evaluation"])

@eval_router.get("/compare")
def compare_models():
    return evaluation_service.get_comparison_data()

@eval_router.post("/threshold")
def eval_threshold(req: ThresholdRequest):
    return evaluation_service.get_threshold_metrics(req.model_name, req.threshold)

# 5. Router XAI
xai_router = APIRouter(prefix="/api/v1/xai", tags=["Explainability"])

@xai_router.get("/global/{model_name}")
def get_global_xai(model_name: str):
    return xai_service.get_global_explanations(model_name)

@xai_router.get("/local/{model_name}/{patient_index}")
def get_local_xai(model_name: str, patient_index: int):
    return xai_service.get_local_explanations(model_name, patient_index)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
import boto3
import os
from io import BytesIO
import uvicorn

#env variables config
MODEL_BUCKET = os.environ.get('MODEL_BUCKET', 'cancer-pred-pipeline-models')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-2')

#FastAPI init
app = FastAPI(
    title="Cancer Prediction Model API",
    description="API for cancer prediction model statistics and health checks",
    version="1.0.0"
)

s3_client = boto3.client('s3', region_name=AWS_REGION)

#global vars
metrics = None
class_mapping = None
metadata_summary = None
last_updated = None

#Pydantic models
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    last_updated: Optional[str]

class ModelStatsResponse(BaseModel):
    status: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    class_labels: list
    per_class_metrics: Dict[str, Any]
    confusion_matrix: list
    metadata_summary: Optional[Dict[str, Any]]

def load_model_artifacts_from_s3():

    global metrics, class_mapping, metadata_summary, last_updated
    
    try:
        # Load latest model timestamp
        latest_obj = s3_client.get_object(Bucket=MODEL_BUCKET, Key='models/latest.txt')
        timestamp = latest_obj['Body'].read().decode('utf-8').strip()
        
        # Load metrics
        metrics_key = f'models/{timestamp}/metrics.json'
        metrics_obj = s3_client.get_object(Bucket=MODEL_BUCKET, Key=metrics_key)
        metrics = json.loads(metrics_obj['Body'].read().decode('utf-8'))
        
        # Load class mapping
        mapping_key = f'models/{timestamp}/class_mapping.json'
        mapping_obj = s3_client.get_object(Bucket=MODEL_BUCKET, Key=mapping_key)
        class_mapping = json.loads(mapping_obj['Body'].read().decode('utf-8'))
        
        # Load metadata summary (optional)
        try:
            metadata_key = f'models/{timestamp}/metadata_summary.json'
            metadata_obj = s3_client.get_object(Bucket=MODEL_BUCKET, Key=metadata_key)
            metadata_summary = json.loads(metadata_obj['Body'].read().decode('utf-8'))
        except:
            metadata_summary = None
            print("Metadata summary not found, continuing without it")
        
        last_updated = timestamp
        
        print(f"Model artifacts loaded successfully. Timestamp: {timestamp}")
        return True
        
    except Exception as e:
        print(f"Error loading model artifacts: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """Load model artifacts on startup"""
    print("Loading model artifacts on startup...")
    success = load_model_artifacts_from_s3()
    if not success:
        print("Warning: Failed to load model artifacts on startup")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=metrics is not None,
        last_updated=last_updated
    )

@app.get("/model/stats", response_model=ModelStatsResponse)
async def model_stats():
    """Get model statistics and metrics"""
    if metrics is None:
        raise HTTPException(status_code=503, detail="Model metrics not loaded")
    
    return ModelStatsResponse(
        status="loaded",
        accuracy=metrics['accuracy'],
        precision=metrics['precision'],
        recall=metrics['recall'],
        f1_score=metrics['f1_score'],
        class_labels=metrics['class_labels'],
        per_class_metrics=metrics['per_class_metrics'],
        confusion_matrix=metrics['confusion_matrix'],
        metadata_summary=metadata_summary
    )

@app.get("/model/info")
async def model_info():
    """Get general information about the loaded model"""
    if metrics is None or class_mapping is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "loaded",
        "last_updated": last_updated,
        "num_classes": len(metrics['class_labels']),
        "class_labels": metrics['class_labels'],
        "class_mapping": class_mapping,
        "total_samples_evaluated": sum(metrics['per_class_metrics']['support']) if 'per_class_metrics' in metrics else None
    }

@app.post("/model/reload")
async def reload_model():
    """Reload model artifacts from S3"""
    success = load_model_artifacts_from_s3()
    
    if success:
        return {
            "status": "success",
            "message": "Model artifacts reloaded successfully",
            "timestamp": last_updated
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model artifacts")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Cancer Prediction Model API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "model_stats": "/model/stats",
            "model_info": "/model/info",
            "reload_model": "/model/reload"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
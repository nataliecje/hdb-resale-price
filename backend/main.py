from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import logging
import time
import json
from datetime import datetime
import joblib
import pickle

# Import our services
from llm_service import LLMService
from bto_recommendation import BTORecommendationSystem
from model_training import train_models  # We'll create this function

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HDB BTO Recommendation System",
    description="AI-powered system for HDB BTO development recommendations and price predictions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
llm_service = LLMService(provider="openai")
bto_system = BTORecommendationSystem()

# Pydantic models for API requests/responses
class BTORecommendationRequest(BaseModel):
    query: str
    max_estates: Optional[int] = 10
    years_back: Optional[int] = 10
    flat_types: Optional[List[str]] = ["3 ROOM", "4 ROOM"]
    floor_levels: Optional[List[str]] = ["low", "middle", "high"]

class PricePredictionRequest(BaseModel):
    town: str
    flat_type: str
    floor_level: str = "middle"
    floor_area_sqm: Optional[float] = None

class EstateAnalysisRequest(BaseModel):
    town: str

class ModelTrainingRequest(BaseModel):
    retrain: bool = False
    models: Optional[List[str]] = ["Random Forest", "Gradient Boosting", "Linear Regression"]

# API endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "HDB BTO Recommendation System API",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/bto-recommendations")
async def get_bto_recommendations(request: BTORecommendationRequest):
    """
    Generate BTO development recommendations based on user query
    """
    try:
        start_time = time.time()
        
        # Generate recommendations
        response = llm_service.generate_bto_recommendations_response(request.query)
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "response": response,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
            "request_params": request.dict()
        }
        
    except Exception as e:
        logger.error(f"Error generating BTO recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/price-prediction")
async def predict_price(request: PricePredictionRequest):
    """
    Predict BTO and resale prices for specific parameters
    """
    try:
        start_time = time.time()
        
        # Make price prediction
        prediction = bto_system.predict_bto_prices(
            request.town, 
            request.flat_type, 
            request.floor_level
        )
        
        if not prediction:
            raise HTTPException(status_code=404, detail=f"No data available for {request.town}")
        
        # Calculate income requirements
        income_req = bto_system.calculate_income_requirements(
            prediction['predicted_bto_price']
        )
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "prediction": {
                "predicted_resale_price": prediction['predicted_resale_price'],
                "predicted_bto_price": prediction['predicted_bto_price'],
                "discount_applied": prediction['discount_applied'],
                "income_requirements": income_req
            },
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
            "request_params": request.dict()
        }
        
    except Exception as e:
        logger.error(f"Error predicting price: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/estate-analysis")
async def get_estate_analysis(request: EstateAnalysisRequest):
    """
    Get detailed analysis for a specific estate
    """
    try:
        start_time = time.time()
        
        analysis = llm_service.get_estate_specific_analysis(request.town)
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "analysis": analysis,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
            "town": request.town
        }
        
    except Exception as e:
        logger.error(f"Error analyzing estate: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/estates")
async def get_available_estates():
    """
    Get list of all available estates in the database
    """
    try:
        estate_data = bto_system.get_estate_bto_analysis()
        estates = estate_data['town'].tolist()
        
        return {
            "success": True,
            "estates": estates,
            "total_estates": len(estates),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching estates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/train-model")
async def train_model(request: ModelTrainingRequest, background_tasks: BackgroundTasks):
    """
    Train or retrain the price prediction model
    """
    try:
        # Add training task to background
        background_tasks.add_task(train_models, request.retrain, request.models)
        
        return {
            "success": True,
            "message": "Model training started in background",
            "retrain": request.retrain,
            "models": request.models,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting model training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model-status")
async def get_model_status():
    """
    Get current model status and performance metrics
    """
    try:
        # Check if model files exist
        model_files = {
            "resale_price_model.pkl": False,
            "preprocessing_objects.pkl": False,
            "model_results.pkl": False,
            "model_comparison.csv": False
        }
        
        import os
        for file in model_files.keys():
            model_files[file] = os.path.exists(file)
        
        # Load model performance if available
        performance_metrics = {}
        if model_files["model_comparison.csv"]:
            import pandas as pd
            try:
                comparison_df = pd.read_csv("model_comparison.csv")
                performance_metrics = comparison_df.to_dict('records')
            except:
                pass
        
        return {
            "success": True,
            "model_files": model_files,
            "performance_metrics": performance_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """
    Comprehensive health check endpoint
    """
    try:
        health_status = {
            "api_status": "healthy",
            "database_connection": "unknown",
            "model_loaded": False,
            "services_ready": False,
            "timestamp": datetime.now().isoformat()
        }
        
        # Check database connection
        try:
            estate_data = bto_system.get_estate_bto_analysis()
            health_status["database_connection"] = "healthy"
            health_status["data_records"] = len(estate_data)
        except Exception as e:
            health_status["database_connection"] = f"error: {str(e)}"
        
        # Check model status
        if bto_system.model is not None:
            health_status["model_loaded"] = True
        
        # Check services
        if health_status["database_connection"] == "healthy" and health_status["model_loaded"]:
            health_status["services_ready"] = True
        
        return health_status
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Monitoring and metrics endpoints
@app.get("/api/metrics")
async def get_metrics():
    """
    Get system performance metrics
    """
    try:
        # Get basic system metrics
        import psutil
        import os
        
        metrics = {
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent
            },
            "application": {
                "uptime": time.time(),  # You might want to track this properly
                "model_files_size": {},
                "database_size": "unknown"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Get model file sizes
        model_files = ["resale_price_model.pkl", "preprocessing_objects.pkl", "model_results.pkl"]
        for file in model_files:
            if os.path.exists(file):
                metrics["application"]["model_files_size"][file] = os.path.getsize(file)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "success": False,
        "error": "Endpoint not found",
        "timestamp": datetime.now().isoformat()
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {
        "success": False,
        "error": "Internal server error",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


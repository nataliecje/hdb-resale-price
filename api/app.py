from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import Optional, List, Dict, Any
import pandas as pd
import joblib
import logging
from data.data_ingestion import engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HDB Resale Price Prediction API",
    description="API for predicting HDB resale prices and recommending investment opportunities",
    version="1.0.0"
)

# CORS configuration
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",  # Add additional frontend URLs as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request validation
class PredictionRequest(BaseModel):
    town: str
    flat_type: str
    storey_range: str
    month: Optional[int] = 1
    remaining_lease: Optional[float] = 99.0
    block: Optional[str] = "unknown"
    transaction_id: Optional[int] = 0
    street_name: Optional[str] = "unknown"
    floor_area_sqm: Optional[float] = None
    flat_model: Optional[str] = "Improved"
    lease_commence_date: Optional[int] = 2010
    
    @field_validator('flat_type')
    @classmethod
    def validate_flat_type(cls, v):
        valid_types = ["1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE"]
        if v not in valid_types:
            raise ValueError(f'flat_type must be one of: {valid_types}')
        return v
    
    @field_validator('remaining_lease')
    @classmethod
    def validate_remaining_lease(cls, v):
        if v is not None and (v < 0 or v > 99):
            raise ValueError('remaining_lease must be between 0 and 99 years')
        return v

class EstateRecommendation(BaseModel):
    town: str
    predictions: List[Dict[str, Any]]

# Load trained model with error handling
try:
    model = joblib.load('resale_price_model.pkl')
    logger.info("Model loaded successfully")
except FileNotFoundError:
    logger.error("Model file not found. Please ensure 'resale_price_model.pkl' exists.")
    model = None
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

@app.get("/")
def root():
    return {
        "message": "HDB Resale Price Prediction API is running",
        "endpoints": {
            "predict_price": "/predict_price",
            "recommend_estates": "/recommend_estates",
            "health": "/health"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not_loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "database_status": "connected"  # You might want to add actual DB health check
    }

@app.post("/predict_price")
def predict_price(data: PredictionRequest):
    """Predict HDB resale price based on input features"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        # Convert Pydantic model to dict
        input_data = data.dict()
        
        # Set default floor area if not provided
        if input_data['floor_area_sqm'] is None:
            area_mapping = {
                "1 ROOM": 35,
                "2 ROOM": 45,
                "3 ROOM": 70,
                "4 ROOM": 90,
                "5 ROOM": 110,
                "EXECUTIVE": 130
            }
            input_data['floor_area_sqm'] = area_mapping.get(data.flat_type, 80)
        
        # Create DataFrame for prediction
        df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        # Calculate additional insights
        price_per_sqm = prediction / input_data['floor_area_sqm']
        
        return {
            "predicted_price": round(prediction, 2),
            "price_per_sqm": round(price_per_sqm, 2),
            "input_features": {
                "town": data.town,
                "flat_type": data.flat_type,
                "storey_range": data.storey_range,
                "floor_area_sqm": input_data['floor_area_sqm']
            }
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/recommend_estates", response_model=List[EstateRecommendation])
def recommend_estates(limit: int = 10, min_txn_threshold: int = 50):
    """Recommend estates with investment potential based on low transaction volumes"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        # Step 1: Find estates with low past transaction counts
        query = f"""
        SELECT town, COUNT(*) AS txn_count
        FROM resale_transactions
        GROUP BY town
        HAVING COUNT(*) < {min_txn_threshold}
        ORDER BY txn_count ASC
        LIMIT {limit}
        """
        
        estates_df = pd.read_sql(query, engine)
        
        if estates_df.empty:
            return []
        
        results = []
        
        # Configuration for predictions
        floor_categories = ["01 TO 03", "04 TO 06", "07 TO 09", "10 TO 12"]
        flat_types = ["3 ROOM", "4 ROOM", "5 ROOM"]
        
        for _, row in estates_df.iterrows():
            town_name = row["town"]
            
            estate_entry = {
                "town": town_name,
                "transaction_count": int(row["txn_count"]),
                "predictions": []
            }
            
            for flat_type in flat_types:
                for floor_cat in floor_categories:
                    try:
                        # Get appropriate floor area
                        area_mapping = {
                            "3 ROOM": 70,
                            "4 ROOM": 90,
                            "5 ROOM": 110
                        }
                        floor_area = area_mapping[flat_type]
                        
                        # Construct feature dict
                        input_data = {
                            "town": town_name,
                            "flat_type": flat_type,
                            "storey_range": floor_cat,
                            "floor_area_sqm": floor_area,
                            "flat_model": "Improved",
                            "lease_commence_date": 2010,
                            "remaining_lease": 99.0,
                            "month": 1,
                            "block": "unknown",
                            "transaction_id": 0,
                            "street_name": "unknown"
                        }
                        
                        # Predict resale price
                        df_pred = pd.DataFrame([input_data])
                        resale_pred = model.predict(df_pred)[0]
                        
                        # BTO estimate (20% discount assumption)
                        bto_estimate = resale_pred * 0.8
                        
                        # Required monthly income (simplified calculation)
                        # Assume 30% of income for housing, 25-year loan
                        monthly_payment = bto_estimate * 0.004  # Rough estimate
                        required_income = monthly_payment / 0.3
                        
                        estate_entry["predictions"].append({
                            "flat_type": flat_type,
                            "storey_range": floor_cat,
                            "floor_area_sqm": floor_area,
                            "predicted_resale": round(resale_pred, 2),
                            "predicted_bto": round(bto_estimate, 2),
                            "required_monthly_income": round(required_income, 2),
                            "price_per_sqm": round(resale_pred / floor_area, 2)
                        })
                        
                    except Exception as e:
                        logger.warning(f"Failed to predict for {town_name}, {flat_type}, {floor_cat}: {str(e)}")
                        continue
            
            if estate_entry["predictions"]:  # Only add if we have predictions
                results.append(estate_entry)
        
        return results
        
    except Exception as e:
        logger.error(f"Estate recommendation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

@app.get("/towns")
def get_available_towns():
    """Get list of available towns from the database"""
    try:
        query = "SELECT DISTINCT town FROM resale_transactions ORDER BY town"
        towns_df = pd.read_sql(query, engine)
        return {"towns": towns_df['town'].tolist()}
    except Exception as e:
        logger.error(f"Error fetching towns: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch towns")

@app.get("/flat_types")
def get_flat_types():
    """Get available flat types"""
    return {
        "flat_types": ["1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
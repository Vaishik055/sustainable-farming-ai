"""
Sustainable Farming AI - Backend API
=====================================
Main FastAPI application for the Agentic AI Assistant
for Sustainable Farming Decisions.

Week 2: Crop Recommendation Endpoint
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import sys
import os

# Ensure backend directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.crop_service import CropRecommendationService

# ============================================================
# FastAPI App Initialization
# ============================================================
app = FastAPI(
    title="Sustainable Farming AI - Crop Recommendation",
    description=(
        "Agentic AI Assistant for Sustainable Farming Decisions\n\n"
        "Week 2: Crop Recommendation Module\n\n"
        "This API predicts the best crop to grow based on soil "
        "nutrients (N, P, K) and weather conditions "
        "(temperature, humidity, pH, rainfall)."
    ),
    version="1.0.0"
)

# Load the crop recommendation service
try:
    crop_service = CropRecommendationService()
except FileNotFoundError as e:
    crop_service = None
    print(f"⚠️ Warning: {e}")
    print("   The /predict/crop endpoint will return errors until the model is available.")


# ============================================================
# Request Model (Input Validation with Pydantic)
# ============================================================
class CropRecommendationRequest(BaseModel):
    """
    Input schema for crop recommendation.
    
    All values are validated using Pydantic Field constraints.
    """
    nitrogen: float = Field(
        ...,
        ge=0, le=300,
        description="Nitrogen content in soil (kg/ha)",
        example=90
    )
    phosphorus: float = Field(
        ...,
        ge=0, le=300,
        description="Phosphorus content in soil (kg/ha)",
        example=42
    )
    potassium: float = Field(
        ...,
        ge=0, le=300,
        description="Potassium content in soil (kg/ha)",
        example=43
    )
    temperature: float = Field(
        ...,
        ge=-10, le=60,
        description="Average temperature (°C)",
        example=25.5
    )
    humidity: float = Field(
        ...,
        ge=0, le=100,
        description="Average relative humidity (%)",
        example=80
    )
    ph: float = Field(
        ...,
        ge=0, le=14,
        description="Soil pH value",
        example=6.5
    )
    rainfall: float = Field(
        ...,
        ge=0, le=5000,
        description="Average rainfall (mm)",
        example=200
    )

    class Config:
        ison_schema_extra = {
            "example": {
                "nitrogen": 90,
                "phosphorus": 42,
                "potassium": 43,
                "temperature": 25.5,
                "humidity": 80,
                "ph": 6.5,
                "rainfall": 200
            }
        }


# ============================================================
# Response Model
# ============================================================
class CropRecommendationResponse(BaseModel):
    """
    Output schema for crop recommendation.
    """
    recommended_crop: str = Field(
        ...,
        description="Name of the recommended crop"
    )
    confidence: Optional[float] = Field(
        None,
        description="Model confidence score (0-1) if available"
    )


# ============================================================
# API Endpoints
# ============================================================
@app.get("/")
def root():
    """Root endpoint - API health check."""
    return {
        "message": "Sustainable Farming AI API is running!",
        "module": "Crop Recommendation",
        "status": "healthy",
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    model_loaded = crop_service is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded
    }


@app.post("/predict/crop", response_model=CropRecommendationResponse)
def predict_crop(request: CropRecommendationRequest):
    """
    Predict the best crop to grow based on soil and weather conditions.
    
    **Parameters:**
    - nitrogen: Nitrogen content in soil (kg/ha)
    - phosphorus: Phosphorus content in soil (kg/ha)
    - potassium: Potassium content in soil (kg/ha)
    - temperature: Average temperature (°C)
    - humidity: Average relative humidity (%)
    - ph: Soil pH value
    - rainfall: Average rainfall (mm)
    
    **Returns:**
    - recommended_crop: Name of the suggested crop
    - confidence: Model's confidence in the prediction
    """
    # Check if model is loaded
    if crop_service is None:
        raise HTTPException(
            status_code=503,
            detail="Crop recommendation model is not available. Please ensure the model file exists."
        )
    
    # Make prediction
    result = crop_service.predict(
        nitrogen=request.nitrogen,
        phosphorus=request.phosphorus,
        potassium=request.potassium,
        temperature=request.temperature,
        humidity=request.humidity,
        ph=request.ph,
        rainfall=request.rainfall
    )
    
    # Check for validation errors
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    # Return clean response
    return CropRecommendationResponse(
        recommended_crop=result["recommended_crop"],
        confidence=result.get("confidence")
    )


# ============================================================
# Run the app directly (for development)
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
import math
from typing import List, Optional
from enum import Enum
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

app = FastAPI(
    title="FoodLoop AI Backend API",
    description="ML Demand Prediction, Surplus Detection, and NGO Matching",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. ML Model Setup
def train_demand_model():
    np.random.seed(42)
    n_samples = 300
    
    day_of_week = np.random.randint(0, 7, n_samples)
    meal_type = np.random.randint(0, 2, n_samples)
    prev_customers = np.random.randint(50, 250, n_samples)
    is_weekend = np.where(day_of_week >= 5, 1, 0)
    is_festival = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
    
    demand = (
        prev_customers * 0.85 +
        day_of_week * 4 +
        meal_type * 25 +
        is_weekend * 35 +
        is_festival * 50 +
        np.random.normal(0, 10, n_samples)
    )
    
    X = pd.DataFrame({
        'day_of_week': day_of_week,
        'meal_type': meal_type,
        'previous_customers': prev_customers,
        'is_weekend': is_weekend,
        'is_festival': is_festival
    })
    y = np.maximum(demand, 20)
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model

ml_model = train_demand_model()

# 2. Schemas
class PredictionRequest(BaseModel):
    day_of_week: int = Field(..., ge=0, le=6)
    meal_type: int = Field(..., ge=0, le=1)
    previous_customers: int = Field(..., ge=0)
    is_weekend: bool
    is_festival: bool
    avg_food_per_person_kg: float = Field(0.27, gt=0)

class PredictionResponse(BaseModel):
    predicted_customers: int
    recommended_food_kg: float

class SurplusRequest(BaseModel):
    prepared_food_kg: float = Field(..., ge=0)
    consumed_food_kg: float = Field(..., ge=0)
    threshold_percentage: float = Field(15.0, ge=0)

class SurplusResponse(BaseModel):
    surplus_kg: float
    surplus_percentage: float
    donation_recommended: bool

class Location(BaseModel):
    latitude: float
    longitude: float

class NGO(BaseModel):
    id: str
    name: str
    capacity_kg: float
    location: Location

class MatchRequest(BaseModel):
    donation_quantity_kg: float = Field(..., gt=0)
    donor_location: Location
    perishability_hours: float = Field(..., gt=0)

class NGOMatchResult(BaseModel):
    ngo_id: str
    ngo_name: str
    distance_km: float
    capacity_kg: float
    match_score_percentage: float
    distance_score: float
    capacity_score: float
    urgency_score: float

class DonationStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    ACCEPTED = "ACCEPTED"
    PICKED_UP = "PICKED_UP"
    DELIVERED = "DELIVERED"
    CANCELLED = "CANCELLED"

class DonationCreate(BaseModel):
    food_category: str
    quantity_kg: float
    usable_until_hours: float
    location: Location
    donor_name: str

class Donation(DonationCreate):
    id: str
    status: DonationStatus = DonationStatus.AVAILABLE

DEMO_NGOS: List[NGO] = [
    NGO(id="ngo_1", name="Helping Hands NGO", capacity_kg=80.0, location=Location(latitude=28.6315, longitude=77.2197)),
    NGO(id="ngo_2", name="Food For All Foundation", capacity_kg=50.0, location=Location(latitude=28.6500, longitude=77.2300)),
    NGO(id="ngo_3", name="Community Kitchen", capacity_kg=120.0, location=Location(latitude=28.5800, longitude=77.1900)),
]

donations_db: List[Donation] = []

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# 3. Endpoints
@app.post("/api/ai/predict", response_model=PredictionResponse)
def predict_demand(req: PredictionRequest):
    features = pd.DataFrame([{
        'day_of_week': req.day_of_week,
        'meal_type': req.meal_type,
        'previous_customers': req.previous_customers,
        'is_weekend': 1 if req.is_weekend else 0,
        'is_festival': 1 if req.is_festival else 0
    }])
    pred = int(round(ml_model.predict(features)[0]))
    rec = round(pred * req.avg_food_per_person_kg, 1)
    return PredictionResponse(predicted_customers=pred, recommended_food_kg=rec)

@app.post("/api/surplus/calculate", response_model=SurplusResponse)
def calculate_surplus(req: SurplusRequest):
    if req.consumed_food_kg > req.prepared_food_kg:
        raise HTTPException(status_code=400, detail="Consumed food cannot exceed prepared food.")
    surplus_kg = round(req.prepared_food_kg - req.consumed_food_kg, 2)
    surplus_pct = round((surplus_kg / req.prepared_food_kg * 100) if req.prepared_food_kg > 0 else 0.0, 1)
    return SurplusResponse(
        surplus_kg=surplus_kg,
        surplus_percentage=surplus_pct,
        donation_recommended=surplus_pct >= req.threshold_percentage
    )

@app.post("/api/ngos/match", response_model=List[NGOMatchResult])
def match_ngos(req: MatchRequest):
    results = []
    for ngo in DEMO_NGOS:
        dist_km = haversine_distance(
            req.donor_location.latitude, req.donor_location.longitude,
            ngo.location.latitude, ngo.location.longitude
        )
        dist_score = max(0.0, 100.0 * (1.0 - (dist_km / 15.0)))
        cap_ratio = min(1.0, ngo.capacity_kg / req.donation_quantity_kg) if req.donation_quantity_kg > 0 else 1.0
        cap_score = cap_ratio * 100.0
        urgency_score = max(0.0, min(100.0, (10.0 / max(0.5, req.perishability_hours)) * 30.0))
        final_score = round(0.40 * dist_score + 0.30 * cap_score + 0.30 * urgency_score, 1)
        
        results.append(NGOMatchResult(
            ngo_id=ngo.id,
            ngo_name=ngo.name,
            distance_km=round(dist_km, 2),
            capacity_kg=ngo.capacity_kg,
            match_score_percentage=min(100.0, final_score),
            distance_score=round(dist_score, 1),
            capacity_score=round(cap_score, 1),
            urgency_score=round(urgency_score, 1)
        ))
    return sorted(results, key=lambda x: x.match_score_percentage, reverse=True)

@app.post("/api/donations", response_model=Donation)
def create_donation(donation_in: DonationCreate):
    donation = Donation(id=f"don_{len(donations_db) + 1}", **donation_in.dict(), status=DonationStatus.AVAILABLE)
    donations_db.append(donation)
    return donation

@app.post("/api/donations/{donation_id}/accept", response_model=Donation)
def accept_donation(donation_id: str):
    for don in donations_db:
        if don.id == donation_id:
            don.status = DonationStatus.ACCEPTED
            return don
    raise HTTPException(status_code=404, detail="Donation not found")
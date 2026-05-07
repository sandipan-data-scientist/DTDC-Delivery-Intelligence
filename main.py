# main.py
# DTDC Delivery Intelligence API
# exposes three endpoints:
#   /predict-delay    -> predicts if a shipment will be delayed
#   /predict-transit  -> predicts how many days delivery will take
#   /pricing-advice   -> returns demand-based pricing recommendation for a route

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import uvicorn

app = FastAPI(
    title="DTDC Delivery Intelligence API",
    description="Predicts delivery delays, transit times, and pricing recommendations",
    version="1.0.0"
)

# load all models at startup (not on each request, that would be slow)
delay_model    = joblib.load("models/delay_classifier.pkl")
transit_model  = joblib.load("models/transit_regressor.pkl")
scaler         = joblib.load("models/feature_scaler.pkl")
le_mode        = joblib.load("models/le_mode.pkl")
le_nature      = joblib.load("models/le_nature.pkl")
le_payment     = joblib.load("models/le_payment.pkl")
route_stats    = pd.read_csv("models/route_stats.csv")
city_forecasts = pd.read_csv("models/city_forecasts.csv")


# request schema for prediction endpoints
class ShipmentInput(BaseModel):
    chargeable_wt: float
    actual_wt: float
    volumetric_wt: float
    total_pieces: int
    tariff: float
    vas_charges: float
    total_amount: float
    mode: str           # "Surface", "Express", "Air Cargo"
    nature: str         # "Dox", "Non-Dox"
    payment: str        # "Cash", "Card", "Wallet"
    is_b2b: int         # 1 if business shipment, 0 if personal
    has_vas: int        # 1 if value added service opted
    is_inter_state: int # 1 if sender and receiver are in different states
    day_of_week: int    # 0=Monday ... 6=Sunday
    week_number: int
    month: int


def prepare_features(data: ShipmentInput) -> np.ndarray:
    """
    Converts the input schema into a feature vector the model can understand.
    Applies the same label encoding and scaling as training time.
    """
    # encode categoricals
    try:
        mode_enc    = le_mode.transform([data.mode])[0]
        nature_enc  = le_nature.transform([data.nature])[0]
        payment_enc = le_payment.transform([data.payment])[0]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid category value: {e}")

    # vol_to_actual_ratio
    vol_ratio = data.volumetric_wt / data.actual_wt if data.actual_wt > 0 else 0
    rev_per_kg = data.total_amount / data.chargeable_wt if data.chargeable_wt > 0 else 0

    # scale the numeric features (same order as training)
    numeric_raw = np.array([[
        data.chargeable_wt, data.tariff, data.vas_charges,
        vol_ratio, data.total_pieces, rev_per_kg
    ]])
    numeric_scaled = scaler.transform(numeric_raw)

    # final feature vector
    features = np.array([[
        numeric_scaled[0][0],    # chargeable_wt scaled
        data.actual_wt,
        data.volumetric_wt,
        data.total_pieces,
        numeric_scaled[0][1],    # tariff scaled
        numeric_scaled[0][2],    # vas_charges scaled
        data.total_amount,
        mode_enc,
        nature_enc,
        payment_enc,
        data.is_b2b,
        data.has_vas,
        data.is_inter_state,
        numeric_scaled[0][3],    # vol_ratio scaled
        data.day_of_week,
        data.week_number,
        data.month
    ]])

    return features


@app.get("/")
def root():
    return {"message": "DTDC Delivery Intelligence API is running"}


@app.post("/predict-delay")
def predict_delay(shipment: ShipmentInput):
    """
    Returns whether a shipment is likely to be delayed and the probability.
    A shipment is considered delayed if transit takes more than 3 days.
    """
    features = prepare_features(shipment)
    prediction = int(delay_model.predict(features)[0])
    probability = float(delay_model.predict_proba(features)[0][1])

    return {
        "will_be_delayed": bool(prediction),
        "delay_probability": round(probability, 4),
        "risk_level": (
            "High" if probability > 0.6 else
            "Medium" if probability > 0.35 else
            "Low"
        )
    }


@app.post("/predict-transit")
def predict_transit(shipment: ShipmentInput):
    """
    Returns the predicted number of transit days for the shipment.
    """
    features = prepare_features(shipment)
    days = float(transit_model.predict(features)[0])

    return {
        "predicted_transit_days": round(days, 2),
        "expected_delivery": f"Approximately {round(days)} day(s) after booking"
    }


@app.get("/pricing-advice/{sender_city}/{receiver_city}")
def pricing_advice(sender_city: str, receiver_city: str):
    """
    Returns demand-based pricing recommendation for a route.
    """
    route = f"{sender_city}_to_{receiver_city}"
    match = route_stats[route_stats["route"] == route]

    if match.empty:
        # check reverse route
        reverse = f"{receiver_city}_to_{sender_city}"
        match = route_stats[route_stats["route"] == reverse]
        if match.empty:
            raise HTTPException(status_code=404, detail=f"Route {route} not found")

    row = match.iloc[0]
    return {
        "route": route,
        "avg_tariff_inr": round(row["avg_tariff"], 2),
        "demand_score": round(row["demand_score"], 3),
        "delay_rate": round(row["delay_rate"], 3),
        "total_shipments": int(row["total_shipments"]),
        "pricing_action": row["pricing_action"],
        "avg_revenue_per_shipment": round(row["revenue_per_shipment"], 2)
    }


@app.get("/city-forecast/{city_name}")
def city_forecast(city_name: str):
    """
    Returns the 30-day ahead demand forecast for a given city.
    """
    match = city_forecasts[city_forecasts["city"].str.lower() == city_name.lower()]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"City {city_name} not found")

    row = match.iloc[0]
    return {
        "city": row["city"],
        "predicted_avg_daily_demand": round(row["pred_avg_demand"], 2),
        "confidence_lower": round(row["pred_lower"], 2),
        "confidence_upper": round(row["pred_upper"], 2),
        "current_delay_rate": round(row.get("delay_rate", 0), 3)
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
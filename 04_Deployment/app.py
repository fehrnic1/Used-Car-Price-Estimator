import base64
import json
import os
import re
import pickle
import traceback

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image
from openai import OpenAI
from pydantic import BaseModel, Field
from transformers import pipeline

# ---------------------------------------------------------------------------
# Load models at startup — wrapped so startup never crashes silently
# ---------------------------------------------------------------------------

STARTUP_ERRORS = []

try:
    with open("car_price_model.pkl", "rb") as f:
        model_payload = pickle.load(f)
    ml_model     = model_payload["model"]
    features     = model_payload["features"]
    le           = model_payload["label_encoders"]
    importances  = dict(zip(features, ml_model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
except Exception as e:
    ml_model = None
    STARTUP_ERRORS.append(f"ML model load failed: {e}")

try:
    RECOGNITION_MODEL_ID = os.environ.get("RECOGNITION_MODEL_ID", "fehrnic1/car-recognition-model")
    car_recognizer = pipeline("image-classification", model=RECOGNITION_MODEL_ID)
except Exception as e:
    car_recognizer = None
    STARTUP_ERRORS.append(f"Recognition model load failed ({RECOGNITION_MODEL_ID}): {e}")

try:
    DAMAGE_MODEL_ID   = os.environ.get("DAMAGE_MODEL_ID", "fehrnic1/car-damage-model")
    damage_classifier = pipeline("image-classification", model=DAMAGE_MODEL_ID)
except Exception as e:
    damage_classifier = None
    STARTUP_ERRORS.append(f"Damage model load failed ({DAMAGE_MODEL_ID}): {e}")

try:
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except Exception as e:
    openai_client = None
    STARTUP_ERRORS.append(f"OpenAI client failed: {e}")

if STARTUP_ERRORS:
    print("STARTUP ERRORS:", STARTUP_ERRORS)

# ---------------------------------------------------------------------------
# Helper: parse brand + year from Stanford Cars class name
# ---------------------------------------------------------------------------

def parse_class(class_name):
    brand      = class_name.split()[0]
    year_match = re.search(r"(\d{4})$", class_name.strip())
    model_year = int(year_match.group(1)) if year_match else 2015
    return brand, model_year

# ---------------------------------------------------------------------------
# NLP: extract features from text description
# ---------------------------------------------------------------------------

class CarFeatures(BaseModel):
    brand:            str = Field(description="Car manufacturer, e.g. BMW, Toyota")
    model_year:       int = Field(description="4-digit year of manufacture")
    milage:           int = Field(description="Mileage in miles")
    fuel_type:        str = Field(description="One of: Gasoline, Diesel, Electric, Hybrid")
    transmission:     str = Field(description="One of: Automatic, Manual")
    has_accident:     int = Field(description="1 if any accident history, 0 if none")
    clean_title_flag: int = Field(description="1 if clean title, 0 if salvage/rebuilt")

SYSTEM_PROMPT = """You are a car data extraction specialist. Extract structured information from
natural language car descriptions. Normalise:
- Fuel types to: Gasoline, Diesel, Electric, Hybrid
- Transmission to: Automatic, Manual
- Any accident/collision/fender bender → has_accident = 1
- Salvage/rebuilt title → clean_title_flag = 0"""

def extract_from_text(description):
    response = openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Extract car features from: {description}"}
        ],
        response_format=CarFeatures,
        temperature=0
    )
    return response.choices[0].message.parsed.model_dump()

# ---------------------------------------------------------------------------
# ML: predict price
# ---------------------------------------------------------------------------

def predict_price(extracted):
    brand        = extracted.get("brand", "Other")
    model_year   = extracted.get("model_year", 2015)
    milage       = extracted.get("milage", 60000)
    fuel_type    = extracted.get("fuel_type", "Gasoline")
    transmission = extracted.get("transmission", "Automatic")
    has_accident = extracted.get("has_accident", 0)
    clean_title  = extracted.get("clean_title_flag", 1)
    engine_hp    = extracted.get("engine_hp", 180)
    condition_score = extracted.get("condition_score", 0)

    car_age          = 2024 - model_year
    age_times_milage = car_age * milage

    fuel_enc  = le["fuel_type"].transform([fuel_type])[0]   if le.get("fuel_type")  and fuel_type  in le["fuel_type"].classes_  else 0
    trans_enc = le["transmission"].transform([transmission])[0] if le.get("transmission") and transmission in le["transmission"].classes_ else 0
    brand_grp = brand if le.get("brand") and brand in le["brand"].classes_ else "Other"
    brand_enc = le["brand"].transform([brand_grp])[0] if le.get("brand") else 0

    row = {
        "model_year": model_year, "milage": milage, "car_age": car_age,
        "age_times_milage": age_times_milage, "fuel_type_enc": fuel_enc,
        "transmission_enc": trans_enc, "brand_enc": brand_enc,
        "has_accident": has_accident, "clean_title_flag": clean_title,
        "engine_hp": engine_hp, "condition_score": condition_score
    }
    X = pd.DataFrame([{f: row.get(f, 0) for f in features}])
    return round(float(ml_model.predict(X)[0]), 0)

# ---------------------------------------------------------------------------
# NLP: generate explanation
# ---------------------------------------------------------------------------

def generate_explanation(extracted, price):
    feature_str = "\n".join([f"  - {f} (importance: {imp:.3f})" for f, imp in top_features])
    prompt = f"""Predicted price: ${price:,.0f}

Car details:
{json.dumps(extracted, indent=2)}

The model's top 5 most important features are:
{feature_str}

Provide a structured explanation with:
1. One sentence summarising the estimated price
2. Two or three bullet points identifying the most significant price drivers
3. One sentence noting any limitations or uncertainty"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a car valuation expert. Explain used car price estimates clearly for a non-technical audience."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# ---------------------------------------------------------------------------
# Main prediction function — same pattern as week 7 reference
# ---------------------------------------------------------------------------

def predict(image, description, milage_input):
    try:
        if STARTUP_ERRORS:
            return {"startup_errors": STARTUP_ERRORS}
        if image is None and (not description or not description.strip()):
            return {"error": "Please upload a photo or enter a description."}

        extracted = {}

        # --- CV path (image provided) ---
        if image is not None:
            img = Image.open(image).convert("RGB")

            recog_result  = car_recognizer(img)
            top_recog     = max(recog_result, key=lambda x: x["score"])
            brand, model_year = parse_class(top_recog["label"])

            damage_result = damage_classifier(img)
            top_damage    = max(damage_result, key=lambda x: x["score"])
            damage_id2label = damage_classifier.model.config.id2label
            damage_label2id = {v: k for k, v in damage_id2label.items()}
            condition_score = damage_label2id.get(top_damage["label"], 0)

            extracted.update({
                "brand": brand, "model_year": model_year,
                "condition_score": condition_score
            })
            cv_info = {
                "recognised_class": top_recog["label"],
                "recognition_confidence": round(top_recog["score"], 4),
                "damage_class": top_damage["label"],
                "damage_confidence": round(top_damage["score"], 4),
            }
        else:
            cv_info = "No image provided."

        # --- NLP path (text provided) ---
        if description and description.strip():
            text_features = extract_from_text(description)
            for k, v in text_features.items():
                if k not in extracted or v:
                    extracted[k] = v

        # --- Mileage override ---
        if milage_input and float(milage_input) > 0:
            extracted["milage"] = int(float(milage_input))
        if "milage" not in extracted:
            extracted["milage"] = 60000

        # --- ML prediction ---
        price = predict_price(extracted)

        # --- NLP explanation ---
        explanation = generate_explanation(extracted, price)

        return {
            "estimated_price": f"${price:,.0f}",
            "explanation": explanation,
            "extracted_features": extracted,
            "cv_analysis": cv_info
        }

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


# ---------------------------------------------------------------------------
# Gradio Interface — same pattern as reference projects
# ---------------------------------------------------------------------------

iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="filepath", label="Car Photo (optional)"),
        gr.Textbox(
            label="Car Description (optional)",
            placeholder="e.g. 2019 BMW 3 Series, 45,000 miles, gasoline, automatic, no accidents, clean title",
            lines=3
        ),
        gr.Number(label="Mileage (miles) — required if not in description", value=None)
    ],
    outputs=gr.JSON(label="Result"),
    title="Used Car Price Estimator",
    description=(
        "Upload a car photo **and/or** describe your car in text. "
        "The system identifies the car, assesses damage, predicts the market price, "
        "and explains the key factors."
    ),
    examples=[
        [None, "2019 BMW 3 Series, 45,000 miles, gasoline, automatic, no accidents, clean title", 45000],
        [None, "2015 Toyota Camry, 87,000 miles, gasoline, manual, had one accident, clean title", 87000],
        [None, "2022 Tesla Model 3, 28,000 miles, electric, automatic, no accidents, clean title", 28000],
    ]
)

if __name__ == "__main__":
    iface.launch()

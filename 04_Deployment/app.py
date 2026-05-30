import json
import os
import re
import pickle
import traceback

import gradio as gr
import pandas as pd
from PIL import Image
from pydantic import BaseModel, Field
from openai import OpenAI
from transformers import pipeline

# ---------------------------------------------------------------------------
# Load models at startup
# ---------------------------------------------------------------------------

with open("car_price_model.pkl", "rb") as f:
    model_payload = pickle.load(f)
ml_model     = model_payload["model"]
features     = model_payload["features"]
le           = model_payload["label_encoders"]
importances  = dict(zip(features, ml_model.feature_importances_))
top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]

RECOGNITION_MODEL_ID = os.environ.get("RECOGNITION_MODEL_ID", "fehrnic1/car-recognition-model")
car_recognizer = pipeline("image-classification", model=RECOGNITION_MODEL_ID)

DAMAGE_MODEL_ID  = os.environ.get("DAMAGE_MODEL_ID", "fehrnic1/car-damage-model")
damage_classifier = pipeline("image-classification", model=DAMAGE_MODEL_ID)

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_class(class_name):
    brand      = class_name.split()[0]
    year_match = re.search(r"(\d{4})$", class_name.strip())
    model_year = int(year_match.group(1)) if year_match else 2015
    return brand, model_year

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

def generate_explanation(extracted, price):
    feature_str = "\n".join([f"  - {f} (importance: {imp:.3f})" for f, imp in top_features])
    prompt = f"""Predicted price: ${price:,.0f}
Car details: {json.dumps(extracted, indent=2)}
Top 5 model features: {feature_str}
Write: 1 summary sentence, 2-3 bullet points on key price drivers, 1 uncertainty note."""
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a car valuation expert. Be concise and clear."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------

def predict(image_file, description, milage_str):
    try:
        if image_file is None and (not description or not description.strip()):
            return "Please upload a photo or enter a description.", "", ""

        extracted = {}
        cv_info      = "No image provided."
        display_img  = None

        # CV path
        if image_file is not None:
            img = Image.open(image_file).convert("RGB")
            display_img = img

            recog_result = car_recognizer(img)
            top_recog    = max(recog_result, key=lambda x: x["score"])
            brand, model_year = parse_class(top_recog["label"])

            damage_result = damage_classifier(img)
            top_damage    = max(damage_result, key=lambda x: x["score"])
            damage_label2id = {v: k for k, v in damage_classifier.model.config.id2label.items()}
            condition_score = damage_label2id.get(top_damage["label"], 0)

            extracted.update({"brand": brand, "model_year": model_year, "condition_score": condition_score})
            cv_info = f"Recognised: {top_recog['label']} ({top_recog['score']:.1%})\nDamage: {top_damage['label']} ({top_damage['score']:.1%})"

        # NLP path
        if description and description.strip():
            text_features = extract_from_text(description)
            for k, v in text_features.items():
                if k not in extracted or v:
                    extracted[k] = v

        # Mileage
        if milage_str and milage_str.strip():
            try:
                extracted["milage"] = int(float(milage_str.strip()))
            except ValueError:
                pass
        if "milage" not in extracted:
            extracted["milage"] = 60000

        price       = predict_price(extracted)
        explanation = generate_explanation(extracted, price)

        price_output   = f"Estimated Price: ${price:,.0f}"
        details_output = f"CV Analysis:\n{cv_info}\n\nExtracted Features:\n{json.dumps(extracted, indent=2)}"

        return display_img, price_output, explanation, details_output

    except Exception as e:
        return None, f"Error: {str(e)}", traceback.format_exc(), ""

# ---------------------------------------------------------------------------
# Gradio Interface — reference project style
# ---------------------------------------------------------------------------

iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.File(label="Car Photo (optional)", file_types=["image"]),
        gr.Textbox(
            label="Car Description (optional)",
            placeholder="e.g. 2019 BMW 3 Series, 45,000 miles, gasoline, automatic, no accidents, clean title",
            lines=3
        ),
        gr.Textbox(label="Mileage (miles)", placeholder="e.g. 45000")
    ],
    outputs=[
        gr.Image(label="Uploaded Car"),
        gr.Textbox(label="Estimated Price"),
        gr.Textbox(label="Explanation", lines=8),
        gr.Textbox(label="Details", lines=6)
    ],
    title="Used Car Price Estimator",
    description="Upload a car photo and/or describe your car in text. Get an estimated market price with an explanation of the key factors."
)

if __name__ == "__main__":
    iface.launch()

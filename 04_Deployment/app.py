import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import gradio as gr
import torch
from PIL import Image
from pydantic import BaseModel, Field
from openai import OpenAI
from transformers import (
    ViTForImageClassification,
    AutoImageProcessor,
    pipeline as hf_pipeline
)

# ---------------------------------------------------------------------------
# Load models at startup
# ---------------------------------------------------------------------------

# ML model (pickle — included in Space repo)
with open("car_price_model.pkl", "rb") as f:
    model_payload = pickle.load(f)
ml_model  = model_payload["model"]
features  = model_payload["features"]
le        = model_payload["label_encoders"]

# Feature importances for explanation prompt
importances    = dict(zip(features, ml_model.feature_importances_))
top_features   = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]

# Car recognition model (from HF Hub)
RECOGNITION_MODEL_ID = os.environ.get("RECOGNITION_MODEL_ID", "YOUR_HF_USERNAME/car-recognition-model")
recog_processor = AutoImageProcessor.from_pretrained(RECOGNITION_MODEL_ID)
recog_model     = ViTForImageClassification.from_pretrained(RECOGNITION_MODEL_ID)
car_recognizer  = hf_pipeline("image-classification", model=recog_model, image_processor=recog_processor)

# Car damage model (from HF Hub)
DAMAGE_MODEL_ID  = os.environ.get("DAMAGE_MODEL_ID", "YOUR_HF_USERNAME/car-damage-model")
damage_processor = AutoImageProcessor.from_pretrained(DAMAGE_MODEL_ID)
damage_model     = ViTForImageClassification.from_pretrained(DAMAGE_MODEL_ID)
damage_classifier = hf_pipeline("image-classification", model=damage_model, image_processor=damage_processor)

# OpenAI client (API key from Space secret)
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# Helper: parse brand + year from Stanford Cars class name
# ---------------------------------------------------------------------------

def parse_class(class_name):
    brand      = class_name.split()[0]
    year_match = re.search(r"(\d{4})$", class_name.strip())
    model_year = int(year_match.group(1)) if year_match else 2015
    return brand, model_year

# ---------------------------------------------------------------------------
# CV: predict from image
# ---------------------------------------------------------------------------

def predict_from_image(image: Image.Image):
    # Car recognition → brand + model_year
    recog_result = car_recognizer(image)
    top_recog    = max(recog_result, key=lambda x: x["score"])
    brand, model_year = parse_class(top_recog["label"])

    # Damage assessment → condition_score
    damage_result   = damage_classifier(image)
    top_damage      = max(damage_result, key=lambda x: x["score"])
    damage_label2id = {v: k for k, v in damage_model.config.id2label.items()}
    condition_score = damage_label2id.get(top_damage["label"], 0)

    return {
        "brand":           brand,
        "model_year":      model_year,
        "condition_score": condition_score,
        "_recog_label":    top_recog["label"],
        "_recog_conf":     top_recog["score"],
        "_damage_label":   top_damage["label"],
        "_damage_conf":    top_damage["score"],
    }

# ---------------------------------------------------------------------------
# NLP: extract features from text
# ---------------------------------------------------------------------------

class CarFeatures(BaseModel):
    brand:            str = Field(description="Car manufacturer, e.g. BMW, Toyota")
    model_year:       int = Field(description="4-digit year of manufacture")
    milage:           int = Field(description="Mileage in miles")
    fuel_type:        str = Field(description="One of: Gasoline, Diesel, Electric, Hybrid")
    transmission:     str = Field(description="One of: Automatic, Manual")
    has_accident:     int = Field(description="1 if any accident history, 0 if none")
    clean_title_flag: int = Field(description="1 if clean title, 0 if salvage/rebuilt")

SYSTEM_PROMPT = """
You are a car data extraction specialist. Extract structured information from
natural language car descriptions. Normalise:
- Fuel types to: Gasoline, Diesel, Electric, Hybrid
- Transmission to: Automatic, Manual
- Any accident/collision/fender bender → has_accident = 1
- Salvage/rebuilt title → clean_title_flag = 0
"""

def extract_from_text(description: str) -> dict:
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

def predict_price(extracted: dict) -> float:
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

def generate_explanation(extracted: dict, price: float) -> str:
    feature_str = "\n".join([f"  - {f} (importance: {imp:.3f})" for f, imp in top_features])
    prompt = f"""
Predicted price: ${price:,.0f}

Car details:
{json.dumps(extracted, indent=2)}

The model's top 5 most important features are:
{feature_str}

Provide a structured explanation with:
1. One sentence summarising the estimated price
2. Two or three bullet points identifying the most significant price drivers
3. One sentence noting any limitations or uncertainty
"""
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
# Main prediction function
# ---------------------------------------------------------------------------

def predict(image, description, milage_override):
    if image is None and not description.strip():
        return "Please upload a photo or enter a description.", "", ""

    extracted = {}

    # Image path
    if image is not None:
        img = Image.fromarray(image).convert("RGB")
        cv_result = predict_from_image(img)
        extracted.update({
            "brand":           cv_result["brand"],
            "model_year":      cv_result["model_year"],
            "condition_score": cv_result["condition_score"],
        })
        cv_summary = (
            f"**Recognised:** {cv_result['_recog_label']} ({cv_result['_recog_conf']:.1%} confidence)\n"
            f"**Damage:** {cv_result['_damage_label']} ({cv_result['_damage_conf']:.1%} confidence)"
        )
    else:
        cv_summary = "No image provided."

    # Text path — fills in or overrides extracted fields
    if description.strip():
        text_features = extract_from_text(description)
        for k, v in text_features.items():
            if k not in extracted or v:
                extracted[k] = v

    # Manual mileage override (user input always wins)
    if milage_override and milage_override > 0:
        extracted["milage"] = int(milage_override)

    # Default mileage if still missing
    if "milage" not in extracted:
        extracted["milage"] = 60000

    # Predict price
    price = predict_price(extracted)

    # Generate explanation
    explanation = generate_explanation(extracted, price)

    price_str = f"## Estimated Price: ${price:,.0f}"
    details   = f"**Extracted features:**\n```json\n{json.dumps(extracted, indent=2)}\n```\n\n{cv_summary}"

    return price_str, explanation, details

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Used Car Price Estimator") as demo:
    gr.Markdown("# 🚗 Used Car Price Estimator")
    gr.Markdown(
        "Upload a photo of your car **and/or** describe it in text. "
        "The system identifies the car, assesses its condition, predicts the market price, "
        "and explains the key factors."
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Car Photo (optional)")
            text_input  = gr.Textbox(
                label="Car Description (optional)",
                placeholder="e.g. 2019 BMW 3 Series, 45,000 miles, gasoline, automatic, no accidents, clean title",
                lines=3
            )
            milage_input = gr.Number(
                label="Mileage (miles) — required if not in description",
                value=None,
                precision=0
            )
            submit_btn = gr.Button("Estimate Price", variant="primary")

        with gr.Column():
            price_output       = gr.Markdown(label="Price Estimate")
            explanation_output = gr.Markdown(label="Explanation")
            details_output     = gr.Markdown(label="Details")

    submit_btn.click(
        fn=predict,
        inputs=[image_input, text_input, milage_input],
        outputs=[price_output, explanation_output, details_output]
    )

    gr.Markdown(
        "**Example descriptions you can copy:**\n"
        "- `2019 BMW 3 Series, 45,000 miles, gasoline, automatic, no accidents, clean title`\n"
        "- `2015 Toyota Camry, 87,000 miles, gasoline, manual, had one accident, clean title`\n"
        "- `2022 Tesla Model 3, 28,000 miles, electric, automatic, no accidents, clean title`"
    )

demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)

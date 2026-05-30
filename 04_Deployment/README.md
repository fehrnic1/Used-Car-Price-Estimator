---
title: Used Car Price Estimator
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.9.1"
app_file: app.py
pinned: false
python_version: "3.11"
---

# Used Car Price Estimator

Upload a car photo and/or provide a text description to get an estimated market price with a plain-language explanation.

## How it works
1. **Computer Vision** — identifies the car make/model/year and assesses damage from the photo
2. **NLP** — extracts structured features from a text description
3. **ML** — predicts the market price using a Random Forest model trained on 4,000+ US car listings
4. **NLP** — generates a plain-language explanation of the predicted price

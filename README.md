# Fake News Detector

A Flask REST API that classifies news articles as FAKE or REAL using TF-IDF + Logistic Regression.

## Run locally
1. Create venv and activate
2. pip install -r requirements.txt
3. python app.py

## Deploy
This repo is ready for Render.com deployment (uses gunicorn).

## Files
- app.py — Flask API
- model.pkl, vectorizer.pkl — trained model & vectorizer

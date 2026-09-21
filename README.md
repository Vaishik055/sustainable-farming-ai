# 🌾 Sustainable Farming AI — Agentic AI Assistant

> **Agentic AI Assistant for Sustainable Farming Decisions**

## Week 2: Crop Recommendation Module

This module recommends a suitable crop based on soil nutrients and weather conditions using a trained Machine Learning model.

---

## 📁 Project Structure

```text
sustainable-farming-ai/
│
├── backend/
│   ├── agents/                     # Future: AI agents
│   ├── services/
│   │   ├── __init__.py
│   │   └── crop_service.py         # Crop recommendation service
│   ├── models/
│   │   └── crop_model.pkl           # Trained Random Forest model
│   ├── scalers/
│   │   └── crop_scaler.pkl          # StandardScaler object
│   ├── encoders/
│   │   └── label_encoder.pkl        # LabelEncoder object
│   └── main.py                      # FastAPI application
│
├── datasets/
│   └── crop_recommendation.csv      # Crop recommendation dataset
│
├── notebooks/
│   └── crop_recommendation.ipynb    # Full ML pipeline notebook
│
├── tests/
│   └── test_crop_service.py         # Service unit tests
│
├── requirements.txt
├── .gitignore
└── README.md
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path

BASE = Path(__file__).parent.parent / "models"


def load_artifacts():
    model    = joblib.load(BASE / "xgb_model.joblib")
    encoders = joblib.load(BASE / "encoders.joblib")
    explainer = joblib.load(BASE / "shap_explainer.joblib")
    with open(BASE / "feature_names.json") as f:
        features = json.load(f)
    with open(BASE / "metrics.json") as f:
        metrics = json.load(f)
    return model, encoders, explainer, features, metrics


def build_input(user_input: dict, encoders: dict, features: list) -> pd.DataFrame:
    """Convert raw user form values into a model-ready DataFrame row."""
    luxury_brands = {
        'bmw', 'mercedes-benz', 'audi', 'lexus', 'cadillac',
        'lincoln', 'infiniti', 'acura', 'volvo', 'jaguar',
        'land rover', 'porsche', 'tesla'
    }
    condition_map = {'salvage': 0, 'fair': 1, 'good': 2, 'excellent': 3, 'like new': 4, 'new': 5}

    row = {
        'year':         user_input['year'],
        'car_age':      2024 - user_input['year'],
        'log_odometer': np.log1p(user_input['odometer']),
        'miles_per_year': user_input['odometer'] / max(2024 - user_input['year'], 1),
        'condition_num':  condition_map.get(user_input.get('condition', 'good'), 2),
        'manufacturer':   user_input.get('manufacturer', 'unknown'),
        'fuel':           user_input.get('fuel', 'gas'),
        'transmission':   user_input.get('transmission', 'automatic'),
        'drive':          user_input.get('drive', 'fwd'),
        'type':           user_input.get('type', 'sedan'),
        'cylinders':      user_input.get('cylinders', '6 cylinders'),
        'state':          user_input.get('state', 'ca'),
        'is_luxury':      int(user_input.get('manufacturer', '').lower() in luxury_brands),
        'is_clean_title': int(user_input.get('title_status', 'clean') == 'clean'),
        'is_automatic':   int(user_input.get('transmission', 'automatic') == 'automatic'),
    }

    df = pd.DataFrame([row])

    cat_cols = ['manufacturer', 'fuel', 'transmission', 'drive', 'type', 'cylinders', 'state']
    for col in [c for c in cat_cols if c in encoders and c in df.columns]:
        le = encoders[col]
        val = df[col].astype(str).values[0]
        if val in le.classes_:
            df[col] = le.transform([val])
        else:
            df[col] = le.transform([le.classes_[0]])

    return df[[f for f in features if f in df.columns]]


def predict_price(model, df_input: pd.DataFrame) -> dict:
    log_pred = model.predict(df_input)[0]
    price = np.expm1(log_pred)
    low   = price * 0.85
    high  = price * 1.15
    return {'price': float(price), 'low': float(low), 'high': float(high)}

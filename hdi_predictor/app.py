"""
app.py  –  HDI Predictor Flask Application
"""

import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
import webbrowser


# ── App setup ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder='templates', static_folder='static')

# ── Load model ─────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'HDI.pkl')

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

try:
    model_data = load_model()
    print("✓ Model loaded successfully")
except FileNotFoundError:
    print("⚠  Model not found – run train_model.py first")
    model_data = None

# ── HDI category helper ────────────────────────────────────────────────────────
def get_hdi_category(score: float) -> tuple[str, str]:
    """Returns (category, css_class)"""
    if score >= 0.800:
        return "Very High", "very-high"
    elif score >= 0.700:
        return "High", "high"
    elif score >= 0.550:
        return "Medium", "medium"
    else:
        return "Low", "low"

def get_model_stats() -> dict:
    if not model_data:
        return {}
    return {
        'r2_train': model_data.get('r2_train', 'N/A'),
        'r2_test':  model_data.get('r2_test',  'N/A'),
        'rmse':     model_data.get('rmse',      'N/A'),
        'train_size': model_data.get('train_size', 'N/A'),
        'test_size':  model_data.get('test_size',  'N/A'),
    }

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    stats = get_model_stats()
    return render_template('indexnew.html', stats=stats)


@app.route('/predict', methods=['POST'])
def predict():
    if not model_data:
        return render_template('resultnew.html',
                               error="Model not loaded. Please run train_model.py first.")

    try:
        # Grab & validate form inputs
        life_exp     = float(request.form['life_expectancy'])
        school_years = float(request.form['mean_years_schooling'])
        gni          = float(request.form['gni_per_capita'])

        # Bounds checks
        if not (20 <= life_exp <= 90):
            raise ValueError("Life Expectancy must be between 20 and 90 years.")
        if not (0 <= school_years <= 20):
            raise ValueError("Mean Years of Schooling must be between 0 and 20.")
        if not (100 <= gni <= 100000):
            raise ValueError("GNI Per Capita must be between 100 and 100,000 USD.")

        # Prediction
        features      = np.array([[life_exp, school_years, gni]])
        predicted_hdi = float(model_data['model'].predict(features)[0])
        predicted_hdi = round(np.clip(predicted_hdi, 0, 1), 4)
        category, css_class = get_hdi_category(predicted_hdi)

        # Component sub-indices (for display)
        life_index   = round((life_exp - 20) / (85 - 20), 3)
        edu_index    = round(school_years / 18, 3)
        income_index = round((np.log(gni) - np.log(100)) / (np.log(75000) - np.log(100)), 3)

        country = request.form.get('country', '').strip() or 'Your Country'

        return render_template(
            'resultnew.html',
            predicted_hdi=predicted_hdi,
            category=category,
            css_class=css_class,
            country=country,
            life_exp=life_exp,
            school_years=school_years,
            gni=int(gni),
            life_index=life_index,
            edu_index=edu_index,
            income_index=income_index,
            stats=get_model_stats(),
            error=None
        )

    except ValueError as ve:
        return render_template('resultnew.html', error=str(ve), stats=get_model_stats())
    except Exception as e:
        return render_template('resultnew.html',
                               error=f"Prediction failed: {str(e)}",
                               stats=get_model_stats())


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API endpoint"""
    if not model_data:
        return jsonify({'error': 'Model not loaded'}), 503
    try:
        data         = request.get_json(force=True)
        life_exp     = float(data['life_expectancy'])
        school_years = float(data['mean_years_schooling'])
        gni          = float(data['gni_per_capita'])
        features     = np.array([[life_exp, school_years, gni]])
        hdi          = float(model_data['model'].predict(features)[0])
        hdi          = round(np.clip(hdi, 0, 1), 4)
        cat, _       = get_hdi_category(hdi)
        return jsonify({'hdi_score': hdi, 'category': cat})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/visualizations')
def visualizations():
    return render_template('visualizations.html', stats=get_model_stats())


if __name__ == '__main__':
    import webbrowser
    import threading
    threading.Timer(1.5, lambda: webbrowser.open('http://127.0.0.1:5000')).start()
    app.run(debug=False, host='127.0.0.1', port=5000)

# 🌍 HDI Predictor

A full-stack machine learning web application that predicts the **Human Development Index (HDI)** of countries using Python, Flask, and Linear Regression.

---

## 🏗️ Project Architecture

```
hdi_predictor/
├── app.py                  # Flask web application
├── train_model.py          # Dataset generation, EDA, model training
├── requirements.txt        # Python dependencies
├── data/
│   └── hdi_dataset.csv     # Generated HDI dataset (88 countries)
├── model/
│   └── HDI.pkl             # Serialized trained model (Pickle)
├── static/
│   ├── css/style.css       # Full responsive stylesheet
│   ├── js/main.js          # Animations & interactivity
│   └── plots/              # EDA visualizations (PNG)
│       ├── heatmap.png
│       ├── distributions.png
│       ├── strip_plot.png
│       ├── scatter_gni.png
│       ├── actual_vs_predicted.png
│       └── feature_importance.png
└── templates/
    ├── indexnew.html       # Home / prediction form
    ├── resultnew.html      # Prediction result display
    └── visualizations.html # EDA visualizations gallery
```

---

## ⚙️ Setup & Installation

### 1. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model (generates dataset + plots)
```bash
python train_model.py
```
This will:
- Generate a realistic HDI dataset (88 countries)
- Run Exploratory Data Analysis (EDA)
- Save 6 visualization plots to `static/plots/`
- Train a Linear Regression model
- Serialize the model to `model/HDI.pkl`

### 4. Launch the Flask app
```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 🤖 ML Pipeline

| Stage | Detail |
|---|---|
| Dataset | 88 countries, synthetically modeled on UNDP HDI methodology |
| Features | Life Expectancy, Mean Years of Schooling, GNI Per Capita |
| Target | HDI Score (0–1) |
| Preprocessing | Mean imputation, Label Encoding, Train/Test split (75/25) |
| Algorithm | Linear Regression (Scikit-learn) |
| Serialization | Pickle (.pkl) |
| Evaluation | R², RMSE, Actual vs Predicted scatter |

## 📊 HDI Categories (UNDP)

| Category | Score Range |
|---|---|
| Very High | ≥ 0.800 |
| High | 0.700 – 0.799 |
| Medium | 0.550 – 0.699 |
| Low | < 0.550 |

## 🛠️ Technology Stack

- **Python** – Core language
- **Flask** – Web framework
- **Scikit-learn** – ML model
- **Pandas** – Data manipulation
- **NumPy** – Numerical computing
- **Matplotlib** – Plotting
- **Seaborn** – Statistical visualization
- **Pickle** – Model serialization

## 📡 API Endpoint

```bash
POST /api/predict
Content-Type: application/json

{
  "life_expectancy": 72.4,
  "mean_years_schooling": 12.0,
  "gni_per_capita": 15000
}

# Response:
{
  "hdi_score": 0.7341,
  "category": "High"
}
```

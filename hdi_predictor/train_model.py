"""
train_model.py
Generates a realistic HDI dataset, performs EDA, trains a Linear Regression model,
saves it via Pickle, and exports visualizations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
MODEL_DIR  = os.path.join(BASE_DIR, 'model')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
PLOTS_DIR  = os.path.join(STATIC_DIR, 'plots')

for d in [DATA_DIR, MODEL_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── 1. Generate realistic HDI dataset ──────────────────────────────────────────
np.random.seed(42)

countries = [
    "Norway","Ireland","Switzerland","Iceland","Germany","Sweden","Australia",
    "Netherlands","Denmark","Finland","Singapore","United Kingdom","Belgium",
    "New Zealand","Canada","United States","Austria","Israel","Japan","Liechtenstein",
    "Slovenia","South Korea","Luxembourg","Spain","France","Czech Republic",
    "Malta","Estonia","Italy","United Arab Emirates","Greece","Cyprus","Lithuania",
    "Poland","Andorra","Latvia","Portugal","Saudi Arabia","Montenegro","Romania",
    "Belarus","Turkey","Uruguay","Bulgaria","Panama","Bahrain","Russia","Malaysia",
    "Serbia","Thailand","Albania","Mexico","Georgia","Sri Lanka","Cuba",
    "Iran","Bolivia","Philippines","South Africa","Egypt","Morocco","India",
    "Pakistan","Cambodia","Bangladesh","Kenya","Nigeria","Ethiopia","Chad",
    "Niger","Central African Republic","Mozambique","Burkina Faso","Mali",
    "Burundi","Somalia","Yemen","Haiti","Sudan","Guinea","Afghanistan",
    "Sierra Leone","Eritrea","DR Congo","Liberia","Guinea-Bissau","Malawi"
]

n = len(countries)

# Generate correlated HDI features
life_exp     = np.clip(np.random.normal(72, 12, n), 50, 85)
school_years = np.clip(np.random.normal(10, 4, n), 3, 16)
gni_per_cap  = np.clip(np.random.lognormal(9.5, 1.5, n), 500, 90000)

# HDI formula approximation (UNDP methodology simplified)
life_index   = (life_exp - 20) / (85 - 20)
edu_index    = school_years / 18
income_index = (np.log(gni_per_cap) - np.log(100)) / (np.log(75000) - np.log(100))
hdi_score    = np.cbrt(life_index * edu_index * income_index)
hdi_score    = np.clip(hdi_score + np.random.normal(0, 0.02, n), 0.25, 0.97)

def hdi_category(score):
    if score >= 0.800: return "Very High"
    elif score >= 0.700: return "High"
    elif score >= 0.550: return "Medium"
    else: return "Low"

hdi_categories = [hdi_category(s) for s in hdi_score]

df = pd.DataFrame({
    'Country': countries,
    'Life_Expectancy': np.round(life_exp, 2),
    'Mean_Years_Schooling': np.round(school_years, 2),
    'GNI_Per_Capita': np.round(gni_per_cap, 0).astype(int),
    'HDI_Score': np.round(hdi_score, 4),
    'HDI_Category': hdi_categories
})

# Sort by HDI descending
df = df.sort_values('HDI_Score', ascending=False).reset_index(drop=True)

csv_path = os.path.join(DATA_DIR, 'hdi_dataset.csv')
df.to_csv(csv_path, index=False)
print(f"✓ Dataset saved → {csv_path}")
print(df.head(10).to_string())
print(f"\nShape: {df.shape}")
print(f"\nHDI Category distribution:\n{df['HDI_Category'].value_counts()}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# ── 2. Data Preprocessing ──────────────────────────────────────────────────────
df['Life_Expectancy'].fillna(df['Life_Expectancy'].mean(), inplace=True)
df['Mean_Years_Schooling'].fillna(df['Mean_Years_Schooling'].mean(), inplace=True)
df['GNI_Per_Capita'].fillna(df['GNI_Per_Capita'].mean(), inplace=True)

le = LabelEncoder()
df['HDI_Category_Encoded'] = le.fit_transform(df['HDI_Category'])

features = ['Life_Expectancy', 'Mean_Years_Schooling', 'GNI_Per_Capita']
X = df[features]
y = df['HDI_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(f"\n✓ Train size: {len(X_train)}, Test size: {len(X_test)}")

# ── 3. EDA Visualizations ──────────────────────────────────────────────────────
sns.set_style("whitegrid")
palette = ["#2563EB","#7C3AED","#059669","#DC2626","#D97706"]

# 3a. Correlation Heatmap
fig, ax = plt.subplots(figsize=(7, 5))
corr = df[features + ['HDI_Score']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues', ax=ax,
            linewidths=0.5, square=True)
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'heatmap.png'), dpi=120, bbox_inches='tight')
plt.close()
print("✓ Heatmap saved")

# 3b. Distribution Plots
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
cols = ['Life_Expectancy', 'Mean_Years_Schooling', 'GNI_Per_Capita']
labels = ['Life Expectancy (years)', 'Mean Years of Schooling', 'GNI Per Capita (USD)']
for ax, col, lbl, color in zip(axes, cols, labels, palette):
    sns.histplot(df[col], kde=True, ax=ax, color=color, alpha=0.7)
    ax.set_title(lbl, fontsize=11, fontweight='bold')
    ax.set_xlabel('')
plt.suptitle('Feature Distributions', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'distributions.png'), dpi=120, bbox_inches='tight')
plt.close()
print("✓ Distribution plots saved")

# 3c. Strip Plot – HDI by Category
fig, ax = plt.subplots(figsize=(9, 5))
order = ['Very High', 'High', 'Medium', 'Low']
cat_palette = {'Very High':'#2563EB','High':'#059669','Medium':'#D97706','Low':'#DC2626'}
sns.stripplot(data=df, x='HDI_Category', y='HDI_Score', order=order,
              palette=cat_palette, size=8, jitter=True, alpha=0.8, ax=ax)
ax.set_title('HDI Score by Category', fontsize=14, fontweight='bold')
ax.set_xlabel('HDI Category', fontsize=11)
ax.set_ylabel('HDI Score', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'strip_plot.png'), dpi=120, bbox_inches='tight')
plt.close()
print("✓ Strip plot saved")

# 3d. Scatter – GNI vs HDI
fig, ax = plt.subplots(figsize=(9, 5))
colors = df['HDI_Category'].map(cat_palette)
sc = ax.scatter(df['GNI_Per_Capita'], df['HDI_Score'], c=colors, alpha=0.75, s=70, edgecolors='white', linewidth=0.4)
ax.set_xlabel('GNI Per Capita (USD)', fontsize=11)
ax.set_ylabel('HDI Score', fontsize=11)
ax.set_title('GNI Per Capita vs HDI Score', fontsize=14, fontweight='bold')
handles = [plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=v, markersize=9, label=k)
           for k, v in cat_palette.items()]
ax.legend(handles=handles, title='Category', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'scatter_gni.png'), dpi=120, bbox_inches='tight')
plt.close()
print("✓ Scatter plot (GNI vs HDI) saved")

# ── 4. Train Linear Regression ────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

r2_train = r2_score(y_train, y_pred_train)
r2_test  = r2_score(y_test, y_pred_test)
rmse     = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"\n✓ Model trained")
print(f"   R² (train): {r2_train:.4f}")
print(f"   R² (test) : {r2_test:.4f}")
print(f"   RMSE      : {rmse:.4f}")
print(f"   Coefficients: {dict(zip(features, model.coef_))}")

# 4a. Actual vs Predicted scatter
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, y_pred_test, color='#2563EB', alpha=0.75, s=70, edgecolors='white', linewidth=0.5, label='Test points')
mn, mx = min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())
ax.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect fit')
ax.set_xlabel('Actual HDI Score', fontsize=11)
ax.set_ylabel('Predicted HDI Score', fontsize=11)
ax.set_title(f'Actual vs Predicted HDI Score\n(R² = {r2_test:.4f})', fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'actual_vs_predicted.png'), dpi=120, bbox_inches='tight')
plt.close()
print("✓ Actual vs Predicted plot saved")

# 4b. Feature importance bar
fig, ax = plt.subplots(figsize=(7, 4))
coef_df = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_}).sort_values('Coefficient')
colors_bar = ['#DC2626' if v < 0 else '#2563EB' for v in coef_df['Coefficient']]
ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors_bar, alpha=0.85)
ax.set_title('Linear Regression Coefficients', fontsize=13, fontweight='bold')
ax.set_xlabel('Coefficient Value')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance.png'), dpi=120, bbox_inches='tight')
plt.close()
print("✓ Feature importance plot saved")

# ── 5. Save model & metadata ──────────────────────────────────────────────────
model_data = {
    'model': model,
    'features': features,
    'label_encoder': le,
    'r2_train': round(r2_train, 4),
    'r2_test': round(r2_test, 4),
    'rmse': round(rmse, 4),
    'train_size': len(X_train),
    'test_size': len(X_test)
}

pkl_path = os.path.join(MODEL_DIR, 'HDI.pkl')
with open(pkl_path, 'wb') as f:
    pickle.dump(model_data, f)
print(f"\n✓ Model serialized → {pkl_path}")
print("\n✅ All done! Run:  python app.py")

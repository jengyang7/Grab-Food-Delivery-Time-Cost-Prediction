# Grab Food Delivery Time & Cost Prediction

A Streamlit web application that predicts **delivery time** and **delivery cost** for Grab Food orders using the latest feature-engineering pipeline and the best-performing trained models.

**Live App:** https://grab-food-delivery-time-cost-prediction.streamlit.app/

---

## Overview

This project is part of **WQD 7007 Big Data Management** group work. The app allows users to upload a CSV of restaurant/order data and get predictions for:

- **Delivery Time** — estimated time (in minutes) for an order to be delivered
- **Delivery Cost** — estimated delivery fee (in SGD)

---

## Prediction Implementation

### Model

Both prediction tasks use the **best model selected during the latest retraining** (see `notebooks/training.ipynb`). Models are trained on the reduced feature set and the winner (by held-out test \(R^2\)) is exported for the Streamlit app.

| Target | Model File |
|---|---|
| Delivery Time | `xg_best_time.pkl` |
| Delivery Cost | `xg_best_cost.pkl` |

The app loads local model files when they exist and otherwise downloads the configured deployment artifacts on first run.

### Input Features

The model expects the following columns in the uploaded CSV:

| Column | Description |
|---|---|
| `name` | Restaurant name |
| `address` | Restaurant address |
| `cuisine` | Cuisine type(s), e.g. `['Western', 'Local']` |
| `lat` / `lon` | Restaurant coordinates |
| `opening_hours` | JSON string of operating hours |
| `radius` | Delivery radius |
| `rating` | Restaurant rating |
| `reviews_nr` | Number of reviews |
| `delivery_options` | e.g. `ONLY_DELIVERY`, `DELIVERY_TAKEAWAY` |
| `promo` | Whether a promo is available (`Yes`/`No`) |
| `loc_type` | Location type (e.g. `FOOD`) |
| `delivery_by` | Delivery provider (e.g. `GRAB`) |
| `region` | Region (e.g. `Central`) |
| `promo_code` | Promo code string |
| `delivery_time` | Optional ground truth for evaluation |
| `delivery_cost` | Optional ground truth for evaluation |

### Model Architecture

Training compares classical baselines (Linear / Decision Tree / Random Forest / XGBoost) and then tunes the best tree ensembles. The final selection is made on a held-out **90/10 split (`random_state=42`)** and the chosen models are saved for deployment.

The tuned Random Forest baseline uses:

| Parameter | Time Model | Cost Model |
|---|---|---|
| `n_estimators` | 300 | 300 |
| `max_depth` | None | None |
| `max_features` | `sqrt` | `sqrt` |
| `min_samples_split` | 10 | 10 |
| `min_samples_leaf` | 1 | 1 |

The selected production models from the latest notebook run are:

- **Delivery time**: **XGBoost (tuned)** (`xgboost_tuned`)
- **Delivery cost**: **Random Forest (tuned)** (`random_forest_tuned`)

### Preprocessing Pipeline

Before feeding data into the models, the following transformations are applied:

1. **Identity reduction** — `name` becomes `is_chain`, and `address` becomes a capped `mall_or_building_name` bucket instead of a full one-hot identity column.
2. **Cuisine reduction** — the top 20 cuisines are kept explicitly and all remaining cuisines fall into `Other`.
3. **Geospatial features** — raw `lat` / `lon` are retained alongside `dist_to_region_centroid` and `grid_restaurant_density`.
4. **Opening-hours features** — weekly hours, early-open, late-close, and consistency flags are parsed from the JSON string.
5. **Promo features** — free-delivery, minimum-spend, and promo-discount-type signals are extracted from promo text.
6. **Stable alignment** — the shared pipeline aligns inputs to the 154-feature training schema and never uses `delivery_time` or `delivery_cost` as model inputs.

### Prediction Flow

```
Upload CSV
    │
    ▼
preprocess_data()
    │  - Derive reduced cuisine / building buckets
    │  - Add promo, opening-hours, and geospatial features
    │  - Align columns to the saved training schema
    ▼
model.predict()
    │
    ▼
Append predicted column to original DataFrame
    │
    ▼
Display table + Download as CSV
```

---

## Model Evaluation

All models were evaluated on a held-out test set. Metrics below are from the final comparison.

### Delivery Time Prediction

| Model | MAE (min) | R² | MSE | RMSE (min) |
|---|---|---|---|---|
| Random Forest (tuned) | 5.0240 | 0.7660 | 59.8434 | 7.7358 |
| **XGBoost (tuned, selected)** | **4.9122** | **0.7767** | **57.0928** | **7.5560** |

The selected tuned XGBoost model reaches **77.7% \(R^2\)** with **4.91 minutes MAE** on the held-out test set.

### Delivery Cost Prediction

| Model | MAE (SGD) | R² | MSE | RMSE (SGD) |
|---|---|---|---|---|
| **Random Forest (tuned, selected)** | **2.0731** | **0.7180** | **11.5380** | **3.3968** |
| XGBoost (tuned) | 2.1550 | 0.7177 | 11.5510 | 3.3987 |

Both tuned models perform similarly for delivery cost; the latest notebook run selects tuned Random Forest by a small margin on held-out \(R^2\).

---

## Usage

### Online

Visit https://grab-food-delivery-time-cost-prediction.streamlit.app/ and:

1. Click **Download Test Dataset** to get a sample CSV.
2. Upload a CSV file using the file uploader.
3. Click **Predict Delivery Time** or **Predict Delivery Cost**.
4. View the results table and download predictions as CSV.

### Local Setup

```bash
git clone <repo-url>
cd Grab-Food-Delivery-Time-Cost-Prediction
pip install -r requirements.txt
streamlit run grab_app.py
```

---

## Project Structure

```
.
├── grab_app.py          # Streamlit app (prediction interface)
├── feature_pipeline.py  # Shared feature engineering logic
├── requirements.txt     # Python dependencies
├── feature_artifacts.json
├── model_metrics.json
├── scripts/train_phase2_models.py
├── test_dataset.csv     # Sample dataset for testing
├── unique_cuisines.txt  # Top cuisine buckets used in training
├── other_columns.txt    # Categorical feature families encoded in training
└── final_columns.txt    # Ordered column list expected by the models
```

---

## Dependencies

- `streamlit`
- `scikit-learn==1.3.2`
- `pandas`
- `requests`
- `lightgbm`
- `matplotlib`

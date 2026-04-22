# Grab Food Delivery Time & Cost Prediction

A Streamlit web application that predicts **delivery time** and **delivery cost** for Grab Food orders using trained Random Forest Regressor models.

**Live App:** https://grab-food-delivery-time-cost-prediction.streamlit.app/

---

## Overview

This project is part of **WQD 7007 Big Data Management** group work. The app allows users to upload a CSV of restaurant/order data and get predictions for:

- **Delivery Time** — estimated time (in minutes) for an order to be delivered
- **Delivery Cost** — estimated delivery fee (in SGD)

---

## Prediction Implementation

### Model

Both prediction tasks use a **Random Forest Regressor** (`scikit-learn 1.3.2`), trained separately for each target:

| Target | Model File |
|---|---|
| Delivery Time | `rf_best_time.pkl` |
| Delivery Cost | `rf_best_cost.pkl` |

The models are hosted on OneDrive and downloaded on first run, then cached locally to avoid repeated downloads.

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
| `delivery_time` | Actual delivery time (used as ground truth; dropped before time prediction) |
| `delivery_cost` | Actual delivery cost (used as ground truth; dropped before cost prediction) |

### Model Architecture

Both models are **Random Forest Regressors** tuned with `RandomizedSearchCV`. The best hyperparameters found:

| Parameter | Time Model | Cost Model |
|---|---|---|
| `n_estimators` | 200 | 200 |
| `max_depth` | None (unlimited) | None (unlimited) |
| `max_features` | 1.0 | 1.0 |
| `min_samples_split` | 2 | 2 |
| `min_samples_leaf` | 1 | 1 |

Training was performed on Google Colab. The dataset was split into train/test sets and models were fitted using the optimal hyperparameters above.

The models were compared against Decision Tree Regressors (default and tuned) before selecting Random Forest as the final choice.

### Preprocessing Pipeline

Before feeding data into the models, the following transformations are applied:

1. **Cuisine one-hot encoding** — the `cuisine` column (stored as a string list) is parsed and expanded into individual binary columns, one per unique cuisine type.
2. **Categorical encoding** — columns `name`, `address`, `delivery_options`, `loc_type`, `delivery_by`, and `region` are one-hot encoded using `pd.get_dummies`.
3. **Promo encoding** — `Yes`/`No` values in the `promo` column are mapped to `1`/`0`.
4. **Column alignment** — any columns expected by the model but absent from the input are added with value `0`, and the final column order is enforced to match training.

### Prediction Flow

```
Upload CSV
    │
    ▼
preprocess_data()
    │  - Parse cuisine lists → one-hot encode
    │  - get_dummies on categorical columns
    │  - Encode promo Yes/No → 1/0
    │  - Align columns to training schema
    ▼
Drop target column (delivery_time or delivery_cost)
    │
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
| Decision Tree (default) | 6.855 | 0.544 | 120.432 | 10.974 |
| Decision Tree (tuned) | 6.303 | 0.634 | 96.768 | 9.837 |
| Random Forest (default) | 5.539 | 0.702 | 78.762 | 8.875 |
| **Random Forest (tuned)** | **5.436** | **0.701** | **78.922** | **8.884** |

The tuned Random Forest achieves the best overall performance — predictions deviate by ~5.4 minutes on average and explain 70.1% of variance in delivery times.

### Delivery Cost Prediction

| Model | MAE (SGD) | R² | MSE | RMSE (SGD) |
|---|---|---|---|---|
| Decision Tree (default) | 2.913 | 0.405 | 25.777 | 5.077 |
| Decision Tree (tuned) | 2.836 | 0.465 | 23.210 | 4.818 |
| Random Forest (default) | 2.495 | 0.609 | 16.949 | 4.117 |
| **Random Forest (tuned)** | **2.494** | **0.611** | **16.869** | **4.111** |

The tuned Random Forest achieves the best performance — predictions deviate by ~$2.49 on average and explain 61.1% of variance in delivery costs.

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
├── requirements.txt     # Python dependencies
├── test_dataset.csv     # Sample dataset for testing
├── unique_cuisines.txt  # List of all cuisine types seen during training
├── other_columns.txt    # Categorical columns to one-hot encode
└── final_columns.txt    # Ordered column list expected by the models
```

---

## Dependencies

- `streamlit`
- `scikit-learn==1.3.2`
- `pandas`
- `requests`

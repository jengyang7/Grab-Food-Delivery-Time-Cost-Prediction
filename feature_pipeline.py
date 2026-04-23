import json
import math
import re
from collections import Counter

import numpy as np
import pandas as pd


OPTIONAL_TARGET_COLUMNS = ("delivery_time", "delivery_cost")
REQUIRED_INPUT_COLUMNS = {
    "name",
    "cuisine",
    "lat",
    "lon",
    "radius",
    "rating",
    "reviews_nr",
    "delivery_options",
    "promo",
    "loc_type",
    "delivery_by",
    "region",
}

# Columns that can be provided either in raw form (and parsed) or already engineered.
# This makes it easier to train/infer from a "minimal" CSV that precomputes features.
ALTERNATIVE_INPUT_REQUIREMENTS = (
    # Building / mall features come from address parsing unless building_name is provided.
    ({"address"}, {"building_name"}),
    # Opening-hours features can be parsed from opening_hours unless provided.
    ({"opening_hours"}, {"total_weekly_hours", "opens_early", "closes_late", "consistent_hours"}),
)
KNOWN_NAME_FIXES = {
    13754: "A-One",
    13756: "Fish & Co.",
    13787: "Shake Shack",
    13788: "The Coffee Bean & Tea Leaf",
    13795: "The Coffee Bean & Tea Leaf",
    13860: "FOODQA-BJ_SG's Sea Resto GKMM Mart",
}
KNOWN_CUISINE_FIXES = {
    12076: '["Western"]',
    14058: '["Asian", "Thai"]',
}
TOP_CUISINE_COUNT = 20
TOP_BUILDING_COUNT = 100


def validate_input_columns(df, require_targets=False):
    """
    Validate that the DataFrame has the minimum required columns.

    Supports alternatives (raw -> engineered) for certain feature groups:
    - address OR building_name
    - opening_hours OR opening-hours engineered columns
    Promo parsing does NOT impose extra requirements: `prepare_raw_dataframe()` will
    create `promo_code` from `promo` when it is missing.
    """
    cols = set(df.columns)
    required = set(REQUIRED_INPUT_COLUMNS)
    if require_targets:
        required.update(OPTIONAL_TARGET_COLUMNS)

    missing_base = sorted(required - cols)
    if missing_base:
        raise ValueError(f"Uploaded CSV is missing required columns: {missing_base}")

    missing_alternatives = []
    for raw_set, engineered_set in ALTERNATIVE_INPUT_REQUIREMENTS:
        if not (raw_set <= cols or engineered_set <= cols):
            missing_alternatives.append({"one_of": [sorted(raw_set), sorted(engineered_set)]})
    if missing_alternatives:
        raise ValueError(
            "Uploaded CSV is missing required columns. Provide at least one option for each group: "
            f"{missing_alternatives}"
        )


def _sanitize_token(text):
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(text)).strip("_") or "value"


def _build_feature_name_map(prefix, values):
    mapping = {}
    used = set()
    for value in values:
        base = f"{prefix}__{_sanitize_token(value)}"
        name = base
        counter = 2
        while name in used:
            name = f"{base}_{counter}"
            counter += 1
        mapping[str(value)] = name
        used.add(name)
    return mapping


def _apply_known_training_fixes(df):
    df = df.copy()
    for idx, value in KNOWN_NAME_FIXES.items():
        if idx in df.index:
            df.at[idx, "name"] = value
    for idx, value in KNOWN_CUISINE_FIXES.items():
        if idx in df.index:
            df.at[idx, "cuisine"] = value
    return df


def parse_cuisine_list(value):
    if pd.isna(value):
        return []
    cleaned = str(value).strip("[]").replace("'", "").replace('"', "")
    return [item.strip() for item in cleaned.split(",") if item.strip()]


def extract_building_name(address):
    text = "Unknown" if pd.isna(address) else str(address).strip()
    for delimiter in (" - ", "-", "@", "–"):
        if delimiter in text:
            parts = [part.strip() for part in text.split(delimiter) if part.strip()]
            if len(parts) > 1:
                text = parts[-1]
                break
    text = re.sub(r"\[Islandwide Delivery\]", "", text).strip()
    text = re.sub(r"\b[Vv]ivo\s?[Cc]ity\b|\b[Vv]ivocity\b|\b[Vv]ivo\b", "VivoCity", text)
    text = re.sub(
        r"\b[Hh]abourfront\s[cC]entre\b|\b[Hh]arbourfront\s[cC]entre\b",
        "HarbourFront Centre",
        text,
    )
    return "VivoCity" if "VivoCity" in text else (text or "Unknown")


def _parse_time_to_hours(value):
    hours, minutes = value.split(":")
    return int(hours) + int(minutes) / 60


def _parse_range(token):
    """Parse a single 'HH:MM-HH:MM' token. Returns (start_h, end_h) or None."""
    if "-" not in token:
        return None
    halves = token.split("-", 1)
    try:
        start_h = _parse_time_to_hours(halves[0].strip())
        end_h = _parse_time_to_hours(halves[1].strip())
    except (ValueError, AttributeError):
        return None
    if end_h < start_h:
        end_h += 24
    return start_h, end_h


def _duration_hours(spec):
    # A day spec may have multiple space-separated ranges, e.g. "00:00-01:00 10:30-23:59"
    if not spec or spec.strip().lower() == "closed":
        return 0.0
    total = 0.0
    for token in spec.split():
        parsed = _parse_range(token)
        if parsed:
            start_h, end_h = parsed
            total += max(0.0, end_h - start_h)
    return total


def extract_opening_features(value):
    if pd.isna(value) or not str(value).strip():
        return {
            "total_weekly_hours": 0.0,
            "opens_early": 0,
            "closes_late": 0,
            "consistent_hours": 0,
        }
    try:
        data = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {
            "total_weekly_hours": 0.0,
            "opens_early": 0,
            "closes_late": 0,
            "consistent_hours": 0,
        }

    schedules = []
    opens_early = 0
    closes_late = 0
    total_weekly_hours = 0.0
    for day in ("mon", "tue", "wed", "thu", "fri", "sat", "sun"):
        spec = str(data.get(day, "") or "")
        schedules.append(spec)
        total_weekly_hours += _duration_hours(spec)
        for token in spec.split():
            parsed = _parse_range(token)
            if parsed:
                start_h, end_h = parsed
                opens_early = int(opens_early or start_h < 10)
                closes_late = int(closes_late or end_h > 21)

    return {
        "total_weekly_hours": round(total_weekly_hours, 2),
        "opens_early": opens_early,
        "closes_late": closes_late,
        "consistent_hours": int(len({spec for spec in schedules if spec}) <= 1 and any(schedules)),
    }


def extract_promo_features(value):
    text = "" if pd.isna(value) else str(value)
    lower = text.lower()
    has_free_delivery = int("free delivery" in lower)
    has_min_spend = int(
        bool(re.search(r"min(?:imum)?\s*spend|orders?\s+over|above\s+s?\$|with\s+min", lower))
    )
    if has_free_delivery:
        promo_type = "free_delivery"
    elif "%" in lower or "percent" in lower:
        promo_type = "percentage"
    elif re.search(r"\$\s*\d|s\$\s*\d|\d+\s*(?:sgd|dollars?)\s+off|\boff\b", lower):
        promo_type = "dollar_off"
    else:
        promo_type = "none"

    return {
        "has_free_delivery_promo": has_free_delivery,
        "has_min_spend_condition": has_min_spend,
        "promo_discount_type": promo_type,
    }


def prepare_raw_dataframe(df, training=False):
    validate_input_columns(df, require_targets=training)
    prepared = df.copy()
    if training:
        prepared = _apply_known_training_fixes(prepared)

    # Back-compat: some datasets only have "promo" text. Use it as promo_code if needed.
    if "promo_code" not in prepared.columns:
        prepared["promo_code"] = prepared["promo"]

    prepared = prepared[(prepared["rating"].isna()) | prepared["rating"].between(0, 5)].copy()
    if training:
        prepared = prepared.dropna(subset=list(OPTIONAL_TARGET_COLUMNS)).copy()

    prepared["name"] = prepared["name"].fillna("Unknown Restaurant").astype(str).str.strip()
    if "address" in prepared.columns:
        prepared["address"] = prepared["address"].fillna("Unknown").astype(str)
    prepared["region"] = prepared["region"].fillna("Unknown").astype(str)
    prepared["delivery_options"] = prepared["delivery_options"].fillna("Unknown").astype(str)
    prepared["loc_type"] = prepared["loc_type"].fillna("Unknown").astype(str)
    prepared["delivery_by"] = prepared["delivery_by"].fillna("Unknown").astype(str)
    prepared["promo_code"] = prepared["promo_code"].fillna("-")
    prepared["promo"] = prepared["promo"].apply(
        lambda value: "Yes" if pd.notna(value) and str(value).strip() and str(value).strip().lower() != "no" else "No"
    )
    prepared["rating"] = pd.to_numeric(prepared["rating"], errors="coerce")
    prepared["reviews_nr"] = pd.to_numeric(prepared["reviews_nr"], errors="coerce")
    prepared["radius"] = pd.to_numeric(prepared["radius"], errors="coerce")
    prepared["lat"] = pd.to_numeric(prepared["lat"], errors="coerce")
    prepared["lon"] = pd.to_numeric(prepared["lon"], errors="coerce")

    # Building name: if provided, trust it; otherwise derive from address.
    if "building_name" in prepared.columns and prepared["building_name"].notna().any():
        prepared["building_name"] = prepared["building_name"].fillna("Unknown").astype(str)
    else:
        if "address" not in prepared.columns:
            # Should not happen due to validation, but keep a safe fallback.
            prepared["address"] = "Unknown"
        prepared["building_name"] = prepared["address"].apply(extract_building_name)

    prepared["cuisine_list"] = prepared["cuisine"].apply(parse_cuisine_list)

    if training and "delivery_cost" in prepared.columns:
        prepared["delivery_cost"] = pd.to_numeric(prepared["delivery_cost"], errors="coerce") / 100
        prepared["delivery_time"] = pd.to_numeric(prepared["delivery_time"], errors="coerce")

    # Ensure engineered opening-hours and promo columns exist if provided.
    # If user supplies them, keep them as-is; if not, they'll be computed later.
    for col in ("total_weekly_hours", "opens_early", "closes_late", "consistent_hours"):
        if col in prepared.columns:
            prepared[col] = pd.to_numeric(prepared[col], errors="coerce")
    for col in ("has_free_delivery_promo", "has_min_spend_condition"):
        if col in prepared.columns:
            prepared[col] = pd.to_numeric(prepared[col], errors="coerce").fillna(0).astype(int)
    if "promo_discount_type" in prepared.columns:
        prepared["promo_discount_type"] = prepared["promo_discount_type"].fillna("none").astype(str)

    return prepared.reset_index(drop=True)


def build_feature_artifacts(train_df):
    cuisine_counts = Counter(cuisine for cuisines in train_df["cuisine_list"] for cuisine in cuisines)
    top_cuisines = [name for name, _ in cuisine_counts.most_common(TOP_CUISINE_COUNT)]
    frequent_buildings = train_df["building_name"].value_counts().head(TOP_BUILDING_COUNT).index.tolist()
    chain_names = train_df["name"].value_counts()
    chain_names = chain_names[chain_names >= 3].index.tolist()
    region_centroids = train_df.groupby("region")[["lat", "lon"]].mean().round(6).to_dict("index")
    lat_grid = np.floor(train_df["lat"] / 0.01).fillna(-999).astype(int)
    lon_grid = np.floor(train_df["lon"] / 0.01).fillna(-999).astype(int)
    grid_density = Counter(f"{lat}_{lon}" for lat, lon in zip(lat_grid, lon_grid))

    levels = {
        "mall_or_building_name": frequent_buildings + ["Other"],
        "promo_discount_type": sorted({"free_delivery", "percentage", "dollar_off", "none"}),
        "delivery_options": sorted(train_df["delivery_options"].fillna("Unknown").astype(str).unique()),
        "loc_type": sorted(train_df["loc_type"].fillna("Unknown").astype(str).unique()),
        "delivery_by": sorted(train_df["delivery_by"].fillna("Unknown").astype(str).unique()),
        "region": sorted(train_df["region"].fillna("Unknown").astype(str).unique()),
    }
    categorical_maps = {
        column: _build_feature_name_map(column, values) for column, values in levels.items()
    }

    return {
        "top_cuisines": top_cuisines,
        "top_cuisine_feature_names": _build_feature_name_map("cuisine", top_cuisines + ["Other"]),
        "frequent_buildings": frequent_buildings,
        "chain_names": chain_names,
        "region_centroids": region_centroids,
        "grid_density": dict(grid_density),
        "categorical_levels": levels,
        "categorical_feature_names": categorical_maps,
    }


def feature_group_name(feature_name):
    if feature_name.startswith("cuisine__"):
        return "Cuisine"
    if feature_name.startswith("mall_or_building_name__"):
        return "Mall / Building"
    if feature_name.startswith(("delivery_options__", "loc_type__", "delivery_by__", "region__")):
        return "Service / Region"
    if feature_name.startswith("promo_discount_type__") or feature_name in {
        "promo",
        "has_free_delivery_promo",
        "has_min_spend_condition",
    }:
        return "Promotions"
    if feature_name in {"lat", "lon", "dist_to_region_centroid", "grid_restaurant_density"}:
        return "Geospatial"
    if feature_name in {"total_weekly_hours", "opens_early", "closes_late", "consistent_hours"}:
        return "Opening Hours"
    if feature_name in {"radius", "rating", "reviews_nr"}:
        return "Core Numeric"
    if feature_name == "is_chain":
        return "Restaurant Identity"
    return "Other"


def transform_features(df, artifacts):
    train_medians = {
        "radius": float(pd.to_numeric(df["radius"], errors="coerce").median())
        if pd.to_numeric(df["radius"], errors="coerce").notna().any()
        else 0.0,
        "rating": float(pd.to_numeric(df["rating"], errors="coerce").mean())
        if pd.to_numeric(df["rating"], errors="coerce").notna().any()
        else 0.0,
        "reviews_nr": float(pd.to_numeric(df["reviews_nr"], errors="coerce").median())
        if pd.to_numeric(df["reviews_nr"], errors="coerce").notna().any()
        else 0.0,
        "lat": float(pd.to_numeric(df["lat"], errors="coerce").median())
        if pd.to_numeric(df["lat"], errors="coerce").notna().any()
        else 0.0,
        "lon": float(pd.to_numeric(df["lon"], errors="coerce").median())
        if pd.to_numeric(df["lon"], errors="coerce").notna().any()
        else 0.0,
    }
    return transform_features_with_fill_values(df, artifacts, train_medians)


def transform_features_with_fill_values(df, artifacts, fill_values):
    base_features = pd.DataFrame(
        {
            "radius": pd.to_numeric(df["radius"], errors="coerce").fillna(fill_values["radius"]),
            "rating": pd.to_numeric(df["rating"], errors="coerce").fillna(fill_values["rating"]),
            "reviews_nr": pd.to_numeric(df["reviews_nr"], errors="coerce").fillna(fill_values["reviews_nr"]),
            "promo": (df["promo"].astype(str).str.lower() == "yes").astype(int),
            "is_chain": df["name"].isin(set(artifacts["chain_names"])).astype(int),
            "lat": pd.to_numeric(df["lat"], errors="coerce").fillna(fill_values["lat"]),
            "lon": pd.to_numeric(df["lon"], errors="coerce").fillna(fill_values["lon"]),
        },
        index=df.index,
    )

    centroids = artifacts["region_centroids"]
    geo_features = pd.DataFrame(index=df.index)
    geo_features["dist_to_region_centroid"] = df.apply(
        lambda row: _distance_to_region_centroid(row, centroids),
        axis=1,
    )
    lat_grid = np.floor(pd.to_numeric(df["lat"], errors="coerce") / 0.01).fillna(-999).astype(int)
    lon_grid = np.floor(pd.to_numeric(df["lon"], errors="coerce") / 0.01).fillna(-999).astype(int)
    geo_features["grid_restaurant_density"] = [
        artifacts["grid_density"].get(f"{lat}_{lon}", 0) for lat, lon in zip(lat_grid, lon_grid)
    ]

    # Opening hours: if engineered columns are present, use them; else parse opening_hours text.
    opening_cols = ("total_weekly_hours", "opens_early", "closes_late", "consistent_hours")
    if all(col in df.columns for col in opening_cols):
        opening_features = df[list(opening_cols)].copy()
        opening_features["total_weekly_hours"] = pd.to_numeric(
            opening_features["total_weekly_hours"], errors="coerce"
        ).fillna(0.0)
        for col in ("opens_early", "closes_late", "consistent_hours"):
            opening_features[col] = pd.to_numeric(opening_features[col], errors="coerce").fillna(0).astype(int)
    else:
        if "opening_hours" not in df.columns:
            # Safe fallback: treat as always closed/unknown.
            opening_features = pd.DataFrame(
                {"total_weekly_hours": 0.0, "opens_early": 0, "closes_late": 0, "consistent_hours": 0},
                index=df.index,
            )
        else:
            opening_features = df["opening_hours"].apply(extract_opening_features).apply(pd.Series)

    # Promo: if engineered columns are present, use them; else parse promo_code text.
    promo_base_cols = ("has_free_delivery_promo", "has_min_spend_condition", "promo_discount_type")
    if all(col in df.columns for col in promo_base_cols):
        promo_features = df[list(promo_base_cols)].copy()
        promo_features["has_free_delivery_promo"] = pd.to_numeric(
            promo_features["has_free_delivery_promo"], errors="coerce"
        ).fillna(0).astype(int)
        promo_features["has_min_spend_condition"] = pd.to_numeric(
            promo_features["has_min_spend_condition"], errors="coerce"
        ).fillna(0).astype(int)
        promo_features["promo_discount_type"] = promo_features["promo_discount_type"].fillna("none").astype(str)
    else:
        if "promo_code" not in df.columns:
            promo_features = pd.DataFrame(
                {"has_free_delivery_promo": 0, "has_min_spend_condition": 0, "promo_discount_type": "none"},
                index=df.index,
            )
        else:
            promo_features = df["promo_code"].apply(extract_promo_features).apply(pd.Series)
    cuisine_features = pd.DataFrame(index=df.index)

    cuisine_names = artifacts["top_cuisines"]
    cuisine_feature_names = artifacts["top_cuisine_feature_names"]
    for cuisine in cuisine_names:
        cuisine_features[cuisine_feature_names[cuisine]] = df["cuisine_list"].apply(lambda items: int(cuisine in items))
    cuisine_features[cuisine_feature_names["Other"]] = df["cuisine_list"].apply(
        lambda items: int(any(cuisine not in cuisine_names for cuisine in items) or not items)
    )

    categorical_values = {
        "mall_or_building_name": df["building_name"].where(
            df["building_name"].isin(set(artifacts["frequent_buildings"])),
            "Other",
        ),
        "promo_discount_type": promo_features["promo_discount_type"].astype(str),
        "delivery_options": df["delivery_options"].fillna("Unknown").astype(str),
        "loc_type": df["loc_type"].fillna("Unknown").astype(str),
        "delivery_by": df["delivery_by"].fillna("Unknown").astype(str),
        "region": df["region"].fillna("Unknown").astype(str),
    }
    categorical_feature_data = {}
    for column, mapping in artifacts["categorical_feature_names"].items():
        series = categorical_values[column].where(
            categorical_values[column].isin(set(artifacts["categorical_levels"][column])),
            "Other" if "Other" in artifacts["categorical_levels"][column] else categorical_values[column],
        )
        for value, feature_name in mapping.items():
            categorical_feature_data[feature_name] = (series == value).astype(int)
    categorical_features = pd.DataFrame(categorical_feature_data, index=df.index)

    transformed = pd.concat(
        [
            base_features,
            geo_features,
            opening_features,
            promo_features.drop(columns=["promo_discount_type"]),
            cuisine_features,
            categorical_features,
        ],
        axis=1,
    )
    return transformed.fillna(0)


def align_features(feature_df, final_columns):
    aligned = feature_df.copy()
    for column in final_columns:
        if column not in aligned.columns:
            aligned[column] = 0
    return aligned[final_columns]


def load_feature_artifacts(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_feature_artifacts(path, artifacts):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(artifacts, handle, indent=2, sort_keys=True)


def read_final_columns(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def write_lines(path, values):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(values) + "\n")


def transform_for_inference(df, artifacts, final_columns, fill_values):
    prepared = prepare_raw_dataframe(df, training=False)
    features = transform_features_with_fill_values(prepared, artifacts, fill_values)
    return align_features(features, final_columns)


def _distance_to_region_centroid(row, centroids):
    region = row.get("region")
    centroid = centroids.get(str(region))
    lat = pd.to_numeric(row.get("lat"), errors="coerce")
    lon = pd.to_numeric(row.get("lon"), errors="coerce")
    if not centroid or pd.isna(lat) or pd.isna(lon):
        return 0.0
    return math.sqrt((lat - centroid["lat"]) ** 2 + (lon - centroid["lon"]) ** 2) * 111

import pandas as pd
import numpy as np

# Enhanced Synthetic Data Generator with Warm-up, Trend, Seasonality, Promotions, and Autocorrelation
np.random.seed(42)
warehouses = ['G1', 'G2']
outlets = [f'O{i}' for i in range(1, 9)]  # O1 to O8
products = ['P1', 'P2', 'P3']

# 1. Parameters
start_date = pd.to_datetime('2025-01-01')
n_days = 252  # days of recorded history
early_warmup = 30  # burn-in period for AR term

# Signal parameters
base_low, base_high = 10, 50         # base random demand
trend_rate = 0.02                    # daily trend increment
weekday_bonus = 10                   # weekday seasonal boost
weekend_penalty = -5                 # weekend seasonal penalty
promo_bonus_value = 20               # extra on promo days
alpha = 0.7                          # AR(1) coefficient
noise_sd = 3                         # white noise std dev

# All dates including warm-up
dates_warmup = pd.date_range(start_date - pd.Timedelta(days=early_warmup),
                              periods=early_warmup + n_days, freq='D')
# Select promo dates within recorded portion
promo_dates = np.random.choice(dates_warmup[-n_days:], size=12, replace=False)

# Initialize prev_qty for AR
prev_qty = {(o, p): None for o in outlets for p in products}
records = []

# 2. Generate sequence with warm-up (skip recording until start_date)
for date in dates_warmup:
    day_idx = (date - start_date).days
    # Components
    trend_factor = max(0, trend_rate * day_idx)
    seasonal = weekday_bonus if date.weekday() < 5 else weekend_penalty
    promo_flag = int(date in promo_dates)
    promo_bonus = promo_bonus_value if promo_flag else 0

    for outlet in outlets:
        for product in products:
            base = np.random.randint(base_low, base_high + 1)
            # AR term
            last = prev_qty[(outlet, product)]
            ar_term = alpha * last if last is not None else 0
            noise = np.random.normal(0, noise_sd)
            qty_calc = base + trend_factor + seasonal + promo_bonus + ar_term + noise
            qty_box = max(1, int(qty_calc))
            # Update prev
            prev_qty[(outlet, product)] = qty_box
            # Record only if past warm-up
            if date >= start_date:
                records.append({
                    'date': date.date(),
                    'outlet': outlet,
                    'product': product,
                    'qty_box': qty_box,
                    'promo': promo_flag
                })

# 3. Vehicles: unchanged
df_vehicles = pd.DataFrame([
    {'vehicle_name': 'T1', 'warehouse': 'G1', 'vehicle_type': 'Truk CDE', 'max_weight_kg': 5000},
    {'vehicle_name': 'T2', 'warehouse': 'G1', 'vehicle_type': 'Truk CDD',   'max_weight_kg': 6000},
    {'vehicle_name': 'T3', 'warehouse': 'G1', 'vehicle_type': 'Pick Up',   'max_weight_kg': 2000},
    {'vehicle_name': 'T4', 'warehouse': 'G2', 'vehicle_type': 'Truk CDE', 'max_weight_kg': 5000},
    {'vehicle_name': 'T5', 'warehouse': 'G2', 'vehicle_type': 'Truk CDD',   'max_weight_kg': 6000},
    {'vehicle_name': 'T6', 'warehouse': 'G1', 'vehicle_type': 'Pick Up',   'max_weight_kg': 2000},
])

# 4. Distances: unchanged
records_dist = []
for wh in warehouses:
    for outlet in outlets:
        records_dist.append({
            'warehouse': wh,
            'outlet': outlet,
            'distance_km': round(np.random.uniform(10, 50), 1)
        })
df_distances = pd.DataFrame(records_dist)

# 5. Product weights: unchanged
df_product_weights = pd.DataFrame([
    {'product': 'P1', 'weight_per_box_kg': 20},
    {'product': 'P2', 'weight_per_box_kg': 15},
    {'product': 'P3', 'weight_per_box_kg': 25},
])

# 6. Save to Excel
file_path = 'D:/project_mostrans/data/historical_sales.xlsx'
with pd.ExcelWriter(file_path) as writer:
    pd.DataFrame(records).to_excel(writer, sheet_name='historical_orders', index=False)
    df_vehicles.to_excel(writer, sheet_name='vehicles', index=False)
    df_distances.to_excel(writer, sheet_name='distances', index=False)
    df_product_weights.to_excel(writer, sheet_name='product_weights', index=False)

print(f"Synthetic data with warm-up saved to {file_path}")

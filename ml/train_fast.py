"""
Fast Model Training on Texas Traffic Data
Trains on 500K samples for quick, high-quality models
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
from pathlib import Path
import json
import time

def main():
    print("=" * 80)
    print("🚀 FAST MODEL TRAINING - Texas Traffic Data")
    print("=" * 80)
    
    ml_dir = Path(__file__).parent
    data_dir = ml_dir / "traffic_data_10M"
    models_dir = ml_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Load first 5 batches (500K samples) - fast and efficient
    print("\n📂 Loading traffic data...")
    csv_files = sorted(list(data_dir.glob("traffic_batch_*.csv")))[:5]
    
    dfs = []
    for i, file in enumerate(csv_files):
        print(f"  Loading batch {i+1}/5: {file.name}")
        df = pd.read_csv(file)
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"\n✅ Loaded {len(df):,} samples")
    
    # Prepare features
    print("\n🔧 Engineering features...")
    feature_columns = [
        'latitude', 'longitude', 'hour', 'day_of_week',
        'weather_condition', 'num_lanes', 'speed_limit',
        'distance_from_center_km', 'is_weekend', 'is_rush_hour'
    ]
    
    X = df[feature_columns].copy()
    
    # Add engineered features
    X['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    X['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    X['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    X['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    X['location_type_encoded'] = df['location_type'].astype('category').cat.codes
    X['rush_hour_lanes'] = X['is_rush_hour'] * X['num_lanes']
    X['weekend_distance'] = X['is_weekend'] * X['distance_from_center_km']
    
    print(f"✅ Features: {X.shape[1]} dimensions")
    
    # Train models for each target
    targets = {
        'congestion': df['congestion_level'],
        'vehicle_count': df['vehicle_count'],
        'average_speed': df['average_speed'],
        'travel_time_index': df['travel_time_index']
    }
    
    print("\n" + "=" * 80)
    print("🤖 TRAINING MODELS")
    print("=" * 80)
    
    model_info = {}
    
    for target_name, y in targets.items():
        print(f"\n🎯 Training {target_name.upper()} model...")
        start_time = time.time()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=25,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"  ✅ Training complete in {training_time:.1f}s")
        print(f"  📊 R² Score: {r2:.4f} ({r2*100:.1f}%)")
        print(f"  📊 MAE: {mae:.4f}")
        print(f"  📊 RMSE: {rmse:.4f}")
        
        # Save model
        model_path = models_dir / f"{target_name}_real_data_model.pkl"
        joblib.dump(model, model_path)
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  💾 Saved → {model_path.name} ({size_mb:.1f} MB)")
        
        model_info[target_name] = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'training_time': training_time,
            'n_estimators': 100,
            'training_samples': len(X_train)
        }
    
    # Save feature info
    feature_info = {
        'features': list(X.columns),
        'version': '4.1.0',
        'trained_date': str(pd.Timestamp.now()),
        'data_source': 'texas_synthetic_10M',
        'training_samples': len(df)
    }
    
    feature_path = models_dir / "real_data_features.json"
    with open(feature_path, 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    info_path = models_dir / "real_data_model_info.json"
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Dataset: {len(df):,} samples")
    print(f"🎯 Models trained: {len(targets)}")
    print(f"\n📈 Average R² Score: {np.mean([info['r2'] for info in model_info.values()])*100:.1f}%")
    print(f"\n✨ Models ready for production!")
    print(f"   Location: {models_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()

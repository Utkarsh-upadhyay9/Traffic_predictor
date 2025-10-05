"""
Train Location-Based Traffic Prediction Models
Uses latitude, longitude, hour, and day_of_week as primary features
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
from pathlib import Path
import json

def train_models():
    """Train location-aware traffic prediction models"""
    
    print("=" * 60)
    print("🤖 TRAINING LOCATION-BASED TRAFFIC MODELS")
    print("=" * 60)
    
    # Load data
    data_path = Path(__file__).parent / "real_world_traffic_data.csv"
    print(f"\n📂 Loading data from {data_path}...")
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            "Please run generate_real_world_data.py first!"
        )
    
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df):,} samples")
    
    # Feature engineering
    print("\n🔧 Engineering features...")
    
    # Core features: latitude, longitude, hour, day_of_week
    # Additional contextual features (ENHANCED for v2.0)
    feature_columns = [
        'latitude',
        'longitude',
        'hour',
        'day_of_week',
        'distance_from_campus_km',
        'location_type_encoded',
        'num_lanes',
        'road_capacity',
        'weather_condition',
        'is_holiday',
        'has_event',       # NEW
        'has_incident',    # NEW
        'speed_limit',     # NEW
    ]
    
    # Simplified feature set for quick predictions (just location + time)
    simple_features = [
        'latitude',
        'longitude',
        'hour',
        'day_of_week',
    ]
    
    X_full = df[feature_columns].copy()
    X_simple = df[simple_features].copy()
    
    # Add time-based features
    X_full['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    X_full['is_rush_hour'] = ((df['hour'] >= 6) & (df['hour'] <= 9) | 
                               (df['hour'] >= 15) & (df['hour'] <= 18)).astype(int)
    X_full['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    X_full['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    X_simple['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    X_simple['is_rush_hour'] = ((df['hour'] >= 6) & (df['hour'] <= 9) | 
                                 (df['hour'] >= 15) & (df['hour'] <= 18)).astype(int)
    X_simple['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    X_simple['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Targets
    targets = {
        'congestion': df['congestion_level'],
        'travel_time': df['travel_time_min'],
        'vehicle_count': df['predicted_vehicle_count'],
    }
    
    print(f"✅ Full feature set: {X_full.shape[1]} features")
    print(f"✅ Simple feature set: {X_simple.shape[1]} features (for quick predictions)")
    
    # Train models
    models = {}
    model_info = {}
    
    for target_name, y in targets.items():
        print(f"\n{'='*60}")
        print(f"🎯 Training {target_name.upper()} models...")
        print(f"{'='*60}")
        
        # Split data
        X_train_full, X_test_full, y_train, y_test = train_test_split(
            X_full, y, test_size=0.2, random_state=42
        )
        
        X_train_simple, X_test_simple, _, _ = train_test_split(
            X_simple, y, test_size=0.2, random_state=42
        )
        
        # Train full model (Random Forest) - ENHANCED PARAMETERS
        print(f"\n  Training full Random Forest model (enhanced)...")
        rf_model = RandomForestRegressor(
            n_estimators=300,      # Increased from 200
            max_depth=30,          # Increased from 25
            min_samples_split=4,   # More aggressive splitting
            min_samples_leaf=2,
            max_features='sqrt',   # Better generalization
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_full, y_train)
        
        # Train simple model (Gradient Boosting - faster predictions) - ENHANCED
        print(f"  Training simple Gradient Boosting model (enhanced)...")
        gb_model = GradientBoostingRegressor(
            n_estimators=200,      # Increased from 150
            max_depth=10,          # Increased from 8
            learning_rate=0.08,    # Slightly lower for better convergence
            subsample=0.8,         # Add subsampling for robustness
            random_state=42
        )
        gb_model.fit(X_train_simple, y_train)
        
        # Evaluate full model
        y_pred_full = rf_model.predict(X_test_full)
        mae_full = mean_absolute_error(y_test, y_pred_full)
        r2_full = r2_score(y_test, y_pred_full)
        rmse_full = np.sqrt(mean_squared_error(y_test, y_pred_full))
        
        # Evaluate simple model
        y_pred_simple = gb_model.predict(X_test_simple)
        mae_simple = mean_absolute_error(y_test, y_pred_simple)
        r2_simple = r2_score(y_test, y_pred_simple)
        rmse_simple = np.sqrt(mean_squared_error(y_test, y_pred_simple))
        
        print(f"\n  📊 Full Model Performance:")
        print(f"    MAE:  {mae_full:.4f}")
        print(f"    RMSE: {rmse_full:.4f}")
        print(f"    R²:   {r2_full:.4f} ({r2_full*100:.1f}%)")
        
        print(f"\n  📊 Simple Model Performance:")
        print(f"    MAE:  {mae_simple:.4f}")
        print(f"    RMSE: {rmse_simple:.4f}")
        print(f"    R²:   {r2_simple:.4f} ({r2_simple*100:.1f}%)")
        
        # Feature importance (full model)
        feature_importance = pd.DataFrame({
            'feature': X_full.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n  🔝 Top 5 Important Features:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"    {row['feature']:25s}: {row['importance']:.4f}")
        
        # Store models
        models[f'{target_name}_full'] = rf_model
        models[f'{target_name}_simple'] = gb_model
        
        model_info[target_name] = {
            'full_model': {
                'mae': float(mae_full),
                'rmse': float(rmse_full),
                'r2': float(r2_full),
                'feature_count': X_full.shape[1],
            },
            'simple_model': {
                'mae': float(mae_simple),
                'rmse': float(rmse_simple),
                'r2': float(r2_simple),
                'feature_count': X_simple.shape[1],
            },
            'feature_importance': feature_importance.head(10).to_dict('records')
        }
    
    # Save models
    print(f"\n{'='*60}")
    print("💾 Saving models...")
    print(f"{'='*60}")
    
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    for model_name, model in models.items():
        model_path = models_dir / f"{model_name}_location_model.pkl"
        joblib.dump(model, model_path)
        print(f"  ✅ Saved {model_name} → {model_path.name}")
    
    # Save feature columns
    feature_info = {
        'full_features': list(X_full.columns),
        'simple_features': list(X_simple.columns),
        'version': '2.0.0',
        'trained_date': str(pd.Timestamp.now()),
    }
    
    feature_path = models_dir / "location_features.json"
    with open(feature_path, 'w') as f:
        json.dump(feature_info, f, indent=2)
    print(f"  ✅ Saved feature info → {feature_path.name}")
    
    # Save model performance info
    info_path = models_dir / "location_model_info.json"
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"  ✅ Saved model info → {info_path.name}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'='*60}")
    print("\n📈 Model Performance Summary:")
    for target_name, info in model_info.items():
        print(f"\n  {target_name.upper()}:")
        print(f"    Full Model R²:   {info['full_model']['r2']*100:.1f}%")
        print(f"    Simple Model R²: {info['simple_model']['r2']*100:.1f}%")
    
    print("\n🎯 Models ready for location-based predictions!")
    print("   Use latitude, longitude, hour, and day_of_week for predictions")

if __name__ == "__main__":
    train_models()

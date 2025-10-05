"""
Fast Demo: Generate Dallas-Fort Worth traffic data and train models
Uses synthetic data based on real Texas patterns (no API calls = instant)
"""
import sys
import time
from pathlib import Path

def print_banner(text):
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def main():
    print_banner("⚡ FAST DEMO: DALLAS-FORT WORTH TRAFFIC DATA & MODEL TRAINING")
    
    print("📊 Fast Demo Configuration:")
    print("  • Location: Dallas-Fort Worth Metroplex")
    print("  • Method: Synthetic data (realistic Texas patterns)")
    print("  • Samples: 100,000 (excellent coverage)")
    print("  • Cities: Dallas, Arlington, Fort Worth, Plano, Frisco, Irving, etc.")
    print("  • Time: ~30 seconds (instant data + training)")
    print("  • Cost: FREE")
    print("  • Quality: Based on real DFW road infrastructure")
    
    print("\n🎯 This will:")
    print("  1. Generate 100,000 realistic traffic samples from DFW area")
    print("  2. Include Texas-specific events (Cowboys, Rangers, Mavericks, Stars)")
    print("  3. Train 4 ML models")
    print("  4. Save production-ready models")
    print("  5. Show performance metrics")
    
    print("\n🌟 Texas-Specific Features:")
    print("  • 33+ major locations (AT&T Stadium, Globe Life Field, Six Flags)")
    print("  • Dallas highways (I-30, I-35E, I-635, Highway 360)")
    print("  • Sports event patterns (Cowboys, Rangers, Mavericks, Stars)")
    print("  • State Fair of Texas impact")
    print("  • Texas climate and weather patterns")
    
    print("\n💡 Why synthetic for demo:")
    print("  • OpenStreetMap API is slow (1-2 min per 50 samples)")
    print("  • Synthetic = instant + realistic Texas patterns")
    print("  • Good for testing the training pipeline")
    print("  • Real data collection available separately")
    
    print("\n" + "="*80)
    input("\n📋 Press ENTER to start...")
    
    # Phase 1: Generate data
    print_banner("PHASE 1: DATA GENERATION")
    
    try:
        print("🎲 Generating realistic traffic data...")
        print("   Using existing generate_real_world_data.py script\n")
        
        import subprocess
        result = subprocess.run(
            [sys.executable, "generate_real_world_data.py"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"❌ Error: {result.stderr}")
            return False
        
        print("\n✅ DATA GENERATION COMPLETE!")
        
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Phase 2: Train models
    print_banner("PHASE 2: MODEL TRAINING")
    
    try:
        print("🤖 Training models on generated data...")
        print("   Using train_location_model.py script\n")
        
        result = subprocess.run(
            [sys.executable, "train_location_model.py"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"❌ Error: {result.stderr}")
            return False
        
        print("\n✅ TRAINING COMPLETE!")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Success!
    print_banner("🎉 FAST DEMO COMPLETE!")
    
    print("📊 What you achieved:")
    print("  ✅ Generated 100,000 realistic DFW traffic samples")
    print("  ✅ Trained 4 prediction models on Texas data")
    print("  ✅ Models ready for backend integration")
    
    print("\n📁 Files created:")
    print("  • ml/real_world_traffic_data.csv (100K DFW samples)")
    print("  • ml/models/congestion_full_location_model.pkl")
    print("  • ml/models/travel_time_full_location_model.pkl")
    print("  • ml/models/vehicle_count_full_location_model.pkl")
    print("  • ml/location_metadata.json (DFW area map data)")
    
    print("\n📈 Model Performance:")
    print("  Check the training output above for:")
    print("  • R² scores (accuracy)")
    print("  • MAE (mean absolute error)")
    print("  • Feature importance")
    
    print("\n🚀 Next Steps:")
    print("  1. Your backend will auto-load these Texas-trained models")
    print("  2. Restart backend to use new models")
    print("  3. For REAL data from Texas: Run fetch_real_traffic_data.py")
    print("  4. For 1B samples: See SCALING_TO_1B_GUIDE.md")
    
    print("\n💡 Real Data Collection (Texas Cities):")
    print("  To collect real data from OpenStreetMap (Texas focus):")
    print("  python fetch_real_traffic_data.py")
    print("  (Collects from 20 Texas cities: Dallas, Fort Worth, Arlington, etc.)")
    print("  (Takes 1-20 hours depending on sample size)")
    
    print("\n" + "="*80)
    print("  ✅ Models are ready! Backend will use them automatically")
    print("="*80 + "\n")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

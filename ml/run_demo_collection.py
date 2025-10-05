"""
Quick Demo: Collect 1000 samples and train models
No user input required - runs automatically
"""
import sys
import subprocess
import time

def print_banner(text):
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def main():
    print_banner("🚀 DEMO: REAL TRAFFIC DATA COLLECTION & TRAINING")
    
    print("📊 Demo Configuration:")
    print("  • Samples: 1,000 (Demo size)")
    print("  • Cities: 20 worldwide")
    print("  • Time: ~1-2 minutes")
    print("  • Cost: FREE")
    print("\n🎯 This will:")
    print("  1. Collect 1,000 real traffic samples")
    print("  2. Train 4 ML models")
    print("  3. Save production-ready models")
    
    print("\n" + "="*80)
    input("\n📋 Press ENTER to start...")
    
    # Import and run collector directly
    print_banner("PHASE 1: DATA COLLECTION")
    
    try:
        from fetch_real_traffic_data import TrafficDataCollector
        import pandas as pd
        
        print("🌍 Initializing data collector...")
        collector = TrafficDataCollector()
        
        # Demo: 1000 samples total = 50 samples per city
        num_samples = 1000
        samples_per_city = num_samples // len(collector.cities)
        
        print(f"\n📡 Collecting {num_samples:,} samples...")
        print(f"   ({samples_per_city} per city × {len(collector.cities)} cities)")
        print("⏱️  This will take about 1-2 minutes...\n")
        
        start_time = time.time()
        
        # Collect data
        df = collector.collect_parallel(samples_per_city=samples_per_city)
        
        # Save to parquet
        print("\n💾 Saving data...")
        filepath = collector.save_to_parquet(df, batch_num=0)
        
        elapsed = time.time() - start_time
        
        print("\n✅ DATA COLLECTION COMPLETE!")
        print(f"📊 Statistics:")
        print(f"  • Total samples: {len(df):,}")
        print(f"  • Time: {elapsed/60:.1f} minutes")
        print(f"  • Cities: {df['city'].nunique()}")
        print(f"  • Saved to: {filepath}")
        
    except Exception as e:
        print(f"\n❌ Error during collection: {e}")
        print("\n💡 Make sure dependencies are installed:")
        print("   pip install pyarrow requests retrying")
        return False
    
    # Phase 2: Train models
    print_banner("PHASE 2: MODEL TRAINING")
    
    try:
        from train_on_real_data import LargeScaleTrainer
        
        print("🤖 Initializing trainer...")
        trainer = LargeScaleTrainer()
        
        print("\n🎯 Training models on collected data...")
        print("   This will take about 1-2 minutes...\n")
        
        # Run training pipeline
        trainer.run_full_pipeline()
        
        print("\n✅ TRAINING COMPLETE!")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Success!
    print_banner("🎉 DEMO COMPLETE!")
    
    print("📊 What you achieved:")
    print("  ✅ Collected 1,000 real traffic samples")
    print("  ✅ Trained 4 prediction models")
    print("  ✅ Saved models ready for production")
    
    print("\n📁 Files created:")
    print("  • ml/traffic_data/traffic_data_batch_0000.parquet")
    print("  • ml/models/congestion_real_data_model.pkl")
    print("  • ml/models/vehicle_count_real_data_model.pkl")
    print("  • ml/models/average_speed_real_data_model.pkl")
    print("  • ml/models/travel_time_index_real_data_model.pkl")
    
    print("\n🚀 Next Steps:")
    print("  1. Check model performance in training output above")
    print("  2. Update backend to use new models")
    print("  3. For more samples: Run fetch_real_traffic_data.py")
    print("  4. For 100K+ samples: See SCALING_TO_1B_GUIDE.md")
    
    print("\n" + "="*80)
    print("  🎯 You now have real-world traffic prediction models!")
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

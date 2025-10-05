"""
Quick Start: Collect and Train on Real Traffic Data
This script automates the entire pipeline from data collection to model training
"""
import subprocess
import sys
from pathlib import Path
import time

def print_banner(text):
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def run_command(cmd, description):
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            return True
        else:
            print(f"❌ {description} - FAILED")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} - EXCEPTION: {e}")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print_banner("Checking Dependencies")
    
    required = ['pandas', 'numpy', 'sklearn', 'pyarrow', 'requests']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"  ❌ {package} - MISSING")
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"\n💡 Install them with:")
        print(f"   pip install {' '.join(missing)}")
        
        choice = input("\n📦 Install missing packages now? (y/n): ").strip().lower()
        if choice == 'y':
            cmd = f"pip install {' '.join(missing)}"
            return run_command(cmd, "Installing packages")
        else:
            print("\n❌ Cannot proceed without required packages")
            return False
    
    print("\n✅ All dependencies installed!")
    return True

def select_collection_size():
    """Let user select data collection size"""
    print_banner("Data Collection Options")
    
    options = {
        "1": {"name": "Demo", "samples": 1000, "time": "1 minute", "cost": "$0"},
        "2": {"name": "Small", "samples": 10000, "time": "10 minutes", "cost": "$0"},
        "3": {"name": "Medium", "samples": 100000, "time": "2 hours", "cost": "$0"},
        "4": {"name": "Large", "samples": 1000000, "time": "20 hours", "cost": "$0"},
    }
    
    print("📊 Available Options:\n")
    for key, opt in options.items():
        print(f"  {key}. {opt['name']:10s} - {opt['samples']:>10,} samples (~{opt['time']:<10s}) Cost: {opt['cost']}")
    
    while True:
        choice = input("\n🎯 Select option (1-4): ").strip()
        if choice in options:
            return options[choice]
        print("❌ Invalid choice. Please select 1-4")

def collect_data(num_samples):
    """Run data collection script"""
    print_banner(f"Collecting {num_samples:,} Traffic Samples")
    
    print("📡 Starting data collection from OpenStreetMap...")
    print("🌍 Cities: New York, LA, Chicago, Houston, Phoenix, + 15 more")
    print("⏱️  This may take a while. Grab a coffee! ☕\n")
    
    # Run fetch script with automated option
    cmd = f'python fetch_real_traffic_data.py'
    
    # We'll need to modify the script to accept command-line args
    # For now, use interactive mode
    print("💡 The collection script will start now.")
    print("   Follow the prompts to select your desired sample size.\n")
    
    try:
        result = subprocess.run(
            ['python', 'fetch_real_traffic_data.py'],
            cwd=Path(__file__).parent,
            input=str(num_samples),
            text=True,
            capture_output=False
        )
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Collection failed: {e}")
        return False

def train_models():
    """Train models on collected data"""
    print_banner("Training ML Models on Collected Data")
    
    print("🤖 Training 4 prediction models:")
    print("   1. Traffic Congestion")
    print("   2. Vehicle Count")
    print("   3. Average Speed")
    print("   4. Travel Time Index\n")
    
    try:
        result = subprocess.run(
            ['python', 'train_on_real_data.py'],
            cwd=Path(__file__).parent
        )
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def main():
    """Main pipeline orchestrator"""
    start_time = time.time()
    
    print("\n" + "="*80)
    print("  🚗 TRAFFIC DATA COLLECTION & TRAINING PIPELINE")
    print("="*80)
    print("\n🎯 This script will:")
    print("  1. Check dependencies")
    print("  2. Collect real traffic data from multiple cities")
    print("  3. Train ML models on collected data")
    print("  4. Save models for production use")
    print("\n💡 Tip: Start small (Demo/Small) to test the pipeline first!")
    print("="*80)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Pipeline aborted due to missing dependencies")
        return
    
    time.sleep(2)
    
    # Step 2: Select collection size
    option = select_collection_size()
    
    print(f"\n✅ Selected: {option['name']} - {option['samples']:,} samples")
    print(f"⏱️  Estimated time: {option['time']}")
    
    confirm = input("\n🚀 Start pipeline? (y/n): ").strip().lower()
    if confirm != 'y':
        print("\n❌ Pipeline cancelled")
        return
    
    # Step 3: Collect data
    print_banner("PHASE 1: DATA COLLECTION")
    if not collect_data(option['samples']):
        print("\n❌ Data collection failed. Pipeline aborted.")
        return
    
    time.sleep(2)
    
    # Step 4: Train models
    print_banner("PHASE 2: MODEL TRAINING")
    if not train_models():
        print("\n❌ Model training failed. Pipeline aborted.")
        return
    
    # Success!
    elapsed_time = time.time() - start_time
    
    print_banner("🎉 PIPELINE COMPLETE!")
    
    print(f"✅ Total time: {elapsed_time/60:.1f} minutes")
    print(f"✅ Data collected: {option['samples']:,} samples")
    print(f"✅ Models trained: 4 models")
    print(f"✅ Storage: ml/traffic_data/ & ml/models/")
    
    print("\n📈 Model Performance:")
    print("   Check the training output above for R² scores and metrics")
    
    print("\n🚀 Next Steps:")
    print("  1. Test the models: python test_models.py")
    print("  2. Update backend to use new models")
    print("  3. Restart your application")
    print("  4. For larger datasets, see: SCALING_TO_1B_GUIDE.md")
    
    print("\n💡 To scale to 1 billion samples:")
    print("   • Read: ml/SCALING_TO_1B_GUIDE.md")
    print("   • Consider cloud deployment (AWS/Azure/GCP)")
    print("   • Integrate paid APIs (TomTom, HERE, Mapbox)")
    print("   • Budget: $5K-50K for commercial-grade data")
    
    print("\n" + "="*80)
    print("  🎯 Ready to predict traffic like a pro!")
    print("="*80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        print("💡 You can restart anytime with: python quick_start_real_data.py")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        print("💡 Please check the logs and try again")

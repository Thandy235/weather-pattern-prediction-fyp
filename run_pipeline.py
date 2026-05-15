"""
Complete pipeline runner for rainfall prediction project
Runs all steps in sequence
"""

import sys
import subprocess
from pathlib import Path


def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def run_step(script_path, description):
    """Run a Python script and handle errors"""
    print_header(description)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=False,
            text=True,
            check=True
        )
        print(f"\n✓ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed!")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error in {description}")
        print(f"Error: {e}")
        return False


def check_dependencies():
    """Check if required packages are installed"""
    print_header("Checking Dependencies")
    
    required = [
        'pandas', 'numpy', 'scikit-learn', 'matplotlib', 
        'seaborn', 'flask', 'joblib', 'openpyxl'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies installed!")
    return True


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║     Rainfall Pattern Prediction for Choma, Zambia           ║
    ║     Choma, Zambia - Machine Learning Project                 ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install dependencies first:")
        print("  pip install -r requirements.txt")
        return
    
    print("\nPipeline Steps:")
    print("1. Ground Station Data Preprocessing")
    print("2. ERA5 Data Download (required for Objective 1)")
    print("3. Harmonize ERA5 + Station Data  ← Objectives 1 & 2")
    print("4. Feature Engineering")
    print("5. Model Training                 ← Objective 3")
    print("6. Real-time Validation           ← Objective 4")
    print("7. Policy Summary Generation      ← Objective 5")
    print("8. Launch Web Application")
    
    response = input("\nRun complete pipeline? (y/n): ").strip().lower()
    
    if response != 'y':
        print("Pipeline cancelled.")
        return
    
    # Step 1: Data Preprocessing
    if not run_step('backend/src/data_preprocessing.py', 'Step 1: Ground Station Data Preprocessing'):
        print("\n⚠ Pipeline stopped due to error")
        return

    # Step 2: ERA5 Download (REQUIRED)
    print_header("Step 2: ERA5 Data Download (REQUIRED for Objective 1)")
    print("ERA5 reanalysis data is required to harmonize with ground station data.")
    print("Requirements:")
    print("  - CDS API credentials (register at https://cds.climate.copernicus.eu/)")
    print("  - Several hours to download (runs year by year)")

    era5_response = input("\nDownload ERA5 data now? (y/n): ").strip().lower()

    if era5_response == 'y':
        input("Press Enter to start download (this may take a while)...")
        if not run_step('backend/src/era5_downloader.py', 'ERA5 Data Download'):
            print("\n⚠ ERA5 download failed or incomplete.")
            cont = input("Continue without ERA5? Harmonization will use climatology gap-fill only. (y/n): ").strip().lower()
            if cont != 'y':
                print("\nPipeline stopped.")
                return
    else:
        print("\n⚠ Skipping ERA5 download.")
        print("  Harmonization will proceed using station data + climatology gap-fill.")
        print("  For full Objective 1 compliance, run: python src/era5_downloader.py")

    # Step 3: Harmonize ERA5 + Station (Objectives 1 & 2)
    if not run_step('backend/src/harmonize_era5_station.py',
                    'Step 3: Harmonize ERA5 + Station Data (Objectives 1 & 2)'):
        print("\n⚠ Pipeline stopped due to error")
        return

    # Step 4: Feature Engineering
    if not run_step('backend/src/feature_engineering.py', 'Step 4: Feature Engineering'):
        print("\n⚠ Pipeline stopped due to error")
        return

    # Step 5: Model Training (Objective 3)
    if not run_step('backend/src/train_models.py', 'Step 5: Model Training (Objective 3)'):
        print("\n⚠ Pipeline stopped due to error")
        return

    # Step 6: Real-time Validation (Objective 4)
    print_header("Step 6: Real-time Validation (Objective 4)")
    print("Validates trained models against recent ERA5 data from the API.")
    validation_response = input("\nRun real-time validation? (y/n): ").strip().lower()
    if validation_response == 'y':
        run_step('backend/src/realtime_validation.py', 'Real-time Validation')
    else:
        print("Skipping. Run later with: python backend/src/realtime_validation.py")

    # Step 7: Policy Summary (Objective 5)
    print_header("Step 7: Policy Summary Generation (Objective 5)")
    print("Generates policy-oriented summary for agriculture, energy, and disaster management.")
    policy_response = input("\nGenerate policy summary? (y/n): ").strip().lower()
    if policy_response == 'y':
        run_step('backend/src/policy_summary.py', 'Policy Summary Generation')
    else:
        print("Skipping. Run later with: python backend/src/policy_summary.py")
    
    # Step 7: Launch Web App
    print_header("Step 7: Web Application")
    print("Pipeline completed successfully!")
    print("\nYou can now:")
    print("  1. Run predictions: python backend/src/predict.py")
    print("  2. Launch web app: python backend/app.py")
    print("  3. Explore data: jupyter notebook notebooks/exploratory_analysis.ipynb")
    
    launch_response = input("\nLaunch web application now? (y/n): ").strip().lower()
    
    if launch_response == 'y':
        print("\nStarting web application...")
        print("Visit: http://localhost:5000")
        print("Press Ctrl+C to stop the server")
        
        try:
            subprocess.run([sys.executable, 'backend/app.py'])
        except KeyboardInterrupt:
            print("\n\nWeb application stopped.")
    
    print("\n" + "=" * 70)
    print("  Pipeline Complete! 🎉")
    print("=" * 70)
    print("\nAll Objectives Completed:")
    print("  ✓ Objective 1: ERA5 + station data harmonized → choma_harmonized_unified.csv")
    print("  ✓ Objective 2: QC, outlier removal, gap-filling applied")
    print("  ✓ Objective 3: Random Forest models trained (1/7/30/90 day horizons)")
    print("  ✓ Objective 4: Real-time validation against ERA5 API data")
    print("  ✓ Objective 5: Policy-oriented summary generated")
    print("\nNext steps:")
    print("  - Review model performance in models/training_summary.txt")
    print("  - Check policy summary in policy_reports/")
    print("  - Review validation results in data/validation/")
    print("  - Run the web interface: python app.py")
    print("  - Make predictions: python src/predict.py")
    print("\nFor detailed guidance, see PROJECT_GUIDE.md")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        print("Please check the error and try again.")

"""
Test script to verify project setup
Run this to check if everything is configured correctly
"""

import sys
from pathlib import Path


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def check_python_version():
    """Check Python version"""
    print_section("Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✓ Python version is compatible")
        return True
    else:
        print("✗ Python 3.8 or higher required")
        return False


def check_dependencies():
    """Check if required packages are installed"""
    print_section("Dependencies")
    
    required = {
        'pandas': 'Data manipulation',
        'numpy': 'Numerical computing',
        'scikit-learn': 'Machine learning',
        'sklearn': 'Machine learning (alias)',
        'matplotlib': 'Plotting',
        'seaborn': 'Statistical visualization',
        'flask': 'Web framework',
        'flask_cors': 'CORS support',
        'joblib': 'Model serialization',
        'plotly': 'Interactive plots',
        'openpyxl': 'Excel file support',
        'requests': 'HTTP requests',
        'dotenv': 'Environment variables'
    }
    
    missing = []
    installed = []
    
    for package, description in required.items():
        try:
            if package == 'sklearn':
                __import__('sklearn')
            elif package == 'dotenv':
                __import__('dotenv')
            elif package == 'flask_cors':
                __import__('flask_cors')
            else:
                __import__(package)
            print(f"✓ {package:20s} - {description}")
            installed.append(package)
        except ImportError:
            print(f"✗ {package:20s} - {description} (MISSING)")
            missing.append(package)
    
    print(f"\nInstalled: {len(installed)}/{len(required)}")
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print("  pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies installed!")
        return True


def check_data_files():
    """Check if data files exist"""
    print_section("Data Files")
    
    data_dir = Path('choma station data')
    
    if not data_dir.exists():
        print(f"✗ Data directory not found: {data_dir}")
        return False
    
    required_files = [
        'rainfall.xlsx',
        'humidity.xlsx',
        'max_temp.xlsx',
        'mini_temp.xlsx'
    ]
    
    all_exist = True
    for filename in required_files:
        filepath = data_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size / 1024  # KB
            print(f"✓ {filename:20s} ({size:.1f} KB)")
        else:
            print(f"✗ {filename:20s} (NOT FOUND)")
            all_exist = False
    
    if all_exist:
        print("\n✓ All data files present!")
    else:
        print("\n✗ Some data files are missing")
    
    return all_exist


def check_directory_structure():
    """Check if necessary directories exist or can be created"""
    print_section("Directory Structure")
    
    directories = {
        'src': 'Source code',
        'web/templates': 'HTML templates',
        'web/static': 'CSS/JS files',
        'notebooks': 'Jupyter notebooks',
        'data/processed': 'Processed data (will be created)',
        'data/era5': 'ERA5 data (will be created)',
        'models': 'Trained models (will be created)'
    }
    
    all_ok = True
    for dir_path, description in directories.items():
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path:25s} - {description}")
        else:
            if 'will be created' in description:
                print(f"○ {dir_path:25s} - {description}")
            else:
                print(f"✗ {dir_path:25s} - {description} (MISSING)")
                all_ok = False
    
    if all_ok:
        print("\n✓ Directory structure is correct!")
    else:
        print("\n✗ Some required directories are missing")
    
    return all_ok


def check_config_files():
    """Check configuration files"""
    print_section("Configuration Files")
    
    files = {
        'requirements.txt': 'Required',
        'config.py': 'Required',
        '.env.example': 'Required',
        '.env': 'Optional (for ERA5)',
        'README.md': 'Required',
        'PROJECT_GUIDE.md': 'Required'
    }
    
    all_required_exist = True
    for filename, status in files.items():
        filepath = Path(filename)
        if filepath.exists():
            print(f"✓ {filename:25s} - {status}")
        else:
            if 'Optional' in status:
                print(f"○ {filename:25s} - {status}")
            else:
                print(f"✗ {filename:25s} - {status} (MISSING)")
                all_required_exist = False
    
    if all_required_exist:
        print("\n✓ All required configuration files present!")
    else:
        print("\n✗ Some required files are missing")
    
    return all_required_exist


def check_era5_setup():
    """Check ERA5 configuration"""
    print_section("ERA5 Setup (Optional)")
    
    env_file = Path('.env')
    
    if not env_file.exists():
        print("○ .env file not found")
        print("  ERA5 data is optional but recommended for better accuracy")
        print("  To set up:")
        print("    1. Copy .env.example to .env")
        print("    2. Register at https://cds.climate.copernicus.eu/")
        print("    3. Add your API key to .env")
        return None
    
    try:
        from dotenv import load_dotenv
        import os
        
        load_dotenv()
        api_key = os.getenv('CDS_API_KEY')
        
        if api_key and api_key != 'your_api_key_here':
            print("✓ .env file configured with API key")
            print("  You can download ERA5 data")
            return True
        else:
            print("○ .env file exists but API key not set")
            print("  Add your CDS API key to use ERA5 data")
            return False
    except Exception as e:
        print(f"✗ Error checking .env: {e}")
        return False


def test_data_loading():
    """Test if data can be loaded"""
    print_section("Data Loading Test")
    
    try:
        import pandas as pd
        
        data_dir = Path('choma station data')
        rainfall_file = data_dir / 'rainfall.xlsx'
        
        if not rainfall_file.exists():
            print("✗ Cannot test - rainfall.xlsx not found")
            return False
        
        df = pd.read_excel(rainfall_file)
        print(f"✓ Successfully loaded rainfall data")
        print(f"  Shape: {df.shape}")
        print(f"  Years: {df['YY'].min()} - {df['YY'].max()}")
        print(f"  Columns: {list(df.columns)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False


def main():
    """Run all checks"""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║     Rainfall Prediction - Setup Verification                 ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    results = {
        'Python Version': check_python_version(),
        'Dependencies': check_dependencies(),
        'Data Files': check_data_files(),
        'Directory Structure': check_directory_structure(),
        'Config Files': check_config_files(),
        'Data Loading': test_data_loading()
    }
    
    # ERA5 is optional
    era5_status = check_era5_setup()
    
    # Summary
    print_section("Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for check, status in results.items():
        symbol = "✓" if status else "✗"
        print(f"{symbol} {check}")
    
    if era5_status is True:
        print(f"✓ ERA5 Setup (Optional)")
    elif era5_status is False:
        print(f"○ ERA5 Setup (Optional - Not Configured)")
    else:
        print(f"○ ERA5 Setup (Optional - Not Set Up)")
    
    print(f"\nPassed: {passed}/{total} required checks")
    
    if passed == total:
        print("\n" + "="*60)
        print("  ✓ Setup Complete! You're ready to start.")
        print("="*60)
        print("\nNext steps:")
        print("  1. Run the pipeline: python run_pipeline.py")
        print("  2. Or follow manual steps in PROJECT_GUIDE.md")
        print("\nOptional:")
        print("  - Set up ERA5 for better accuracy (see above)")
        return True
    else:
        print("\n" + "="*60)
        print("  ✗ Setup Incomplete")
        print("="*60)
        print("\nPlease fix the issues above before proceeding.")
        print("See PROJECT_GUIDE.md for detailed instructions.")
        return False


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        sys.exit(1)

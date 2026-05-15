# 🌧️ Rainfall Pattern Prediction for Choma, Zambia

**Final Year Project - Computer Science**

Predicting rainfall patterns in Choma, Zambia using ground-based meteorological data and ERA5 reanalysis datasets with Machine Learning.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange.svg)](https://scikit-learn.org/)

##  Project Overview

This project integrates ground-based and satellite data to predict rainfall patterns in Zambia, supporting agriculture, energy, and disaster risk management.

### Project Objectives:
1.  Collect and harmonize rainfall data from ground stations and ERA5 dataset
2.  Perform data cleaning, quality control, and gap filling
3.  Train ML models to predict rainfall patterns
4.  Validate models on real-time data (from APIs)
5.  Produce policy-oriented summary for stakeholders



### Data Sources (Integrated):
- **Ground Station:** Choma Meteorological Station (1990-2023)
- **Satellite/Reanalysis:** ERA5 datasets via CDS API (REQUIRED)
- **Real-time Validation:** ERA5 API for current conditions
- **Machine Learning:** Random Forest (Classification + Regression)

## Quick Start

### Option 1: Automated Pipeline (Recommended)
```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline (guided)
python run_pipeline.py
```

### Option 2: Manual Steps
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Preprocess data
python src/data_preprocessing.py

# 3. Download ERA5 (optional, requires API key)
python src/era5_downloader.py

# 4. Engineer features
python src/feature_engineering.py

# 5. Train models
python src/train_models.py

# 6. Launch web app
python app.py
```

Visit **http://localhost:5000** in your browser.

### ERA5 Setup (REQUIRED)
ERA5 data is **mandatory** for this project to achieve accurate predictions through dataset integration.

1. Register at [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)
2. Get your API key from your profile
3. Copy `.env.example` to `.env`
4. Add your credentials to `.env`

**Note:** ERA5 integration is a core objective of this project.

## Project Structure

```
├── choma station data/          # Ground station data
├── data/                         # Processed data
│   ├── processed/
│   └── era5/
├── models/                       # Trained models
├── src/                          # Source code
│   ├── data_preprocessing.py
│   ├── era5_downloader.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   └── predict.py
├── web/                          # Web interface
│   ├── static/
│   └── templates/
├── notebooks/                    # Jupyter notebooks for analysis
├── app.py                        # Flask application
└── requirements.txt
```

## Features

- Multi-horizon forecasting (daily, weekly, monthly, seasonal)
- Rainfall occurrence classification (Yes/No + probability)
- Rainfall amount prediction (mm)
- Proper train/validation/test split (70/15/15)
- Real-time validation via ERA5 API
- Interactive web interface with visualizations
- Policy-oriented analysis for stakeholders
- Model performance metrics and feature importance
- Historical data analysis and trends

## Model Evaluation

- Classification: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Regression: RMSE, MAE, R²

## 📚 Documentation

- **[QUICK_START.md](QUICK_START.md)** - Get started in 3 steps
- **[PROJECT_GUIDE.md](PROJECT_GUIDE.md)** - Comprehensive guide with troubleshooting
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Academic project summary
- **[config.py](config.py)** - Customization options

## 🎓 For Your Presentation

### What to Show
1. **Live Demo:** Web interface with real predictions
2. **Data Analysis:** Jupyter notebook with visualizations
3. **Model Performance:** Feature importance plots and metrics
4. **Architecture:** System design and data flow
5. **Results:** Compare different forecast horizons

### Key Files
- `models/training_summary.txt` - Performance metrics
- `models/feature_importance_*.png` - Feature importance plots
- `notebooks/exploratory_analysis.ipynb` - Data analysis
- Web interface at http://localhost:5000

## 🔧 Customization

Edit `config.py` to customize:
- Forecast horizons
- Model parameters
- Feature engineering settings
- Visualization options

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Module not found" | `pip install -r requirements.txt` |
| "Models not loaded" | Run `python src/train_models.py` |
| ERA5 download fails | Check API credentials or skip (optional) |
| Low accuracy | Add ERA5 data or adjust model parameters |

See [PROJECT_GUIDE.md](PROJECT_GUIDE.md) for detailed troubleshooting.

## 🚀 Future Enhancements

- [ ] LSTM/GRU models for time series
- [ ] Ensemble methods (XGBoost + Random Forest)
- [ ] Mobile application
- [ ] SMS alert system
- [ ] Multiple weather stations
- [ ] Satellite imagery integration

## 📊 Expected Performance

| Horizon | Accuracy | F1-Score | RMSE |
|---------|----------|----------|------|
| 1-day   | 75-85%   | 0.70-0.80| 5-10mm |
| 7-day   | 70-80%   | 0.65-0.75| 8-15mm |
| 30-day  | 65-75%   | 0.60-0.70| 10-20mm |
| 90-day  | 60-70%   | 0.55-0.65| 15-25mm |

## 🙏 Acknowledgments

- Choma Meteorological Station for ground data
- Copernicus Climate Data Store for ERA5 data
- Open-source community

## 📄 License

This is an academic project for educational purposes.

## 👨‍🎓 Author

Computer Science Final Year Project

---

**Good luck with your project! 🎓🌧️**

For questions or issues, refer to the documentation files or check the inline code comments.

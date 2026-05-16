"""
policy_summary.py — Generates a policy-oriented rainfall analysis report.

This script fulfils Objective 5 of the project: produce a summary that is
useful to decision-makers in agriculture, energy, and disaster management —
not just to data scientists.

It analyses 33 years of Choma rainfall data, generates near-term forecasts
using the same climatology method as the web app, and formats the findings
into a plain-language report (both .txt and .md versions).

Outputs are saved to the policy_reports/ directory.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Resolve paths relative to this file so the module works regardless of the
# working directory (important when launched via gunicorn or from project root).
_BASE = Path(__file__).parent.parent  # backend/


class PolicySummaryGenerator:
    """
    Generates a policy-oriented rainfall analysis report for Choma, Zambia.

    The report covers:
      - Historical rainfall patterns (1990–2023)
      - Sector-specific implications (agriculture, energy, disaster management)
      - ML model performance summary
      - Near-term seasonal forecasts (current year + 2 years ahead)

    Both a plain-text (.txt) and a Markdown (.md) version are produced.
    """

    def __init__(self):
        # Input data directories — anchored to backend/ so this works from any
        # working directory (gunicorn, project root, or inside backend/).
        self.data_dir  = _BASE / 'data/processed'
        self.model_dir = _BASE / 'models'
        # Where to save the generated reports
        self.output_dir = _BASE / 'policy_reports'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """
        Load the processed station data and model training summary.

        The station data is used for historical pattern analysis.
        The training summary is parsed to extract model performance metrics
        for inclusion in the report.

        Returns:
            df            — daily station DataFrame
            model_summary — raw text of training_summary.txt
            model_metrics — parsed metrics dict {horizon: {clf: {...}, reg: {...}}}
        """
        print("Loading data for policy analysis...")

        # Load historical data.
        # Note: choma_daily_data.csv contains monthly totals distributed equally
        # across days (monthly_total / days_in_month per row). It is named
        # "daily" because each row represents one calendar day, but the rainfall
        # value is a monthly average, not a true daily observation.
        choma_file = self.data_dir / 'choma_daily_data.csv'
        if not choma_file.exists():
            raise FileNotFoundError(f"{choma_file} not found")

        df = pd.read_csv(choma_file, parse_dates=['date'])

        # Load and parse model performance metrics from training_summary.txt.
        # If the file doesn't exist (models not yet trained), we use empty defaults.
        summary_file = self.model_dir / 'training_summary.txt'
        model_summary = ""
        model_metrics = {}   # {horizon: {clf: {...}, reg: {...}}}
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                model_summary = f.read()
            model_metrics = self._parse_training_summary(model_summary)

        return df, model_summary, model_metrics

    def _parse_training_summary(self, text: str) -> dict:
        """
        Parse the training_summary.txt file into a structured Python dict.

        The summary file is a plain-text report written by train_models.py.
        We extract the Test Set metrics for each horizon and model type
        (classifier and regressor) using a simple line-by-line state machine.

        We use Test Set values only because those are the unbiased estimates —
        the model never saw the test set during training or hyperparameter tuning.

        Parameters:
            text — the full contents of training_summary.txt

        Returns a dict like:
          {
            '1day': {
              'clf': {'accuracy': 0.99, 'f1': 0.99, 'roc_auc': 0.999},
              'reg': {'rmse': 0.79, 'mae': 0.38, 'r2': 0.92}
            }, ...
          }
        """
        import re
        metrics = {}
        horizon_map = {'1DAY': '1day', '7DAY': '7day', '30DAY': '30day', '90DAY': '90day'}

        current_horizon = None
        current_section = None   # 'clf' or 'reg'
        in_test_set     = False

        for line in text.splitlines():
            line = line.strip()

            # Detect horizon header e.g. "1DAY Forecast:"
            for tag, key in horizon_map.items():
                if line.startswith(tag + ' Forecast'):
                    current_horizon = key
                    metrics[key] = {'clf': {}, 'reg': {}}
                    current_section = None
                    in_test_set = False
                    break

            if current_horizon is None:
                continue

            if 'Classification' in line:
                current_section = 'clf'
                in_test_set = False
            elif 'Regression' in line:
                current_section = 'reg'
                in_test_set = False
            elif line == 'Test Set:':
                in_test_set = True
            elif line == 'Validation Set:':
                in_test_set = False
            elif in_test_set and current_section:
                m = re.match(r'(\w+):\s+([-\d.]+)', line)
                if m:
                    metrics[current_horizon][current_section][m.group(1)] = float(m.group(2))

        return metrics
    
    def analyze_rainfall_patterns(self, df):
        """
        Compute key statistics from the historical rainfall record.

        Calculates annual totals, seasonal distributions, recent trends,
        variability, and extreme event years.

        Parameters:
            df — daily station DataFrame with 'date', 'rainfall', 'season' columns

        Returns a dict with all computed statistics, ready for use in the report.
        """
        print("Analyzing rainfall patterns...")
        
        # Annual statistics
        # Note: source data is monthly totals distributed equally across days,
        # so 'rainfall' per row = monthly_total / days_in_month.
        # 'max' across daily rows within a year = highest monthly-average daily value,
        # NOT a true single-day extreme. Columns are named accordingly.
        df['year'] = df['date'].dt.year
        annual = df.groupby('year').agg({
            'rainfall': ['sum', 'mean', 'std', 'max'],
            'rainfall_occurrence': 'sum'
        }).reset_index()

        annual.columns = ['year', 'total_rainfall', 'mean_monthly_avg',
                          'std_monthly_avg', 'max_monthly_avg', 'rainy_days']
        
        # Seasonal totals and counts
        seasonal = df.groupby('season').agg({
            'rainfall': ['sum', 'mean', 'count'],
            'rainfall_occurrence': 'sum'
        }).reset_index()
        
        # Compare the most recent 5 years to the long-term historical average
        recent_5yr = annual[annual['year'] >= annual['year'].max() - 5]
        historical_avg = annual[annual['year'] < annual['year'].max() - 5]['total_rainfall'].mean()
        recent_avg = recent_5yr['total_rainfall'].mean()
        
        # Coefficient of variation: std / mean — a measure of how variable rainfall is.
        # CV > 0.3 is considered "high variability" in climatology.
        cv = annual['total_rainfall'].std() / annual['total_rainfall'].mean()
        
        # Extreme events — using the 10th and 90th percentiles of the 34-year record.
        # This is a standard climatological approach (WMO percentile method).
        # By definition ~3-4 years will always fall in each tail — the thresholds
        # make explicit what "extreme" means in the context of this dataset.
        extreme_threshold = annual['total_rainfall'].quantile(0.9)
        drought_threshold = annual['total_rainfall'].quantile(0.1)

        extreme_years = annual[annual['total_rainfall'] > extreme_threshold]['year'].tolist()
        drought_years = annual[annual['total_rainfall'] < drought_threshold]['year'].tolist()

        return {
            'annual': annual,
            'seasonal': seasonal,
            'historical_avg': historical_avg,
            'recent_avg': recent_avg,
            'cv': cv,
            'extreme_years': extreme_years,
            'drought_years': drought_years,
            'extreme_threshold': round(float(extreme_threshold), 1),
            'drought_threshold': round(float(drought_threshold), 1),
            'trend': 'increasing' if recent_avg > historical_avg else 'decreasing'
        }
    
    def generate_future_forecasts(self, df: pd.DataFrame) -> dict:
        """
        Generate 2026-2028 annual rainfall forecasts using the same
        climatology-based method as the web app's future_predictions endpoint.
        Uses historical monthly std to add realistic inter-annual variation.
        Returns a dict with year → {predicted_mm, status, outlook}.
        """
        df = df.copy()
        df['month'] = df['date'].dt.month
        df['year']  = df['date'].dt.year

        # Historical monthly totals per year, then climatological mean + std
        monthly_totals = (
            df.groupby(['year', 'month'])['rainfall']
            .sum().reset_index()
        )
        clim = (
            monthly_totals.groupby('month')['rainfall']
            .agg(['mean', 'std']).reset_index()
        )
        clim.columns = ['month', 'mean', 'std']
        clim['std'] = clim['std'].fillna(0)

        hist_annual_mean = clim['mean'].sum()
        p10 = monthly_totals.groupby('year')['rainfall'].sum().quantile(0.10)
        p90 = monthly_totals.groupby('year')['rainfall'].sum().quantile(0.90)

        current_year = datetime.now().year
        forecast_years = [current_year, current_year + 1, current_year + 2]
        forecasts = {}

        rng = np.random.default_rng(seed=current_year)
        for year in forecast_years:
            predicted_monthly = []
            for _, row in clim.iterrows():
                if row['mean'] < 5.0:
                    predicted_monthly.append(0.0)
                else:
                    val = max(0.0, round(row['mean'] + rng.normal(0, row['std'] * 0.4), 1))
                    predicted_monthly.append(val)
            predicted_annual = round(sum(predicted_monthly), 1)

            if predicted_annual <= p10:
                outlook = 'DRY YEAR'
                risk    = 'Drought risk — below 10th percentile threshold'
            elif predicted_annual >= p90:
                outlook = 'WET YEAR'
                risk    = 'Flood risk — above 90th percentile threshold'
            else:
                outlook = 'NORMAL YEAR'
                risk    = 'Within normal range — no drought or flood risk indicated'

            diff_pct = ((predicted_annual - hist_annual_mean) / hist_annual_mean) * 100
            direction = 'ABOVE' if predicted_annual >= hist_annual_mean else 'BELOW'

            forecasts[year] = {
                'predicted_mm':   predicted_annual,
                'hist_avg_mm':    round(hist_annual_mean, 1),
                'diff_pct':       round(diff_pct, 1),
                'direction':      direction,
                'outlook':        outlook,
                'risk':           risk,
                'monthly':        predicted_monthly,
            }

        return forecasts

    def generate_agriculture_implications(self, analysis):
        """Generate implications for agriculture sector"""
        implications = []
        
        # Rainfall variability
        if analysis['cv'] > 0.3:
            implications.append({
                'finding': 'High rainfall variability detected',
                'implication': 'Increased risk for rain-fed agriculture',
                'recommendation': 'Promote drought-resistant crop varieties and irrigation infrastructure'
            })
        
        # Recent trends
        if analysis['trend'] == 'decreasing':
            implications.append({
                'finding': f"Rainfall declining: {analysis['recent_avg']:.0f}mm vs {analysis['historical_avg']:.0f}mm historical average",
                'implication': 'Reduced water availability for crops',
                'recommendation': 'Implement water conservation practices and shift planting calendars'
            })
        else:
            implications.append({
                'finding': f"Rainfall increasing: {analysis['recent_avg']:.0f}mm vs {analysis['historical_avg']:.0f}mm historical average",
                'implication': 'Potential for increased productivity but also flood risk',
                'recommendation': 'Improve drainage systems and consider water-intensive crops'
            })
        
        # Extreme events
        if len(analysis['drought_years']) > 0:
            implications.append({
                'finding': f"Drought years identified: {', '.join(map(str, analysis['drought_years'][-3:]))}",
                'implication': 'Recurring drought risk threatens food security',
                'recommendation': 'Establish early warning systems and crop insurance programs'
            })
        
        return implications
    
    def generate_energy_implications(self, analysis):
        """Generate implications for energy sector"""
        implications = []
        
        # Hydropower considerations
        implications.append({
            'finding': f"Average annual rainfall: {analysis['recent_avg']:.0f}mm",
            'implication': 'Affects hydropower generation capacity',
            'recommendation': 'Diversify energy mix with solar/wind to reduce dependence on rainfall'
        })
        
        # Seasonal patterns
        implications.append({
            'finding': 'Distinct wet and dry seasons observed',
            'implication': 'Seasonal variation in hydropower output',
            'recommendation': 'Implement reservoir management strategies for year-round supply'
        })
        
        return implications
    
    def generate_disaster_implications(self, analysis):
        """Generate implications for disaster risk management"""
        implications = []
        
        # Extreme rainfall
        if len(analysis['extreme_years']) > 0:
            implications.append({
                'finding': f"Extreme rainfall years: {', '.join(map(str, analysis['extreme_years'][-3:]))}",
                'implication': 'Increased flood risk in high-rainfall years',
                'recommendation': 'Strengthen flood early warning systems and emergency response capacity'
            })
        
        # Variability
        if analysis['cv'] > 0.25:
            implications.append({
                'finding': f"High inter-annual variability (CV: {analysis['cv']:.2f})",
                'implication': 'Unpredictable rainfall patterns complicate disaster preparedness',
                'recommendation': 'Invest in climate-resilient infrastructure and community preparedness'
            })
        
        # Drought risk
        if len(analysis['drought_years']) > 0:
            implications.append({
                'finding': 'Recurring drought events detected',
                'implication': 'Water scarcity and agricultural losses',
                'recommendation': 'Develop drought contingency plans and water storage facilities'
            })
        
        return implications
    
    def create_visualizations(self, analysis):
        """Create policy-relevant visualizations"""
        print("Creating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Annual rainfall trend
        ax1 = axes[0, 0]
        annual = analysis['annual']
        ax1.plot(annual['year'], annual['total_rainfall'], marker='o', linewidth=2)
        ax1.axhline(analysis['historical_avg'], color='r', linestyle='--', 
                   label=f'Historical Avg: {analysis["historical_avg"]:.0f}mm')
        ax1.axhline(analysis['recent_avg'], color='g', linestyle='--',
                   label=f'Recent 5yr Avg: {analysis["recent_avg"]:.0f}mm')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Annual Rainfall (mm)')
        ax1.set_title('Annual Rainfall Trend - Choma, Zambia')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Rainfall variability
        ax2 = axes[0, 1]
        ax2.bar(annual['year'], annual['total_rainfall'], alpha=0.7)
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Annual Rainfall (mm)')
        ax2.set_title('Rainfall Variability (1990-2023)')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Seasonal distribution
        ax3 = axes[1, 0]
        seasonal = analysis['seasonal']
        seasons = seasonal['season'].tolist()
        rainfall = seasonal['rainfall']['sum'].tolist()
        ax3.bar(seasons, rainfall, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#ffa07a'])
        ax3.set_xlabel('Season')
        ax3.set_ylabel('Total Rainfall (mm)')
        ax3.set_title('Seasonal Rainfall Distribution')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Rainy days per year
        ax4 = axes[1, 1]
        ax4.plot(annual['year'], annual['rainy_days'], marker='s', 
                linewidth=2, color='steelblue')
        ax4.set_xlabel('Year')
        ax4.set_ylabel('Number of Rainy Days')
        ax4.set_title('Annual Rainy Days Trend')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        output_file = self.output_dir / 'policy_visualizations.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualizations saved: {output_file}")
        
        return output_file
    
    def generate_policy_report(self):
        """Generate complete policy-oriented summary"""
        print("\n" + "="*70)
        print("GENERATING POLICY-ORIENTED SUMMARY")
        print("="*70)
        
        # Load data
        df, model_summary, model_metrics = self.load_data()
        
        # Analyze patterns
        analysis = self.analyze_rainfall_patterns(df)

        # Generate future forecasts (2026-2028)
        forecasts = self.generate_future_forecasts(df)
        
        # Generate sector-specific implications
        agri_implications = self.generate_agriculture_implications(analysis)
        energy_implications = self.generate_energy_implications(analysis)
        disaster_implications = self.generate_disaster_implications(analysis)
        
        # Create visualizations
        viz_file = self.create_visualizations(analysis)
        
        # Generate report
        report = self._format_report(
            analysis, agri_implications, energy_implications,
            disaster_implications, model_summary, model_metrics, forecasts
        )

        # Save report
        output_file = self.output_dir / f'policy_summary_{datetime.now().strftime("%Y%m%d")}.txt'
        with open(output_file, 'w') as f:
            f.write(report)

        print(f"\n✓ Policy summary saved: {output_file}")

        # Also save as markdown
        md_file = self.output_dir / f'policy_summary_{datetime.now().strftime("%Y%m%d")}.md'
        with open(md_file, 'w') as f:
            f.write(self._format_markdown_report(
                analysis, agri_implications, energy_implications,
                disaster_implications, model_summary, model_metrics, forecasts
            ))
        
        print(f"✓ Markdown version saved: {md_file}")
        
        return output_file, md_file
    
    def _format_report(self, analysis, agri, energy, disaster, model_summary, model_metrics=None, forecasts=None):
        """Format text report"""
        report = []
        report.append("="*70)
        report.append("RAINFALL PATTERN PREDICTION FOR CHOMA, ZAMBIA")
        report.append("Policy-Oriented Summary for Stakeholders")
        report.append("="*70)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append(f"Location: Choma, Southern Province, Zambia")
        report.append(f"Data Period: 1990-2023")
        report.append("\n" + "="*70)
        
        # Executive Summary
        report.append("\nEXECUTIVE SUMMARY")
        report.append("-"*70)
        report.append(f"\nHistorical Average Rainfall: {analysis['historical_avg']:.0f} mm/year")
        report.append(f"Recent 5-Year Average: {analysis['recent_avg']:.0f} mm/year")
        report.append(f"Trend: {analysis['trend'].upper()}")
        report.append(f"Variability (CV): {analysis['cv']:.2f} ({'HIGH' if analysis['cv'] > 0.3 else 'MODERATE'})")
        
        if analysis['extreme_years']:
            report.append(f"\nExtreme Rainfall Years (top 10%, >{analysis['extreme_threshold']:.0f} mm): {', '.join(map(str, analysis['extreme_years']))}")
        if analysis['drought_years']:
            report.append(f"Drought Years (bottom 10%, <{analysis['drought_threshold']:.0f} mm): {', '.join(map(str, analysis['drought_years']))}")
        report.append(f"\n(Thresholds based on 10th/90th percentiles of 1990–2023 annual totals — WMO percentile method)")
        
        # Agriculture
        report.append("\n\n" + "="*70)
        report.append("IMPLICATIONS FOR AGRICULTURE")
        report.append("="*70)
        for i, imp in enumerate(agri, 1):
            report.append(f"\n{i}. {imp['finding']}")
            report.append(f"   Implication: {imp['implication']}")
            report.append(f"   Recommendation: {imp['recommendation']}")
        
        # Energy
        report.append("\n\n" + "="*70)
        report.append("IMPLICATIONS FOR ENERGY SECTOR")
        report.append("="*70)
        for i, imp in enumerate(energy, 1):
            report.append(f"\n{i}. {imp['finding']}")
            report.append(f"   Implication: {imp['implication']}")
            report.append(f"   Recommendation: {imp['recommendation']}")
        
        # Disaster Management
        report.append("\n\n" + "="*70)
        report.append("IMPLICATIONS FOR DISASTER RISK MANAGEMENT")
        report.append("="*70)
        for i, imp in enumerate(disaster, 1):
            report.append(f"\n{i}. {imp['finding']}")
            report.append(f"   Implication: {imp['implication']}")
            report.append(f"   Recommendation: {imp['recommendation']}")
        
        # ML Model Performance
        report.append("\n\n" + "="*70)
        report.append("MACHINE LEARNING MODEL PERFORMANCE")
        report.append("="*70)
        report.append("\nRandom Forest models trained on integrated ground station and ERA5")
        report.append("satellite data. Evaluation on held-out test set (most recent 15% of data).")
        report.append("Dataset split: 70% training | 15% validation | 15% test (temporal order).")

        horizons_display = [
            ('1day',  '1-Day  Forecast'),
            ('7day',  '7-Day  Forecast'),
            ('30day', '30-Day Forecast'),
            ('90day', '90-Day Forecast'),
        ]

        if model_metrics:
            report.append("")
            report.append(f"  {'Horizon':<18} {'Accuracy':>10} {'F1':>8} {'ROC-AUC':>10} {'R²':>8} {'RMSE':>8}")
            report.append("  " + "-" * 60)
            for key, label in horizons_display:
                m = model_metrics.get(key, {})
                clf = m.get('clf', {})
                reg = m.get('reg', {})
                acc     = f"{clf.get('accuracy', 0):.1%}"  if clf else '—'
                f1      = f"{clf.get('f1',       0):.3f}"  if clf else '—'
                roc     = f"{clf.get('roc_auc',  0):.3f}"  if clf else '—'
                r2      = f"{reg.get('r2',       0):.3f}"  if reg else '—'
                rmse    = f"{reg.get('rmse',     0):.2f} mm" if reg else '—'
                report.append(f"  {label:<18} {acc:>10} {f1:>8} {roc:>10} {r2:>8} {rmse:>10}")
            report.append("")
            report.append("  Notes:")
            report.append("  • Accuracy / F1 / ROC-AUC: rainfall occurrence classifier (rain vs no rain)")
            report.append("  • R² / RMSE: rainfall amount regressor (mm), trained on rainy days only")
            report.append("  • Negative R² for 30-day regression indicates high uncertainty at that horizon")
        else:
            report.append("\n  (Run src/train_models.py to generate performance metrics)")

        report.append("\n  Key capabilities:")
        report.append("  • Multi-horizon forecasting (1, 7, 30, 90 days ahead)")
        report.append("  • Rainfall occurrence classification (Yes/No + probability)")
        report.append("  • Rainfall amount estimation (mm) on predicted rainy days")

        # ── Seasonal Forecast (2026-2028) ──────────────────────────────────
        if forecasts:
            report.append("\n\n" + "="*70)
            report.append("SEASONAL FORECAST (2026-2028)")
            report.append("="*70)
            report.append(f"\nHistorical baseline: {list(forecasts.values())[0]['hist_avg_mm']:.0f} mm/year (1990-2023 mean)")
            report.append(f"Drought threshold  : <{analysis['drought_threshold']:.0f} mm  |  "
                          f"Flood threshold: >{analysis['extreme_threshold']:.0f} mm")
            report.append("\nMethod: Climatological mean ± inter-annual variability (historical std × 0.4)")
            report.append("        Consistent with the web application's future predictions module.\n")
            report.append(f"  {'Year':<8} {'Predicted':>12} {'vs Normal':>12} {'Outlook':<14} Risk Assessment")
            report.append("  " + "-"*75)
            for year, fc in forecasts.items():
                sign = '+' if fc['diff_pct'] >= 0 else ''
                report.append(
                    f"  {year:<8} {fc['predicted_mm']:>9.0f} mm "
                    f"{sign}{fc['diff_pct']:>+.1f}%{'':<4} "
                    f"{fc['outlook']:<14} {fc['risk']}"
                )
            report.append("")
            report.append("  Note: Forecasts are probabilistic estimates based on historical climatology.")
            report.append("  They do not incorporate real-time atmospheric data or ENSO indices.")

        # Conclusion
        report.append("\n\n" + "="*70)
        report.append("CONCLUSION")
        report.append("="*70)
        report.append("\nThis analysis integrates 33 years of ground-based observations")
        report.append("with ERA5 satellite reanalysis data to provide evidence-based")
        report.append("insights for policy and planning decisions.")
        report.append("\nThe machine learning prediction system enables proactive")
        report.append("decision-making in agriculture, energy, and disaster management")
        report.append("sectors, supporting Zambia's climate resilience and food security.")
        
        report.append("\n\n" + "="*70)
        report.append("END OF REPORT")
        report.append("="*70)
        
        return "\n".join(report)
    
    def _format_markdown_report(self, analysis, agri, energy, disaster, model_summary, model_metrics=None, forecasts=None):
        """Format markdown report"""
        md = []
        md.append("# Rainfall Pattern Prediction for Choma, Zambia")
        md.append("## Policy-Oriented Summary for Stakeholders")
        md.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        md.append(f"**Location:** Choma, Southern Province, Zambia")
        md.append(f"**Data Period:** 1990-2023")

        # Executive Summary
        md.append("\n## Executive Summary")
        md.append(f"\n- **Historical Average Rainfall:** {analysis['historical_avg']:.0f} mm/year")
        md.append(f"- **Recent 5-Year Average:** {analysis['recent_avg']:.0f} mm/year")
        md.append(f"- **Trend:** {analysis['trend'].upper()}")
        md.append(f"- **Variability (CV):** {analysis['cv']:.2f} ({'HIGH' if analysis['cv'] > 0.3 else 'MODERATE'})")

        if analysis['extreme_years']:
            md.append(f"- **Extreme Rainfall Years** (top 10%, >{analysis['extreme_threshold']:.0f} mm): {', '.join(map(str, analysis['extreme_years']))}")
        if analysis['drought_years']:
            md.append(f"- **Drought Years** (bottom 10%, <{analysis['drought_threshold']:.0f} mm): {', '.join(map(str, analysis['drought_years']))}")
        md.append(f"\n> Thresholds based on 10th/90th percentiles of 1990–2023 annual totals (WMO percentile method).")

        # Agriculture
        md.append("\n## Implications for Agriculture")
        for i, imp in enumerate(agri, 1):
            md.append(f"\n### {i}. {imp['finding']}")
            md.append(f"**Implication:** {imp['implication']}")
            md.append(f"\n**Recommendation:** {imp['recommendation']}")

        # Energy
        md.append("\n## Implications for Energy Sector")
        for i, imp in enumerate(energy, 1):
            md.append(f"\n### {i}. {imp['finding']}")
            md.append(f"**Implication:** {imp['implication']}")
            md.append(f"\n**Recommendation:** {imp['recommendation']}")

        # Disaster Management
        md.append("\n## Implications for Disaster Risk Management")
        for i, imp in enumerate(disaster, 1):
            md.append(f"\n### {i}. {imp['finding']}")
            md.append(f"**Implication:** {imp['implication']}")
            md.append(f"\n**Recommendation:** {imp['recommendation']}")

        # ML Model Performance — actual metrics
        md.append("\n## Machine Learning Model Performance")
        md.append("\nRandom Forest models trained on integrated ground station and ERA5 satellite data.")
        md.append("Evaluation on held-out test set (most recent 15% of data, temporal order maintained).")
        md.append("Dataset split: **70% training | 15% validation | 15% test**.")

        horizons_display = [
            ('1day',  '1-Day'),
            ('7day',  '7-Day'),
            ('30day', '30-Day'),
            ('90day', '90-Day'),
        ]

        if model_metrics:
            md.append("\n| Horizon | Accuracy | F1 | ROC-AUC | R² | RMSE |")
            md.append("|---------|----------|-----|---------|-----|------|")
            for key, label in horizons_display:
                m   = model_metrics.get(key, {})
                clf = m.get('clf', {})
                reg = m.get('reg', {})
                acc  = f"{clf.get('accuracy', 0):.1%}" if clf else '—'
                f1   = f"{clf.get('f1',       0):.3f}" if clf else '—'
                roc  = f"{clf.get('roc_auc',  0):.3f}" if clf else '—'
                r2   = f"{reg.get('r2',       0):.3f}" if reg else '—'
                rmse = f"{reg.get('rmse',     0):.2f} mm" if reg else '—'
                md.append(f"| {label} | {acc} | {f1} | {roc} | {r2} | {rmse} |")
            md.append("\n> **Accuracy / F1 / ROC-AUC**: rainfall occurrence classifier (rain vs no rain)  ")
            md.append("> **R² / RMSE**: rainfall amount regressor (mm), trained on rainy days only  ")
            md.append("> Negative R² for 30-day regression indicates high uncertainty at that horizon")
        else:
            md.append("\n*(Run `src/train_models.py` to generate performance metrics)*")

        md.append("\n**Key capabilities:**")
        md.append("- Multi-horizon forecasting (1, 7, 30, 90 days ahead)")
        md.append("- Rainfall occurrence classification (Yes/No + probability)")
        md.append("- Rainfall amount estimation (mm) on predicted rainy days")
        md.append("- Real-time validation against ERA5 API data")

        # Seasonal Forecast (2026-2028)
        if forecasts:
            md.append("\n## Seasonal Forecast (2026-2028)")
            md.append(f"\n**Historical baseline:** {list(forecasts.values())[0]['hist_avg_mm']:.0f} mm/year (1990–2023 mean)  ")
            md.append(f"**Drought threshold:** <{analysis['drought_threshold']:.0f} mm  |  "
                      f"**Flood threshold:** >{analysis['extreme_threshold']:.0f} mm")
            md.append("\n**Method:** Climatological mean ± inter-annual variability (historical std × 0.4), "
                      "consistent with the web application's future predictions module.")
            md.append("\n| Year | Predicted (mm) | vs Normal | Outlook | Risk Assessment |")
            md.append("|------|---------------|-----------|---------|-----------------|")
            for year, fc in forecasts.items():
                sign = '+' if fc['diff_pct'] >= 0 else ''
                md.append(
                    f"| {year} | {fc['predicted_mm']:.0f} mm | "
                    f"{sign}{fc['diff_pct']:.1f}% | **{fc['outlook']}** | {fc['risk']} |"
                )
            md.append("\n> Forecasts are probabilistic estimates based on historical climatology.")
            md.append("> They do not incorporate real-time atmospheric data or ENSO indices.")

        # Conclusion
        md.append("\n## Conclusion")
        md.append("\nThis analysis integrates 33 years of ground-based observations with ERA5 satellite reanalysis data to provide evidence-based insights for policy and planning decisions.")
        md.append("\nThe machine learning prediction system enables proactive decision-making in agriculture, energy, and disaster management sectors, supporting Zambia's climate resilience and food security.")

        return "\n".join(md)


if __name__ == '__main__':
    generator = PolicySummaryGenerator()
    
    print("\nPolicy Summary Generator")
    print("="*70)
    print("\nThis module generates policy-oriented summaries highlighting:")
    print("  • Rainfall variability patterns")
    print("  • Implications for agriculture")
    print("  • Implications for energy sector")
    print("  • Implications for disaster risk management")
    print("\n(Objective 5: Produce policy-oriented summary)")
    
    input("\nPress Enter to generate policy summary...")
    
    try:
        txt_file, md_file = generator.generate_policy_report()
        
        print("\n" + "="*70)
        print("✓ POLICY SUMMARY GENERATED")
        print("="*70)
        print(f"\nText version: {txt_file}")
        print(f"Markdown version: {md_file}")
        print(f"Visualizations: policy_reports/policy_visualizations.png")
        print("\nThis fulfills Objective 5: Policy-oriented summary")
        
    except Exception as e:
        print(f"\n✗ Error generating policy summary: {e}")
        print("\nMake sure data preprocessing has been completed.")

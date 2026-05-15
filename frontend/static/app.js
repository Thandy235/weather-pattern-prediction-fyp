'use strict';

window.addEventListener('DOMContentLoaded', () => {
    const page = window.location.pathname;

    if (page === '/' || page === '') {
        // Forecasts page
        refreshAll();
    } else if (page === '/analysis') {
        // Analysis page
        loadFuturePredictions();
        loadHistoricalData();
        loadRainfallSummary();
    }
});

function refreshAll() {
    // Clear rainfall calendar result and date input
    const dateInput = document.getElementById('predictDateInput');
    const resultDiv = document.getElementById('datePredictResult');
    if (dateInput) dateInput.value = '';
    if (resultDiv) resultDiv.innerHTML = '';

    // Reload forecasts page content
    checkStatus();
    loadPredictions();
}

/* ── Helpers ─────────────────────────────────────────────── */
function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
}

function fmt(n, decimals = 0) {
    if (n === null || n === undefined || isNaN(n)) return '—';
    return Number(n).toLocaleString('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    });
}

/* ── Status pill ─────────────────────────────────────────── */
async function checkStatus() {
    // Status pill is hidden — just run silently in background
    try { await fetch('/api/status'); } catch {}
}

/* ── Current Forecasts ───────────────────────────────────── */
async function loadPredictions() {
    const grid = document.getElementById('forecastGrid');
    if (!grid) return;
    grid.innerHTML = '<div class="loading-state">Loading predictions…</div>';
    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });
        const data = await res.json();
        if (!res.ok || data.error) {
            grid.innerHTML = `<div class="error-msg">${data.error || 'Prediction failed — ensure models are trained.'}</div>`;
            return;
        }
        renderForecasts(data.forecasts);
    } catch (e) {
        grid.innerHTML = '<div class="error-msg">Failed to load predictions.</div>';
    }
}

function renderForecasts(forecasts) {
    const grid = document.getElementById('forecastGrid');
    if (!grid) return;
    grid.innerHTML = '';
    const labels = { '1day': 'Tomorrow', '7day': 'Next 7 Days', '30day': 'Next 30 Days', '90day': 'Seasonal (90 Days)' };
    forecasts.forEach(f => {
        const card = document.createElement('div');
        card.className = `forecast-card ${f.occurrence ? 'rain' : 'no-rain'}`;
        const verdict = f.occurrence ? 'Rain Expected' : 'No Rain';
        const prob    = (f.probability * 100).toFixed(0) + '%';
        const dateStr = new Date(f.target_date).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
        const confClass = f.confidence.toLowerCase().includes('high') ? 'badge-high'
                        : f.confidence.toLowerCase().includes('medium') ? 'badge-medium' : 'badge-low';
        card.innerHTML = `
            <div class="fc-horizon">${labels[f.horizon] || f.horizon}</div>
            <div class="fc-verdict">${verdict}</div>
            <div class="fc-date">${dateStr}</div>
            <div class="fc-divider"></div>
            <div class="fc-row"><span class="fc-label">Rain Probability</span><span class="fc-value">${prob}</span></div>
            ${f.occurrence ? `<div class="fc-row"><span class="fc-label">Est. Amount</span><span class="fc-value">${f.amount.toFixed(1)} mm</span></div>` : ''}
            <div class="fc-row"><span class="fc-label">Confidence</span><span class="badge ${confClass}">${f.confidence}</span></div>
        `;
        grid.appendChild(card);
    });
}

/* ── Future Yearly Predictions ───────────────────────────── */
let _futureData = null;

async function loadFuturePredictions() {
    try {
        const data = await fetch('/api/future_predictions').then(r => r.json());
        if (data.error) return;
        _futureData = data;

        const container = document.getElementById('yearButtons');
        const years = Object.keys(data.future_years);
        years.forEach((yr, i) => {
            const btn = document.createElement('button');
            btn.textContent = yr;
            btn.className = i === 0 ? 'year-btn active' : 'year-btn';
            btn.onclick = () => {
                document.querySelectorAll('#yearButtons .year-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                plotFuture(data, yr);
                // Sync summary panel and disaster risk to same year
                syncSummaryYear(yr);
            };
            container.appendChild(btn);
        });
        plotFuture(data, years[0]);

        // Build summary year buttons with same years
        buildSummaryYearButtons(years);
        // Load summary for first year by default so disaster risk shows immediately
        loadRainfallSummary(years[0]);
    } catch (e) {
        console.error('Future predictions error:', e);
    }
}

function plotFuture(data, year) {
    Plotly.newPlot('futureChart', [
        {
            x: data.months, y: data.historical_avg,
            type: 'bar', name: 'Historical Average (1990–2023)',
            marker: {
                color: 'rgba(147,197,253,0.85)',
                line: { color: '#60a5fa', width: 1 }
            }
        },
        {
            x: data.months, y: data.future_years[year],
            type: 'bar', name: `Predicted ${year}`,
            marker: {
                color: 'rgba(30,58,138,0.90)',
                line: { color: '#1e3a8a', width: 1 }
            }
        }
    ], {
        barmode: 'group',
        xaxis: { title: { text: 'Month', font: { size: 12 } }, tickfont: { size: 12 }, gridcolor: '#f1f5f9' },
        yaxis: { title: { text: 'Monthly Total Rainfall (mm)', font: { size: 12 } }, tickfont: { size: 12 }, gridcolor: '#f1f5f9' },
        legend: { orientation: 'h', x: 0, y: 1.12, font: { size: 12 } },
        hovermode: 'x unified',
        margin: { t: 20, r: 20, b: 50, l: 60 },
        plot_bgcolor: '#fafbfc',
        paper_bgcolor: '#fff',
        font: { family: 'Inter, sans-serif', size: 12 }
    }, { responsive: true, displayModeBar: false });
}

/* ── Historical Chart ────────────────────────────────────── */
async function loadHistoricalData() {
    try {
        const data = await fetch('/api/historical').then(r => r.json());
        if (data.error) {
            document.getElementById('historicalChart').innerHTML = `<div class="error-msg">${data.error}</div>`;
            return;
        }
        Plotly.newPlot('historicalChart', [
            {
                x: data.dates, y: data.rainfall,
                type: 'bar', name: 'Monthly Rainfall (mm)',
                marker: {
                    color: 'rgba(30,58,138,0.88)',
                    line: { color: '#1e3a8a', width: 1 }
                }
            },
            {
                x: data.dates, y: data.max_temp,
                type: 'scatter', name: 'Avg Max Temp (°C)',
                yaxis: 'y2', line: { color: '#93c5fd', width: 2 },
                mode: 'lines+markers', marker: { size: 4, color: '#93c5fd' }
            },
            {
                x: data.dates, y: data.min_temp,
                type: 'scatter', name: 'Avg Min Temp (°C)',
                yaxis: 'y2', line: { color: '#bfdbfe', width: 2, dash: 'dot' },
                mode: 'lines+markers', marker: { size: 4, color: '#bfdbfe' }
            }
        ], {
            xaxis: { title: { text: 'Month', font: { size: 12 } }, tickfont: { size: 11 }, gridcolor: '#f1f5f9' },
            yaxis: { title: { text: 'Monthly Rainfall (mm)', font: { size: 12 } }, tickfont: { size: 11 }, gridcolor: '#f1f5f9' },
            yaxis2: { title: { text: 'Temperature (°C)', font: { size: 12 } }, overlaying: 'y', side: 'right', tickfont: { size: 11 } },
            legend: { orientation: 'h', x: 0, y: 1.12, font: { size: 12 } },
            hovermode: 'x unified',
            margin: { t: 20, r: 60, b: 50, l: 60 },
            plot_bgcolor: '#fafbfc',
            paper_bgcolor: '#fff',
            font: { family: 'Inter, sans-serif', size: 12 }
        }, { responsive: true, displayModeBar: false });
    } catch (e) {
        console.error('Historical data error:', e);
    }
}

/* ── Pattern Lookup ──────────────────────────────────────── */
async function predictForDate() {
    const input     = document.getElementById('predictDateInput');
    const resultDiv = document.getElementById('datePredictResult');
    if (!input.value) {
        resultDiv.innerHTML = '<div class="error-msg">Please select a date.</div>';
        return;
    }
    resultDiv.innerHTML = '<div class="loading-state">Looking up pattern…</div>';
    try {
        const res  = await fetch('/api/predict_date', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ date: input.value })
        });
        const data = await res.json();
        if (!res.ok || data.error) {
            resultDiv.innerHTML = `<div class="error-msg">${data.error || 'Unknown error'}</div>`;
            return;
        }
        resultDiv.innerHTML = `
            <div class="pattern-result">
                <div class="pr-block pr-full-width">
                    <span class="pr-season-badge">${data.season}</span>
                </div>
                <div class="pr-block">
                    <div class="pr-label">Pattern</div>
                    <div class="pr-value">${data.occurrence_text}</div>
                    <div class="pr-note">${data.date}</div>
                </div>
                <div class="pr-block">
                    <div class="pr-label">Historical Rain Occurrence Rate</div>
                    <div class="pr-value">${data.rain_probability_pct}%</div>
                    <div class="pr-note">of days in this window had rain</div>
                </div>
                ${data.avg_rainfall_on_rainy_days > 0 ? `
                <div class="pr-block">
                    <div class="pr-label">Avg Rainfall on Wet Days</div>
                    <div class="pr-value">${data.avg_rainfall_on_rainy_days} mm</div>
                    <div class="pr-note">when it does rain</div>
                </div>
                <div class="pr-block">
                    <div class="pr-label">Maximum Recorded</div>
                    <div class="pr-value">${data.max_recorded} mm</div>
                    <div class="pr-note">single-day record for this period</div>
                </div>` : ''}
            </div>
        `;
    } catch {
        resultDiv.innerHTML = '<div class="error-msg">Prediction failed.</div>';
    }
}

/* ── Rainfall Summary Panel ──────────────────────────────── */
async function loadRainfallSummary(compareYear = null) {
    try {
        const url  = compareYear ? `/api/rainfall_summary?year=${compareYear}` : '/api/rainfall_summary';
        const data = await fetch(url).then(r => r.json());
        if (data.error) return;
        renderSummary(data);
    } catch (e) {
        console.error('Summary error:', e);
    }
}

function renderSummary(data) {
    // ── Predicted Annual card (FIRST) ──────────────────────────────────
    const predCard = document.getElementById('kpiPredCard');
    if (data.predicted_annual !== null && data.predicted_annual !== undefined && data.compare_year) {
        setText('kpiPredVal', fmt(data.predicted_annual) + ' mm');
        setText('kpiPredSub', 'Predicted ' + data.compare_year);
        if (predCard) predCard.style.display = '';
        if (predCard) {
            const diff = data.predicted_annual - data.hist_avg;
            predCard.className = 'kpi-card ' + (diff > 60 ? 'accent-teal' : diff < -60 ? 'accent-red' : 'accent-green');
        }
    } else {
        if (predCard) predCard.style.display = 'none';
    }

    // ── Year outlook badge ─────────────────────────────────────────────
    const outlookEl = document.getElementById('kpiYearOutlook');
    if (outlookEl) {
        if (data.year_outlook && data.compare_year) {
            const colorMap = {
                'Dry Year':    'background:#f1f5f9;color:#475569;border:1px solid #cbd5e1;',
                'Wet Year':    'background:#eff6ff;color:#1e40af;border:1px solid #bfdbfe;',
                'Normal Year': 'background:#f0fdf4;color:#166534;border:1px solid #bbf7d0;'
            };
            const style = colorMap[data.year_outlook] || '';
            outlookEl.innerHTML = `<span style="display:inline-block;padding:3px 12px;border-radius:20px;font-size:12px;font-weight:600;${style}">${data.year_outlook}</span>`;
            outlookEl.style.display = '';
        } else {
            outlookEl.style.display = 'none';
        }
    }

    // ── KPI cards ──────────────────────────────────────────────────────
    setText('kpiHistAvg', fmt(data.hist_avg) + ' mm');

    // Rainy days: show predicted value when a year is selected, else historical avg
    if (data.compare_year && data.predicted_rainy_days !== null && data.predicted_rainy_days !== undefined) {
        setText('kpiRainyDays', fmt(data.predicted_rainy_days) + ' days');
        setText('kpiRainyDaysSub', 'Estimated for ' + data.compare_year);
    } else {
        setText('kpiRainyDays', fmt(data.avg_rainy_days) + ' days');
        setText('kpiRainyDaysSub', 'Days with rainfall ≥ 1 mm');
    }

    setText('kpiMinYr',  data.hist_min_yr);
    setText('kpiMinVal', fmt(data.hist_min) + ' mm total');
    setText('kpiMaxYr',  data.hist_max_yr);
    setText('kpiMaxVal', fmt(data.hist_max) + ' mm total');

    // Future risk alert in disaster panel
    const alertBox = document.getElementById('futureRiskAlert');
    if (alertBox) {
        if (data.compare_year && data.future_drought_risk) {
            alertBox.style.display = '';
            alertBox.innerHTML = `<div style="background:#f8fafc;border-left:3px solid #64748b;border-radius:4px;padding:10px 14px;font-size:12px;color:#334155;line-height:1.6;">
                <strong style="color:#1e293b;">${data.compare_year} Drought Risk:</strong> Predicted annual total of ${fmt(data.predicted_annual)} mm falls below the drought threshold of ${fmt(data.drought_threshold)} mm. Authorities should activate early warning systems and contingency plans.
            </div>`;
        } else if (data.compare_year && data.future_flood_risk) {
            alertBox.style.display = '';
            alertBox.innerHTML = `<div style="background:#f8fafc;border-left:3px solid #3b82f6;border-radius:4px;padding:10px 14px;font-size:12px;color:#334155;line-height:1.6;">
                <strong style="color:#1e293b;">${data.compare_year} Flood Risk:</strong> Predicted annual total of ${fmt(data.predicted_annual)} mm exceeds the flood threshold of ${fmt(data.flood_threshold)} mm. Flood early warnings and evacuation plans should be prepared.
            </div>`;
        } else if (data.compare_year) {
            alertBox.style.display = '';
            alertBox.innerHTML = `<div style="background:#f8fafc;border-left:3px solid #94a3b8;border-radius:4px;padding:10px 14px;font-size:12px;color:#334155;line-height:1.6;">
                <strong style="color:#1e293b;">${data.compare_year} Outlook:</strong> Predicted total of ${fmt(data.predicted_annual)} mm is within the normal range, no drought or flood risk indicated.
            </div>`;
        } else {
            // No year selected — show placeholder
            alertBox.style.display = '';
            alertBox.innerHTML = `<div style="background:var(--bg);border-radius:6px;padding:10px 12px;font-size:12px;color:var(--text-muted);line-height:1.5;">
                Select a year above to see the future risk outlook.
            </div>`;
        }
    }

    // Planting calendar
    // When a future year is selected, show predicted monthly values.
    // When historical only, show the climatological averages.
    const grid = document.getElementById('plantingGrid');
    if (grid && data.month_names) {
        const values = (data.compare_year && data.predicted_monthly)
            ? data.predicted_monthly
            : data.monthly_clim;

        const calTitle = document.querySelector('#sec-summary .panel-title');
        grid.innerHTML = '';

        // Update the panel title to show which data is displayed
        const plantingTitle = grid.closest('.summary-panel')
            ? grid.closest('.summary-panel').querySelector('.panel-title')
            : null;
        if (plantingTitle) {
            plantingTitle.textContent = data.compare_year
                ? `Farmer Planting Calendar (${data.compare_year})`
                : 'Farmer Planting Calendar';
        }

        data.month_names.forEach((m, i) => {
            const val = values ? values[i] : data.monthly_clim[i];
            const wet = val >= 25;
            const div = document.createElement('div');
            div.className = `pm ${wet ? 'wet' : 'dry'}`;
            div.innerHTML = `<span class="pm-name">${m}</span><span class="pm-val">${val} mm</span>`;
            grid.appendChild(div);
        });
    }

    // Drought / flood risk years
    const droughtDiv = document.getElementById('droughtYearTags');
    const floodDiv   = document.getElementById('floodYearTags');
    if (droughtDiv && data.drought_years) {
        droughtDiv.innerHTML = data.drought_years.length
            ? data.drought_years.map(y => `<span class="risk-year-tag drought">${y}</span>`).join('')
            : '<span style="color:var(--text-muted);font-size:12px">None in record</span>';
    }
    if (floodDiv && data.flood_years) {
        floodDiv.innerHTML = data.flood_years.length
            ? data.flood_years.map(y => `<span class="risk-year-tag flood">${y}</span>`).join('')
            : '<span style="color:var(--text-muted);font-size:12px">None in record</span>';
    }
    if (data.drought_threshold) setText('droughtThresh', fmt(data.drought_threshold) + ' mm');
    if (data.flood_threshold)   setText('floodThresh',   fmt(data.flood_threshold)   + ' mm');
}

function buildSummaryYearButtons(years) {
    const container = document.getElementById('summaryYearButtons');
    if (!container) return;
    container.innerHTML = '';

    years.forEach(yr => {
        const btn = document.createElement('button');
        btn.textContent = yr;
        btn.className = 'year-btn';
        btn.onclick = () => {
            container.querySelectorAll('.year-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            // Sync future chart buttons too
            document.querySelectorAll('#yearButtons .year-btn').forEach(b => {
                b.classList.toggle('active', b.textContent === yr);
            });
            if (_futureData) plotFuture(_futureData, yr);
            loadRainfallSummary(yr);
        };
        container.appendChild(btn);
    });

    // Set first year active by default
    if (container.firstChild) container.firstChild.classList.add('active');
}

function syncSummaryYear(yr) {
    // Keep summary year buttons in sync when future chart buttons are clicked
    const container = document.getElementById('summaryYearButtons');
    if (!container) return;
    container.querySelectorAll('.year-btn').forEach(b => {
        b.classList.toggle('active', b.textContent === yr);
    });
    loadRainfallSummary(yr);
}

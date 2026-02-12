# RiverData Regression Lab

Streamlit app for multipanel regression visualization and outlier diagnostics from CSV/XLSX uploads.

## Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Core Features

- Drag-and-drop CSV/XLSX upload and dataframe preview
- Configurable multipanel (default 2x2) regression plots
- OLS regression with 95% CI/PI and optional robust regression line (RLM)
- Per-panel toggles for showing/hiding 95% CI, 95% PI, and legend
- Panel-wise influence diagnostics (`rstudent`, Cook's D, leverage)
- Optional outlier-candidate removal with before/after metric comparison
- Export both full grid figure and individual panel figures
- Adjustable PNG DPI (300 to 1200) for high-resolution output
- Adjustable figure style defaults (title/axis/tick/legend font sizes, marker size, line width, panel size)

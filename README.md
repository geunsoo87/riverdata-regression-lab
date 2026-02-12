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
- Optional outlier-candidate marker overlay on plots (red outline)
- Panel-wise influence diagnostics (`rstudent`, Cook's D, leverage)
- Per-panel coefficient/stat summary table for OLS, Huber, and Tukey models
- Full per-panel diagnostics table for all rows with auto candidate flag, outside-95%CI/PI flags, and manual review checkbox
- Optional outlier-candidate removal with before/after metric comparison
- Export both full grid figure and individual panel figures
- Adjustable PNG DPI (300 to 1200) for high-resolution output
- Adjustable figure style defaults (title/axis/tick/legend font sizes, marker size, line width, panel size)
- Preset JSON save/load for reusing the same panel/style settings on new files
- In-app Help button to open detailed usage and interpretation guide

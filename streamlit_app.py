import io
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.mathtext import MathTextParser
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st

st.set_page_config(page_title="RiverData Regression Lab", layout="wide")

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
    }
)

_MATH_PARSER = MathTextParser("path")
_RSTUDENT_THRESHOLD = 3.0
_DEFAULT_STYLE_CONFIG = {
    "title_size": 16,
    "label_size": 14,
    "tick_size": 12,
    "legend_size": 12,
    "point_size": 36,
    "line_width": 2.8,
    "panel_width": 6.4,
    "panel_height": 5.0,
}
_STYLE_SESSION_MAP = {
    "title_size": "style_title_size",
    "label_size": "style_label_size",
    "tick_size": "style_tick_size",
    "legend_size": "style_legend_size",
    "point_size": "style_point_size",
    "line_width": "style_line_width",
    "panel_width": "style_panel_width",
    "panel_height": "style_panel_height",
}


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(name))
    cleaned = cleaned.strip("._-")
    return cleaned or "figure"


def _metrics_from_fit(fit_result: dict[str, Any]) -> dict[str, float]:
    params = np.asarray(fit_result["line_params"], dtype=float)
    intercept = float(params[0]) if params.size > 0 else float("nan")
    slope = float(params[1]) if params.size > 1 else float("nan")
    return {
        "Intercept": intercept,
        "Slope": slope,
        "R2": float(fit_result["ols_r2"]),
        "RMSE": float(fit_result["ols_rmse"]),
    }


def _comparison_frame(before_fit: dict[str, Any], after_fit: dict[str, Any]) -> pd.DataFrame:
    before = _metrics_from_fit(before_fit)
    after = _metrics_from_fit(after_fit)

    return pd.DataFrame(
        {
            "Metric": ["Intercept", "Slope", "R2", "RMSE"],
            "Before": [
                before["Intercept"],
                before["Slope"],
                before["R2"],
                before["RMSE"],
            ],
            "After": [
                after["Intercept"],
                after["Slope"],
                after["R2"],
                after["RMSE"],
            ],
        }
    )


def _default_export_filename(panel_configs: list[dict[str, Any]], ext: str) -> str:
    date_tag = datetime.now().strftime("%Y%m%d")
    parts: list[str] = []
    for cfg in panel_configs[:4]:
        parts.append(_sanitize_filename(f"{cfg['y_col']}_vs_{cfg['x_col']}"))

    joined = "__".join([p for p in parts if p])
    if not joined:
        joined = "regression"

    if len(joined) > 80:
        joined = joined[:80]

    return f"{date_tag}_{joined}.{ext}"


def _default_panel_export_filename(panel_idx: int, panel_cfg: dict[str, Any], ext: str) -> str:
    date_tag = datetime.now().strftime("%Y%m%d")
    core = _sanitize_filename(
        f"panel{panel_idx + 1}_{panel_cfg['y_col']}_vs_{panel_cfg['x_col']}"
    )
    return f"{date_tag}_{core}.{ext}"


@st.cache_data(show_spinner=False)
def load_uploaded_data(file_bytes: bytes, filename: str) -> pd.DataFrame:
    extension = filename.lower().split(".")[-1]

    if extension == "csv":
        last_error: Exception | None = None
        for encoding in ("utf-8-sig", "cp949"):
            try:
                return pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
            except Exception as exc:  # pragma: no cover - user input dependent
                last_error = exc
        raise ValueError(f"CSV decode failed for utf-8-sig/cp949: {last_error}")

    if extension in {"xlsx", "xls"}:
        return pd.read_excel(io.BytesIO(file_bytes))

    raise ValueError("Unsupported file extension. Use CSV or XLSX.")


def make_panel_label(panel_idx: int, caption: str) -> str:
    caption_clean = caption.strip()
    return caption_clean


def safe_mathtext(label: str, fallback: str, warn_bucket: list[str], context: str) -> str:
    candidate = (label or "").strip()
    if not candidate:
        return fallback

    should_parse = "$" in candidate or "\\" in candidate
    if not should_parse:
        return candidate

    try:
        _MATH_PARSER.parse(candidate, dpi=72)
        return candidate
    except Exception:
        warn_bucket.append(
            f"{context}: invalid LaTeX/MathText label. Falling back to '{fallback}'."
        )
        return fallback


def prepare_panel_data(
    df: pd.DataFrame,
    id_series: pd.Series,
    x_col: str,
    y_col: str,
    log_y: bool,
) -> tuple[pd.DataFrame, dict[str, int], list[str]]:
    warnings: list[str] = []

    panel_df = pd.DataFrame(
        {
            "ROW_KEY": pd.Series(np.arange(len(df))).astype(str),
            "ID": id_series.astype(str),
            "X": pd.to_numeric(df[x_col], errors="coerce"),
            "Y": pd.to_numeric(df[y_col], errors="coerce"),
        }
    )

    dropped_xy_nan = int(panel_df[["X", "Y"]].isna().any(axis=1).sum())
    if dropped_xy_nan > 0:
        warnings.append(
            "Rows with non-numeric or missing X/Y values were dropped: "
            f"{dropped_xy_nan}."
        )

    fit_df = panel_df.dropna(subset=["X", "Y"]).copy()

    dropped_log_nonpositive = 0
    if log_y:
        non_positive_mask = fit_df["Y"] <= 0
        dropped_log_nonpositive = int(non_positive_mask.sum())
        if dropped_log_nonpositive > 0:
            warnings.append(
                "Y log10 enabled. Non-positive Y rows were dropped: "
                f"{dropped_log_nonpositive}."
            )
        fit_df = fit_df.loc[~non_positive_mask].copy()
        fit_df["Y"] = np.log10(fit_df["Y"])

    counts = {
        "input_rows": int(len(panel_df)),
        "dropped_xy_nan": dropped_xy_nan,
        "dropped_log_nonpositive": dropped_log_nonpositive,
        "remaining_rows": int(len(fit_df)),
    }

    return fit_df.reset_index(drop=True), counts, warnings


def fit_ols_and_optional_robust(
    x: np.ndarray,
    y: np.ndarray,
    robust_enabled: bool,
    robust_norm: str,
) -> dict[str, Any]:
    x_flat = np.asarray(x, dtype=float).reshape(-1)
    y_flat = np.asarray(y, dtype=float).reshape(-1)

    design = sm.add_constant(x_flat, has_constant="add")
    ols_model = sm.OLS(y_flat, design).fit()

    robust_model = None
    robust_error = None
    line_params = np.asarray(ols_model.params, dtype=float)

    if robust_enabled:
        norm = (
            sm.robust.norms.TukeyBiweight()
            if robust_norm.lower().startswith("t")
            else sm.robust.norms.HuberT()
        )
        try:
            robust_model = sm.RLM(y_flat, design, M=norm).fit()
            line_params = np.asarray(robust_model.params, dtype=float)
        except Exception as exc:  # pragma: no cover - data dependent
            robust_error = str(exc)

    rmse = float(np.sqrt(np.mean(np.square(ols_model.resid))))

    return {
        "ols_model": ols_model,
        "robust_model": robust_model,
        "robust_error": robust_error,
        "line_params": line_params,
        "used_robust": robust_enabled and robust_model is not None,
        "ols_r2": float(ols_model.rsquared),
        "ols_rmse": rmse,
    }


def compute_prediction_frames(ols_model: Any, x_grid: np.ndarray) -> pd.DataFrame:
    x_values = np.asarray(x_grid, dtype=float).reshape(-1)
    pred_design = sm.add_constant(x_values, has_constant="add")
    summary = ols_model.get_prediction(pred_design).summary_frame(alpha=0.05)

    return pd.DataFrame(
        {
            "X": x_values,
            "mean": summary["mean"].to_numpy(),
            "mean_ci_lower": summary["mean_ci_lower"].to_numpy(),
            "mean_ci_upper": summary["mean_ci_upper"].to_numpy(),
            "obs_ci_lower": summary["obs_ci_lower"].to_numpy(),
            "obs_ci_upper": summary["obs_ci_upper"].to_numpy(),
        }
    )


def compute_influence_table(fit_df: pd.DataFrame, ols_model: Any) -> pd.DataFrame:
    influence = ols_model.get_influence()

    rstudent = influence.resid_studentized_external
    cooks_d = influence.cooks_distance[0]
    leverage = influence.hat_matrix_diag

    n = int(len(fit_df))
    cook_threshold = 4.0 / n if n > 0 else float("inf")

    table = fit_df[["ROW_KEY", "ID", "X", "Y"]].copy()
    table = table.rename(columns={"ROW_KEY": "row_key"})
    table["rstudent"] = pd.to_numeric(rstudent, errors="coerce")
    table["cooks_d"] = pd.to_numeric(cooks_d, errors="coerce")
    table["leverage"] = pd.to_numeric(leverage, errors="coerce")
    table["is_outlier_candidate"] = (
        table["rstudent"].abs() > _RSTUDENT_THRESHOLD
    ) | (table["cooks_d"] > cook_threshold)

    return table


def apply_axis_settings(
    ax: Any,
    cfg: dict[str, Any],
    transformed_y: bool,
    warn_bucket: list[str],
) -> None:
    x_mode = cfg.get("x_mode", "auto")
    y_mode = cfg.get("y_mode", "auto")

    if x_mode == "manual":
        x_min = _safe_float(cfg.get("x_min"))
        x_max = _safe_float(cfg.get("x_max"))
        if x_min is None or x_max is None or not x_min < x_max:
            warn_bucket.append("Invalid manual X range. Using auto range.")
        else:
            ax.set_xlim(x_min, x_max)

    if y_mode == "manual":
        y_min = _safe_float(cfg.get("y_min"))
        y_max = _safe_float(cfg.get("y_max"))
        if y_min is None or y_max is None or not y_min < y_max:
            axis_name = "log10(Y)" if transformed_y else "Y"
            warn_bucket.append(f"Invalid manual {axis_name} range. Using auto range.")
        else:
            ax.set_ylim(y_min, y_max)

    x_tick = _safe_float(cfg.get("x_tick"))
    if x_tick is not None:
        if x_tick <= 0:
            warn_bucket.append("X tick interval must be positive. Ignored.")
        else:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(x_tick))

    y_tick = _safe_float(cfg.get("y_tick"))
    if y_tick is not None:
        if y_tick <= 0:
            warn_bucket.append("Y tick interval must be positive. Ignored.")
        else:
            ax.yaxis.set_major_locator(ticker.MultipleLocator(y_tick))


def render_panel(ax: Any, panel_cfg: dict[str, Any], panel_result: dict[str, Any]) -> None:
    style_cfg = panel_result.get("style_cfg", {})
    title_size = float(style_cfg.get("title_size", _DEFAULT_STYLE_CONFIG["title_size"]))
    label_size = float(style_cfg.get("label_size", _DEFAULT_STYLE_CONFIG["label_size"]))
    tick_size = float(style_cfg.get("tick_size", _DEFAULT_STYLE_CONFIG["tick_size"]))
    legend_size = float(style_cfg.get("legend_size", _DEFAULT_STYLE_CONFIG["legend_size"]))
    point_size = float(style_cfg.get("point_size", _DEFAULT_STYLE_CONFIG["point_size"]))
    line_width = float(style_cfg.get("line_width", _DEFAULT_STYLE_CONFIG["line_width"]))
    show_ci = bool(panel_cfg.get("show_ci", True))
    show_pi = bool(panel_cfg.get("show_pi", True))
    show_legend = bool(panel_cfg.get("show_legend", True))

    panel_title = panel_cfg.get("panel_title", "").strip()
    if panel_title:
        ax.set_title(panel_title, loc="left", fontweight="bold", fontsize=title_size)

    if panel_result["status"] != "ok":
        ax.text(
            0.5,
            0.5,
            panel_result.get("message", "Unable to render panel"),
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=max(10.0, tick_size),
        )
        ax.set_xlabel(panel_cfg["x_label"], fontsize=label_size)
        ax.set_ylabel(panel_cfg["y_label"], fontsize=label_size)
        ax.tick_params(axis="both", labelsize=tick_size)
        ax.grid(True, linestyle=":", linewidth=0.6, color="0.85")
        return

    plot_df = panel_result["plot_df"]
    pred_df = panel_result["pred_df"]
    line_df = panel_result["line_df"]

    if show_pi:
        ax.fill_between(
            pred_df["X"],
            pred_df["obs_ci_lower"],
            pred_df["obs_ci_upper"],
            color="0.90",
            alpha=1.0,
            label="95% PI",
            zorder=1,
        )
    if show_ci:
        ax.fill_between(
            pred_df["X"],
            pred_df["mean_ci_lower"],
            pred_df["mean_ci_upper"],
            color="0.78",
            alpha=1.0,
            label="95% CI",
            zorder=2,
        )
    ax.scatter(
        plot_df["X"],
        plot_df["Y"],
        color="black",
        s=point_size,
        alpha=0.8,
        edgecolors="none",
        label="Data",
        zorder=3,
    )
    ax.plot(
        line_df["X"],
        line_df["Y"],
        color="black",
        linewidth=line_width,
        label=panel_result.get("line_label", "Regression"),
        zorder=4,
    )

    ax.set_xlabel(panel_cfg["x_label"], fontsize=label_size)
    ax.set_ylabel(panel_cfg["y_label"], fontsize=label_size)
    ax.tick_params(axis="both", labelsize=tick_size)

    apply_axis_settings(
        ax,
        panel_cfg,
        transformed_y=bool(panel_cfg.get("log_y", False)),
        warn_bucket=panel_result["warnings"],
    )

    ax.grid(True, linestyle=":", linewidth=0.6, color="0.85")
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(frameon=False, loc="best", prop={"size": legend_size})


def export_figure_bytes(fig: Any, fmt: str, png_dpi: int = 300) -> tuple[bytes, str]:
    fmt_lower = fmt.lower()
    mime_map = {
        "png": "image/png",
        "pdf": "application/pdf",
        "svg": "image/svg+xml",
    }
    if fmt_lower not in mime_map:
        raise ValueError("Unsupported export format")

    buf = io.BytesIO()
    save_kwargs: dict[str, Any] = {"format": fmt_lower, "bbox_inches": "tight"}
    if fmt_lower == "png":
        save_kwargs["dpi"] = int(png_dpi)

    fig.savefig(buf, **save_kwargs)
    buf.seek(0)
    return buf.getvalue(), mime_map[fmt_lower]


def create_single_panel_figure(panel_cfg: dict[str, Any], panel_result: dict[str, Any]) -> Any:
    style_cfg = panel_result.get("style_cfg", {})
    panel_width = float(style_cfg.get("panel_width", _DEFAULT_STYLE_CONFIG["panel_width"]))
    panel_height = float(style_cfg.get("panel_height", _DEFAULT_STYLE_CONFIG["panel_height"]))
    fig, ax = plt.subplots(1, 1, figsize=(panel_width, panel_height))

    result_copy = panel_result.copy()
    result_copy["warnings"] = list(panel_result.get("warnings", []))

    render_panel(ax, panel_cfg, result_copy)
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def load_help_markdown() -> str:
    help_path = Path(__file__).with_name("HELP.md")
    if help_path.exists():
        return help_path.read_text(encoding="utf-8")

    return (
        "# Help\n\n"
        "HELP.md file not found. Add HELP.md next to streamlit_app.py to show guide content."
    )


def build_settings_preset_payload(
    panel_configs: list[dict[str, Any]],
    style_cfg: dict[str, Any],
    global_label_map: dict[str, Any],
) -> dict[str, Any]:
    return {
        "version": 1,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "panel_count": int(len(panel_configs)),
        "panel_configs": panel_configs,
        "style_cfg": style_cfg,
        "global_label_map": {str(k): str(v) for k, v in dict(global_label_map).items()},
    }


def _cast_style_value(style_key: str, value: Any) -> float | int:
    if style_key in {"line_width", "panel_width", "panel_height"}:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(_DEFAULT_STYLE_CONFIG[style_key])

    try:
        return int(value)
    except (TypeError, ValueError):
        return int(_DEFAULT_STYLE_CONFIG[style_key])


def _sanitize_loaded_panel_config(raw_cfg: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    fallback_x = columns[0]
    fallback_y = columns[1] if len(columns) > 1 else columns[0]

    x_col = str(raw_cfg.get("x_col", fallback_x))
    y_col = str(raw_cfg.get("y_col", fallback_y))
    if x_col not in columns:
        x_col = fallback_x
    if y_col not in columns:
        y_col = fallback_y

    log_y = bool(raw_cfg.get("log_y", False))
    x_min = _safe_float(raw_cfg.get("x_min"))
    x_max = _safe_float(raw_cfg.get("x_max"))
    y_min = _safe_float(raw_cfg.get("y_min"))
    y_max = _safe_float(raw_cfg.get("y_max"))

    return {
        "x_col": x_col,
        "y_col": y_col,
        "log_y": log_y,
        "caption": str(raw_cfg.get("caption", "")),
        "x_label": str(raw_cfg.get("x_label", x_col)),
        "y_label": str(raw_cfg.get("y_label", f"log10({y_col})" if log_y else y_col)),
        "x_mode": str(raw_cfg.get("x_mode", "auto")) if str(raw_cfg.get("x_mode", "auto")) in {"auto", "manual"} else "auto",
        "x_min": 0.0 if x_min is None else float(x_min),
        "x_max": 1.0 if x_max is None else float(x_max),
        "y_mode": str(raw_cfg.get("y_mode", "auto")) if str(raw_cfg.get("y_mode", "auto")) in {"auto", "manual"} else "auto",
        "y_min": 0.0 if y_min is None else float(y_min),
        "y_max": 1.0 if y_max is None else float(y_max),
        "x_tick": str(raw_cfg.get("x_tick", "")),
        "y_tick": str(raw_cfg.get("y_tick", "")),
        "remove_outliers": bool(raw_cfg.get("remove_outliers", False)),
        "show_ci": bool(raw_cfg.get("show_ci", True)),
        "show_pi": bool(raw_cfg.get("show_pi", True)),
        "show_legend": bool(raw_cfg.get("show_legend", True)),
    }


def _apply_panel_widget_state(panel_idx: int, panel_cfg: dict[str, Any]) -> None:
    st.session_state[f"panel_{panel_idx}_x_col"] = panel_cfg["x_col"]
    st.session_state[f"panel_{panel_idx}_y_col"] = panel_cfg["y_col"]
    st.session_state[f"panel_{panel_idx}_log_y"] = bool(panel_cfg["log_y"])
    st.session_state[f"panel_{panel_idx}_caption"] = str(panel_cfg["caption"])
    st.session_state[f"panel_{panel_idx}_x_label"] = str(panel_cfg["x_label"])
    st.session_state[f"panel_{panel_idx}_y_label"] = str(panel_cfg["y_label"])
    st.session_state[f"panel_{panel_idx}_x_mode"] = str(panel_cfg["x_mode"])
    st.session_state[f"panel_{panel_idx}_y_mode"] = str(panel_cfg["y_mode"])
    st.session_state[f"panel_{panel_idx}_x_min"] = float(panel_cfg["x_min"])
    st.session_state[f"panel_{panel_idx}_x_max"] = float(panel_cfg["x_max"])
    st.session_state[f"panel_{panel_idx}_y_min"] = float(panel_cfg["y_min"])
    st.session_state[f"panel_{panel_idx}_y_max"] = float(panel_cfg["y_max"])
    st.session_state[f"panel_{panel_idx}_x_tick"] = str(panel_cfg["x_tick"])
    st.session_state[f"panel_{panel_idx}_y_tick"] = str(panel_cfg["y_tick"])
    st.session_state[f"panel_{panel_idx}_remove_outliers"] = bool(panel_cfg["remove_outliers"])
    st.session_state[f"panel_{panel_idx}_show_ci"] = bool(panel_cfg["show_ci"])
    st.session_state[f"panel_{panel_idx}_show_pi"] = bool(panel_cfg["show_pi"])
    st.session_state[f"panel_{panel_idx}_show_legend"] = bool(panel_cfg["show_legend"])
    st.session_state[f"panel_{panel_idx}_x_prev"] = panel_cfg["x_col"]
    st.session_state[f"panel_{panel_idx}_y_prev"] = panel_cfg["y_col"]
    st.session_state[f"panel_{panel_idx}_log_prev"] = bool(panel_cfg["log_y"])


def apply_loaded_preset_to_session(
    preset_obj: dict[str, Any],
    columns: list[str],
) -> tuple[list[str], int]:
    warnings: list[str] = []
    if not columns:
        raise ValueError("No columns available for preset mapping.")

    panel_count_raw = preset_obj.get("panel_count", 4)
    try:
        panel_count = int(panel_count_raw)
    except (TypeError, ValueError):
        panel_count = 4
        warnings.append("Invalid panel_count in preset. Using default 4.")
    panel_count = min(12, max(1, panel_count))

    raw_panel_configs = preset_obj.get("panel_configs", [])
    if not isinstance(raw_panel_configs, list):
        raise ValueError("panel_configs in preset must be a list.")

    normalized_panel_configs: list[dict[str, Any]] = []
    for i in range(panel_count):
        raw_cfg = raw_panel_configs[i] if i < len(raw_panel_configs) else {}
        if not isinstance(raw_cfg, dict):
            raw_cfg = {}
            warnings.append(f"panel_configs[{i}] is invalid. Replaced with defaults.")
        normalized_panel_configs.append(_sanitize_loaded_panel_config(raw_cfg, columns))

    st.session_state["panel_count"] = panel_count
    st.session_state["panel_configs"] = {
        idx: cfg for idx, cfg in enumerate(normalized_panel_configs)
    }

    loaded_label_map = preset_obj.get("global_label_map", {})
    if isinstance(loaded_label_map, dict):
        st.session_state["global_label_map"] = {
            str(k): str(v) for k, v in loaded_label_map.items()
        }
    else:
        st.session_state["global_label_map"] = {}
        warnings.append("global_label_map is invalid. Cleared.")

    loaded_style_cfg = preset_obj.get("style_cfg", {})
    if not isinstance(loaded_style_cfg, dict):
        loaded_style_cfg = {}
        warnings.append("style_cfg is invalid. Using defaults.")

    for style_key, session_key in _STYLE_SESSION_MAP.items():
        st.session_state[session_key] = _cast_style_value(
            style_key,
            loaded_style_cfg.get(style_key, _DEFAULT_STYLE_CONFIG[style_key]),
        )

    for idx, cfg in enumerate(normalized_panel_configs):
        _apply_panel_widget_state(idx, cfg)

    st.session_state["manual_review_flags"] = {}
    return warnings, panel_count


def _ensure_state(key: str, value: Any) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


def main() -> None:
    st.title("RiverData Regression Lab")
    st.caption(
        "Upload CSV/XLSX data, configure per-panel regression settings, and export "
        "SCI-style multipanel figures with diagnostics."
    )

    _ensure_state("panel_count", 4)
    _ensure_state("panel_configs", {})
    _ensure_state("global_label_map", {})
    _ensure_state("style_title_size", _DEFAULT_STYLE_CONFIG["title_size"])
    _ensure_state("style_label_size", _DEFAULT_STYLE_CONFIG["label_size"])
    _ensure_state("style_tick_size", _DEFAULT_STYLE_CONFIG["tick_size"])
    _ensure_state("style_legend_size", _DEFAULT_STYLE_CONFIG["legend_size"])
    _ensure_state("style_point_size", _DEFAULT_STYLE_CONFIG["point_size"])
    _ensure_state("style_line_width", _DEFAULT_STYLE_CONFIG["line_width"])
    _ensure_state("style_panel_width", _DEFAULT_STYLE_CONFIG["panel_width"])
    _ensure_state("style_panel_height", _DEFAULT_STYLE_CONFIG["panel_height"])
    _ensure_state("manual_review_flags", {})
    _ensure_state("show_help_doc", False)
    _ensure_state("preset_apply_notice", "")

    uploaded_file = st.file_uploader(
        "Upload CSV or XLSX (drag and drop)",
        type=["csv", "xlsx"],
        help=(
            "CSV 또는 XLSX 파일을 업로드합니다. 드래그&드롭 가능하며, "
            "업로드 후 컬럼을 자동 인식해 패널별 X/Y를 선택할 수 있습니다."
        ),
    )

    if uploaded_file is None:
        st.info("Upload a CSV/XLSX file to begin.")
        return

    try:
        source_df = load_uploaded_data(uploaded_file.getvalue(), uploaded_file.name)
    except Exception as exc:
        st.error(f"Failed to load uploaded file: {exc}")
        return

    if source_df.empty:
        st.error("The uploaded dataset has no rows.")
        return

    df = source_df.copy()
    df.columns = [str(c) for c in df.columns]
    columns = list(df.columns)

    if not columns:
        st.error("No columns were found in the uploaded dataset.")
        return

    helper_cols = st.columns([1.0, 3.0])
    with helper_cols[0]:
        if st.button("Help / Guide", key="toggle_help_doc"):
            st.session_state["show_help_doc"] = not st.session_state["show_help_doc"]
    with helper_cols[1]:
        st.caption(
            "Use preset JSON to reuse panel/style settings across files with similar columns."
        )

    if st.session_state["show_help_doc"]:
        with st.expander("User Guide", expanded=True):
            st.markdown(load_help_markdown())

    with st.expander("Preset (Load saved settings)", expanded=False):
        preset_file = st.file_uploader(
            "Load preset JSON",
            type=["json"],
            key="preset_loader",
            help=(
                "Load a previously saved settings preset. "
                "Current panel/style/label settings will be overwritten."
            ),
        )
        apply_preset = st.button(
            "Apply preset",
            key="apply_preset_button",
            disabled=(preset_file is None),
        )
        if apply_preset and preset_file is not None:
            try:
                preset_obj = json.loads(preset_file.getvalue().decode("utf-8"))
                if not isinstance(preset_obj, dict):
                    raise ValueError("Preset JSON root must be an object.")
                warnings, loaded_count = apply_loaded_preset_to_session(preset_obj, columns)
                notice = f"Preset applied successfully (panel_count={loaded_count})."
                if warnings:
                    notice += " Warnings: " + " | ".join(warnings)
                st.session_state["preset_apply_notice"] = notice
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to apply preset: {exc}")

    if st.session_state["preset_apply_notice"]:
        st.success(st.session_state["preset_apply_notice"])
        st.session_state["preset_apply_notice"] = ""

    st.subheader("Settings / Panel Selection")
    top_cols = st.columns([1.0, 1.0, 1.0, 1.4])

    with top_cols[0]:
        preview_n = int(
            st.number_input(
                "Preview rows",
                min_value=1,
                max_value=500,
                value=10,
                step=1,
                help=(
                    "데이터프레임 미리보기 행 수입니다. "
                    "분석에는 전체 데이터가 사용되고, 여기서는 확인용 상위 N행만 표시됩니다."
                ),
            )
        )

    id_options = ["<Use row index>"] + columns
    with top_cols[1]:
        id_choice = st.selectbox(
            "ID column",
            options=id_options,
            index=0,
            help=(
                "이상치 표에 표시할 식별자(ID) 컬럼을 선택합니다. "
                "선택하지 않으면 행 인덱스를 ID로 사용합니다."
            ),
        )

    with top_cols[2]:
        panel_count = int(
            st.number_input(
                "Panel count",
                min_value=1,
                max_value=12,
                value=int(st.session_state["panel_count"]),
                step=1,
                help=(
                    "생성할 패널(그래프) 개수입니다. "
                    "4개를 기본으로 하며, 늘리면 자동으로 2열/3열 그리드로 배치됩니다."
                ),
            )
        )
        st.session_state["panel_count"] = panel_count

    with top_cols[3]:
        robust_enabled = st.checkbox(
            "Use robust regression line (RLM)",
            value=False,
            help=(
                "이상치 영향이 큰 데이터에서 강건 회귀선(RLM)을 함께 사용합니다. "
                "구간(CI/PI)은 OLS 기반으로 유지됩니다."
            ),
        )
        robust_norm = st.radio(
            "Robust norm",
            options=["Huber", "Tukey"],
            index=0,
            horizontal=True,
            disabled=not robust_enabled,
            help=(
                "강건 회귀의 손실 함수 선택입니다. "
                "Huber는 완만하게, Tukey는 큰 이상치 가중치를 더 강하게 줄입니다."
            ),
        )

    with st.expander("Figure style defaults (adjustable)", expanded=True):
        st.caption(
            "논문 삽입 시 축소되어도 읽히도록 기본값을 크게 설정했습니다. "
            "필요하면 아래 값으로 더 키울 수 있습니다."
        )

        style_row1 = st.columns(4)
        with style_row1[0]:
            title_size = int(
                st.number_input(
                    "Title size",
                    min_value=10,
                    max_value=40,
                    value=int(st.session_state["style_title_size"]),
                    step=1,
                    key="style_title_size",
                    help="그래프 제목(패널 캡션) 글자 크기입니다.",
                )
            )
        with style_row1[1]:
            label_size = int(
                st.number_input(
                    "Axis label size",
                    min_value=10,
                    max_value=40,
                    value=int(st.session_state["style_label_size"]),
                    step=1,
                    key="style_label_size",
                    help="X/Y 축 제목(라벨) 글자 크기입니다.",
                )
            )
        with style_row1[2]:
            tick_size = int(
                st.number_input(
                    "Tick size",
                    min_value=8,
                    max_value=32,
                    value=int(st.session_state["style_tick_size"]),
                    step=1,
                    key="style_tick_size",
                    help="축 눈금 숫자 글자 크기입니다.",
                )
            )
        with style_row1[3]:
            legend_size = int(
                st.number_input(
                    "Legend size",
                    min_value=8,
                    max_value=32,
                    value=int(st.session_state["style_legend_size"]),
                    step=1,
                    key="style_legend_size",
                    help="범례(legend) 글자 크기입니다.",
                )
            )

        style_row2 = st.columns(4)
        with style_row2[0]:
            point_size = int(
                st.number_input(
                    "Point size",
                    min_value=8,
                    max_value=120,
                    value=int(st.session_state["style_point_size"]),
                    step=2,
                    key="style_point_size",
                    help="산점도 점(marker) 크기입니다.",
                )
            )
        with style_row2[1]:
            line_width = float(
                st.number_input(
                    "Regression line width",
                    min_value=1.0,
                    max_value=8.0,
                    value=float(st.session_state["style_line_width"]),
                    step=0.1,
                    key="style_line_width",
                    help="회귀선 두께입니다.",
                )
            )
        with style_row2[2]:
            panel_width = float(
                st.number_input(
                    "Panel width (inch)",
                    min_value=3.0,
                    max_value=12.0,
                    value=float(st.session_state["style_panel_width"]),
                    step=0.1,
                    key="style_panel_width",
                    help="개별 패널 가로 길이(인치)입니다.",
                )
            )
        with style_row2[3]:
            panel_height = float(
                st.number_input(
                    "Panel height (inch)",
                    min_value=3.0,
                    max_value=12.0,
                    value=float(st.session_state["style_panel_height"]),
                    step=0.1,
                    key="style_panel_height",
                    help="개별 패널 세로 길이(인치)입니다.",
                )
            )

    style_cfg = {
        "title_size": title_size,
        "label_size": label_size,
        "tick_size": tick_size,
        "legend_size": legend_size,
        "point_size": point_size,
        "line_width": line_width,
        "panel_width": panel_width,
        "panel_height": panel_height,
    }

    st.dataframe(df.head(preview_n), use_container_width=True)

    if id_choice == "<Use row index>":
        id_series = df.index.to_series().astype(str)
    else:
        id_series = df[id_choice].astype(str)

    panel_tabs = st.tabs([f"Panel {i + 1}" for i in range(panel_count)])
    panel_configs: list[dict[str, Any]] = []

    for i, tab in enumerate(panel_tabs):
        stored_cfg = st.session_state["panel_configs"].get(i, {})
        default_x = stored_cfg.get("x_col", columns[0])
        default_y = stored_cfg.get("y_col", columns[1] if len(columns) > 1 else columns[0])

        if default_x not in columns:
            default_x = columns[0]
        if default_y not in columns:
            default_y = columns[1] if len(columns) > 1 else columns[0]

        x_key = f"panel_{i}_x_col"
        y_key = f"panel_{i}_y_col"
        log_key = f"panel_{i}_log_y"
        caption_key = f"panel_{i}_caption"
        x_label_key = f"panel_{i}_x_label"
        y_label_key = f"panel_{i}_y_label"
        x_mode_key = f"panel_{i}_x_mode"
        y_mode_key = f"panel_{i}_y_mode"
        x_min_key = f"panel_{i}_x_min"
        x_max_key = f"panel_{i}_x_max"
        y_min_key = f"panel_{i}_y_min"
        y_max_key = f"panel_{i}_y_max"
        x_tick_key = f"panel_{i}_x_tick"
        y_tick_key = f"panel_{i}_y_tick"
        remove_key = f"panel_{i}_remove_outliers"
        show_ci_key = f"panel_{i}_show_ci"
        show_pi_key = f"panel_{i}_show_pi"
        show_legend_key = f"panel_{i}_show_legend"
        x_prev_key = f"panel_{i}_x_prev"
        y_prev_key = f"panel_{i}_y_prev"
        y_log_prev_key = f"panel_{i}_log_prev"

        _ensure_state(x_key, default_x)
        _ensure_state(y_key, default_y)
        _ensure_state(log_key, bool(stored_cfg.get("log_y", False)))
        _ensure_state(caption_key, str(stored_cfg.get("caption", "")))
        _ensure_state(x_mode_key, str(stored_cfg.get("x_mode", "auto")))
        _ensure_state(y_mode_key, str(stored_cfg.get("y_mode", "auto")))
        _ensure_state(x_min_key, float(stored_cfg.get("x_min", 0.0)))
        _ensure_state(x_max_key, float(stored_cfg.get("x_max", 1.0)))
        _ensure_state(y_min_key, float(stored_cfg.get("y_min", 0.0)))
        _ensure_state(y_max_key, float(stored_cfg.get("y_max", 1.0)))
        _ensure_state(x_tick_key, str(stored_cfg.get("x_tick", "")))
        _ensure_state(y_tick_key, str(stored_cfg.get("y_tick", "")))
        _ensure_state(remove_key, bool(stored_cfg.get("remove_outliers", False)))
        _ensure_state(show_ci_key, bool(stored_cfg.get("show_ci", True)))
        _ensure_state(show_pi_key, bool(stored_cfg.get("show_pi", True)))
        _ensure_state(show_legend_key, bool(stored_cfg.get("show_legend", True)))

        if st.session_state[x_key] not in columns:
            st.session_state[x_key] = default_x
        if st.session_state[y_key] not in columns:
            st.session_state[y_key] = default_y

        with tab:
            row1 = st.columns([1.0, 1.0, 1.0])
            with row1[0]:
                x_col = st.selectbox(
                    "X column",
                    columns,
                    key=x_key,
                    help="해당 패널의 독립변수(X) 컬럼을 선택합니다.",
                )
            with row1[1]:
                y_col = st.selectbox(
                    "Y column",
                    columns,
                    key=y_key,
                    help="해당 패널의 종속변수(Y) 컬럼을 선택합니다.",
                )
            with row1[2]:
                log_y = st.checkbox(
                    "Apply log10 to Y",
                    key=log_key,
                    help=(
                        "Y에 log10 변환을 적용합니다. "
                        "Y<=0 값은 자동 제외되며 경고로 안내됩니다."
                    ),
                )

            if st.session_state.get(x_prev_key) != x_col:
                st.session_state[x_label_key] = st.session_state["global_label_map"].get(
                    x_col, x_col
                )
                st.session_state[x_prev_key] = x_col

            y_template = st.session_state["global_label_map"].get(y_col, y_col)
            y_default = f"log10({y_template})" if log_y else y_template
            if (
                st.session_state.get(y_prev_key) != y_col
                or st.session_state.get(y_log_prev_key) != log_y
            ):
                st.session_state[y_label_key] = y_default
                st.session_state[y_prev_key] = y_col
                st.session_state[y_log_prev_key] = log_y

            _ensure_state(x_label_key, st.session_state["global_label_map"].get(x_col, x_col))
            _ensure_state(y_label_key, y_default)

            caption = st.text_input(
                "Panel caption (optional)",
                key=caption_key,
                placeholder="e.g., sediment rating curve",
                help=(
                    "패널 상단 제목입니다. "
                    "비워두면 제목을 표시하지 않습니다."
                ),
            )

            row2 = st.columns([1.0, 1.0])
            with row2[0]:
                x_label = st.text_input(
                    "X-axis label (LaTeX supported)",
                    key=x_label_key,
                    help=(
                        "X축 라벨을 직접 입력합니다. "
                        "예: $Q\\,(m^3/s)$"
                    ),
                )
            with row2[1]:
                y_label = st.text_input(
                    "Y-axis label (LaTeX supported)",
                    key=y_label_key,
                    help=(
                        "Y축 라벨을 직접 입력합니다. "
                        "예: $\\log_{10}(SSC)$"
                    ),
                )

            row3 = st.columns([1.0, 1.0])
            with row3[0]:
                x_mode = st.radio(
                    "X-axis range",
                    options=["auto", "manual"],
                    key=x_mode_key,
                    horizontal=True,
                    help="X축 표시 범위를 자동(auto) 또는 수동(manual)으로 설정합니다.",
                )
                if x_mode == "manual":
                    x_range_cols = st.columns(2)
                    with x_range_cols[0]:
                        x_min = float(
                            st.number_input(
                                "X min",
                                key=x_min_key,
                                help="수동 X축 최소값입니다. X min < X max 이어야 합니다.",
                            )
                        )
                    with x_range_cols[1]:
                        x_max = float(
                            st.number_input(
                                "X max",
                                key=x_max_key,
                                help="수동 X축 최대값입니다. X min < X max 이어야 합니다.",
                            )
                        )
                else:
                    x_min = st.session_state[x_min_key]
                    x_max = st.session_state[x_max_key]

            with row3[1]:
                y_mode = st.radio(
                    "Y-axis range",
                    options=["auto", "manual"],
                    key=y_mode_key,
                    horizontal=True,
                    help=(
                        "Y축 표시 범위를 자동(auto) 또는 수동(manual)으로 설정합니다. "
                        "log10 사용 시 변환된 Y 기준입니다."
                    ),
                )
                if y_mode == "manual":
                    y_range_cols = st.columns(2)
                    with y_range_cols[0]:
                        y_min = float(
                            st.number_input(
                                "Y min",
                                key=y_min_key,
                                help="수동 Y축 최소값입니다. Y min < Y max 이어야 합니다.",
                            )
                        )
                    with y_range_cols[1]:
                        y_max = float(
                            st.number_input(
                                "Y max",
                                key=y_max_key,
                                help="수동 Y축 최대값입니다. Y min < Y max 이어야 합니다.",
                            )
                        )
                else:
                    y_min = st.session_state[y_min_key]
                    y_max = st.session_state[y_max_key]

            row4 = st.columns([1.0, 1.0, 1.0])
            with row4[0]:
                x_tick = st.text_input(
                    "X tick interval (optional)",
                    key=x_tick_key,
                    placeholder="e.g., 10",
                    help=(
                        "X축 눈금 간격을 수동 지정합니다. "
                        "비워두면 자동, 입력 시 양수만 허용됩니다."
                    ),
                )
            with row4[1]:
                y_tick = st.text_input(
                    "Y tick interval (optional)",
                    key=y_tick_key,
                    placeholder="e.g., 0.5",
                    help=(
                        "Y축 눈금 간격을 수동 지정합니다. "
                        "비워두면 자동, 입력 시 양수만 허용됩니다."
                    ),
                )
            with row4[2]:
                remove_outliers = st.checkbox(
                    "Apply outlier removal",
                    key=remove_key,
                    help=(
                        "이상치 후보(|rstudent|>3 또는 Cook's D>4/n)를 제외한 뒤 "
                        "회귀를 다시 계산합니다."
                    ),
                )

            row5 = st.columns([1.0, 1.0, 1.0])
            with row5[0]:
                show_ci = st.checkbox(
                    "Show 95% CI",
                    key=show_ci_key,
                    help="Display the 95% confidence interval band for mean response.",
                )
            with row5[1]:
                show_pi = st.checkbox(
                    "Show 95% PI",
                    key=show_pi_key,
                    help="Display the 95% prediction interval band for individual observations.",
                )
            with row5[2]:
                show_legend = st.checkbox(
                    "Show legend",
                    key=show_legend_key,
                    help="Show or hide legend entries in this panel.",
                )

            save_template = st.button(
                "Save current labels as global templates",
                key=f"panel_{i}_save_label_template",
                help=(
                    "현재 패널의 축 라벨을 컬럼 템플릿으로 저장합니다. "
                    "같은 컬럼을 다른 패널에서 선택할 때 자동 반영됩니다."
                ),
            )
            if save_template:
                st.session_state["global_label_map"][x_col] = x_label
                if not log_y:
                    st.session_state["global_label_map"][y_col] = y_label
                st.success("Saved to global label templates.")

        panel_configs.append(
            {
                "x_col": x_col,
                "y_col": y_col,
                "log_y": bool(log_y),
                "caption": caption,
                "x_label": x_label,
                "y_label": y_label,
                "x_mode": x_mode,
                "x_min": x_min,
                "x_max": x_max,
                "y_mode": y_mode,
                "y_min": y_min,
                "y_max": y_max,
                "x_tick": x_tick,
                "y_tick": y_tick,
                "remove_outliers": bool(remove_outliers),
                "show_ci": bool(show_ci),
                "show_pi": bool(show_pi),
                "show_legend": bool(show_legend),
            }
        )

    st.session_state["panel_configs"] = {idx: cfg for idx, cfg in enumerate(panel_configs)}

    preset_payload = build_settings_preset_payload(
        panel_configs=panel_configs,
        style_cfg=style_cfg,
        global_label_map=st.session_state["global_label_map"],
    )
    preset_bytes = json.dumps(preset_payload, ensure_ascii=False, indent=2).encode("utf-8")
    preset_filename = f"{datetime.now().strftime('%Y%m%d')}_riverdata_settings_preset.json"

    with st.expander("Preset (Save current settings)", expanded=False):
        st.download_button(
            label="Download current preset JSON",
            data=preset_bytes,
            file_name=preset_filename,
            mime="application/json",
            help=(
                "Save current panel/style/label settings as JSON. "
                "You can reload this preset later for another dataset."
            ),
        )

    plot_cols = 2 if panel_count <= 4 else 3
    plot_rows = int(math.ceil(panel_count / plot_cols))
    fig, axes = plt.subplots(
        plot_rows,
        plot_cols,
        figsize=(plot_cols * style_cfg["panel_width"], plot_rows * style_cfg["panel_height"]),
    )
    flat_axes = np.array(axes, dtype=object).reshape(-1)

    panel_results: list[dict[str, Any]] = []

    for idx, panel_cfg in enumerate(panel_configs):
        panel_warnings: list[str] = []

        fallback_x_label = panel_cfg["x_col"]
        fallback_y_label = (
            f"log10({panel_cfg['y_col']})" if panel_cfg["log_y"] else panel_cfg["y_col"]
        )

        sanitized_cfg = panel_cfg.copy()
        sanitized_cfg["x_label"] = safe_mathtext(
            panel_cfg["x_label"],
            fallback=fallback_x_label,
            warn_bucket=panel_warnings,
            context=f"Panel {idx + 1} X label",
        )
        sanitized_cfg["y_label"] = safe_mathtext(
            panel_cfg["y_label"],
            fallback=fallback_y_label,
            warn_bucket=panel_warnings,
            context=f"Panel {idx + 1} Y label",
        )
        sanitized_cfg["panel_title"] = make_panel_label(idx, panel_cfg["caption"])
        panel_signature = (
            f"panel_{idx}|x={panel_cfg['x_col']}|y={panel_cfg['y_col']}|log={int(panel_cfg['log_y'])}"
        )

        fit_df, counts, prep_warnings = prepare_panel_data(
            df=df,
            id_series=id_series,
            x_col=panel_cfg["x_col"],
            y_col=panel_cfg["y_col"],
            log_y=bool(panel_cfg["log_y"]),
        )
        panel_warnings.extend(prep_warnings)

        if len(fit_df) < 3:
            panel_result = {
                "status": "insufficient",
                "message": (
                    "Insufficient numeric data after filtering "
                    f"(n={len(fit_df)}; minimum 3 required)."
                ),
                "warnings": panel_warnings,
                "prep_counts": counts,
                "outlier_table": pd.DataFrame(
                    columns=[
                        "ID",
                        "X",
                        "Y",
                        "rstudent",
                        "cooks_d",
                        "leverage",
                        "outside_95_ci",
                        "outside_95_pi",
                    ]
                ),
                "full_diag_table": pd.DataFrame(
                    columns=[
                        "row_key",
                        "ID",
                        "X",
                        "Y",
                        "rstudent",
                        "cooks_d",
                        "leverage",
                        "is_outlier_candidate",
                        "outside_95_ci",
                        "outside_95_pi",
                        "manual_review",
                    ]
                ),
                "outlier_count": 0,
                "outside_ci_count": 0,
                "outside_pi_count": 0,
                "n": int(len(fit_df)),
                "cook_threshold": float("nan"),
                "compare_df": None,
                "removal_applied": False,
                "removed_count": 0,
                "plot_cfg": sanitized_cfg,
                "style_cfg": style_cfg,
                "panel_signature": panel_signature,
            }
            panel_results.append(panel_result)
            render_panel(flat_axes[idx], sanitized_cfg, panel_result)
            continue

        try:
            fit_before = fit_ols_and_optional_robust(
                fit_df["X"].to_numpy(),
                fit_df["Y"].to_numpy(),
                robust_enabled=robust_enabled,
                robust_norm=robust_norm,
            )

            if fit_before["robust_error"] is not None:
                panel_warnings.append(
                    "Robust fit failed; OLS line is used instead. "
                    f"Reason: {fit_before['robust_error']}"
                )

            influence_df = compute_influence_table(fit_df, fit_before["ols_model"])

            point_design = sm.add_constant(fit_df["X"].to_numpy(), has_constant="add")
            point_sf = fit_before["ols_model"].get_prediction(point_design).summary_frame(alpha=0.05)
            influence_df["outside_95_ci"] = (
                (fit_df["Y"].to_numpy() < point_sf["mean_ci_lower"].to_numpy())
                | (fit_df["Y"].to_numpy() > point_sf["mean_ci_upper"].to_numpy())
            )
            influence_df["outside_95_pi"] = (
                (fit_df["Y"].to_numpy() < point_sf["obs_ci_lower"].to_numpy())
                | (fit_df["Y"].to_numpy() > point_sf["obs_ci_upper"].to_numpy())
            )

            manual_map = st.session_state["manual_review_flags"].get(panel_signature, {})
            influence_df["manual_review"] = (
                influence_df["row_key"].astype(str).map(manual_map).fillna(False).astype(bool)
            )

            outlier_table = influence_df.loc[
                influence_df["is_outlier_candidate"],
                [
                    "ID",
                    "X",
                    "Y",
                    "rstudent",
                    "cooks_d",
                    "leverage",
                    "outside_95_ci",
                    "outside_95_pi",
                ],
            ].copy()

            fit_after = fit_before
            plot_df = fit_df.copy()
            removal_applied = False
            removed_count = int(len(outlier_table))
            compare_df: pd.DataFrame | None = None

            if panel_cfg["remove_outliers"]:
                if removed_count == 0:
                    panel_warnings.append(
                        "Outlier removal enabled, but no candidates matched thresholds."
                    )
                else:
                    keep_mask = ~influence_df["is_outlier_candidate"].to_numpy()
                    filtered_df = fit_df.loc[keep_mask].reset_index(drop=True)
                    if len(filtered_df) < 3:
                        panel_warnings.append(
                            "Outlier removal left fewer than 3 rows. Original fit retained."
                        )
                    else:
                        fit_after = fit_ols_and_optional_robust(
                            filtered_df["X"].to_numpy(),
                            filtered_df["Y"].to_numpy(),
                            robust_enabled=robust_enabled,
                            robust_norm=robust_norm,
                        )
                        if fit_after["robust_error"] is not None:
                            panel_warnings.append(
                                "Robust refit after outlier removal failed; OLS line is used. "
                                f"Reason: {fit_after['robust_error']}"
                            )
                        plot_df = filtered_df
                        removal_applied = True

                compare_df = _comparison_frame(fit_before, fit_after)

            if len(plot_df) < 8:
                panel_warnings.append(
                    "n < 8: confidence/prediction intervals and influence diagnostics may be unstable."
                )

            x_min = float(plot_df["X"].min())
            x_max = float(plot_df["X"].max())
            if math.isclose(x_min, x_max):
                x_grid = np.linspace(x_min - 0.5, x_max + 0.5, 200)
            else:
                x_grid = np.linspace(x_min, x_max, 200)

            # CI/PI are OLS-based even when robust line is selected.
            # Future extension: bootstrap robust coefficients/residuals for robust bands.
            pred_df = compute_prediction_frames(fit_after["ols_model"], x_grid)

            line_params = np.asarray(fit_after["line_params"], dtype=float)
            line_y = line_params[0] + line_params[1] * x_grid
            line_df = pd.DataFrame({"X": x_grid, "Y": line_y})

            panel_result = {
                "status": "ok",
                "warnings": panel_warnings,
                "prep_counts": counts,
                "plot_df": plot_df,
                "pred_df": pred_df,
                "line_df": line_df,
                "line_label": (
                    f"Robust regression ({robust_norm})"
                    if fit_after["used_robust"]
                    else "OLS regression"
                ),
                "outlier_table": outlier_table,
                "full_diag_table": influence_df.copy(),
                "outlier_count": int(len(outlier_table)),
                "outside_ci_count": int(influence_df["outside_95_ci"].sum()),
                "outside_pi_count": int(influence_df["outside_95_pi"].sum()),
                "n": int(len(fit_df)),
                "cook_threshold": float(4.0 / len(fit_df)),
                "compare_df": compare_df,
                "removal_applied": removal_applied,
                "removed_count": removed_count,
                "plot_cfg": sanitized_cfg,
                "style_cfg": style_cfg,
                "panel_signature": panel_signature,
            }

        except Exception as exc:
            panel_result = {
                "status": "error",
                "message": f"Model fitting failed: {exc}",
                "warnings": panel_warnings,
                "prep_counts": counts,
                "outlier_table": pd.DataFrame(
                    columns=[
                        "ID",
                        "X",
                        "Y",
                        "rstudent",
                        "cooks_d",
                        "leverage",
                        "outside_95_ci",
                        "outside_95_pi",
                    ]
                ),
                "full_diag_table": pd.DataFrame(
                    columns=[
                        "row_key",
                        "ID",
                        "X",
                        "Y",
                        "rstudent",
                        "cooks_d",
                        "leverage",
                        "is_outlier_candidate",
                        "outside_95_ci",
                        "outside_95_pi",
                        "manual_review",
                    ]
                ),
                "outlier_count": 0,
                "outside_ci_count": 0,
                "outside_pi_count": 0,
                "n": int(len(fit_df)),
                "cook_threshold": float("nan"),
                "compare_df": None,
                "removal_applied": False,
                "removed_count": 0,
                "plot_cfg": sanitized_cfg,
                "style_cfg": style_cfg,
                "panel_signature": panel_signature,
            }

        panel_results.append(panel_result)
        render_panel(flat_axes[idx], sanitized_cfg, panel_result)

    for idx in range(panel_count, len(flat_axes)):
        flat_axes[idx].axis("off")

    fig.tight_layout()

    st.subheader("Multipanel Regression Figure")
    st.pyplot(fig, use_container_width=True)

    st.subheader("Outlier Diagnostics (Table/Text only)")
    st.caption("Outlier candidates are not annotated on the plot by design.")

    for idx, result in enumerate(panel_results):
        title = make_panel_label(idx, panel_configs[idx]["caption"])
        display_title = title if title else f"Panel {idx + 1}"
        with st.expander(
            f"Panel {idx + 1} diagnostics: {display_title}",
            expanded=(idx == 0),
        ):
            for warning in result["warnings"]:
                st.warning(warning)

            counts = result["prep_counts"]
            st.write(
                "Preprocessing summary: "
                f"input={counts['input_rows']}, "
                f"dropped_xy={counts['dropped_xy_nan']}, "
                f"dropped_log={counts['dropped_log_nonpositive']}, "
                f"remaining={counts['remaining_rows']}"
            )

            if result["status"] != "ok":
                st.info(result.get("message", "No diagnostics available for this panel."))
                continue

            st.write(
                f"n={result['n']} | thresholds: "
                f"|rstudent|>{_RSTUDENT_THRESHOLD:.1f}, "
                f"Cook's D>{result['cook_threshold']:.4g} | "
                f"candidates={result['outlier_count']} | "
                f"outside_95_CI={result['outside_ci_count']} | "
                f"outside_95_PI={result['outside_pi_count']}"
            )

            outlier_table = result["outlier_table"].copy()
            if outlier_table.empty:
                st.info("No outlier candidates found.")
            else:
                st.dataframe(outlier_table, use_container_width=True)

            st.write("Full diagnostics table (all rows)")
            full_diag_table = result["full_diag_table"].copy()
            if full_diag_table.empty:
                st.info("No full diagnostics rows available.")
            else:
                panel_signature = result["panel_signature"]
                manual_map = st.session_state["manual_review_flags"].get(panel_signature, {})
                full_diag_table["manual_review"] = (
                    full_diag_table["row_key"].astype(str).map(manual_map).fillna(False).astype(bool)
                )

                numeric_cols = ["X", "Y", "rstudent", "cooks_d", "leverage"]
                for col in numeric_cols:
                    if col in full_diag_table.columns:
                        full_diag_table[col] = pd.to_numeric(
                            full_diag_table[col], errors="coerce"
                        ).round(6)

                editor_df = full_diag_table[
                    [
                        "row_key",
                        "ID",
                        "X",
                        "Y",
                        "rstudent",
                        "cooks_d",
                        "leverage",
                        "is_outlier_candidate",
                        "outside_95_ci",
                        "outside_95_pi",
                        "manual_review",
                    ]
                ].copy()

                edited_df = st.data_editor(
                    editor_df,
                    key=f"panel_{idx}_full_diag_editor",
                    hide_index=True,
                    use_container_width=True,
                    disabled=[col for col in editor_df.columns if col != "manual_review"],
                    column_config={
                        "row_key": None,
                        "is_outlier_candidate": st.column_config.CheckboxColumn(
                            "is_outlier_candidate",
                            help="Automatic candidate by |rstudent|>3 or Cook's D>4/n.",
                        ),
                        "outside_95_ci": st.column_config.CheckboxColumn(
                            "outside_95_ci",
                            help="Observation lies outside the 95% confidence interval (mean response).",
                        ),
                        "outside_95_pi": st.column_config.CheckboxColumn(
                            "outside_95_pi",
                            help="Observation lies outside the 95% prediction interval.",
                        ),
                        "manual_review": st.column_config.CheckboxColumn(
                            "manual_review",
                            help="Analyst manual review flag (does not auto-remove rows).",
                        ),
                    },
                )

                st.session_state["manual_review_flags"][panel_signature] = {
                    str(row["row_key"]): bool(row["manual_review"])
                    for _, row in edited_df[["row_key", "manual_review"]].iterrows()
                }

                manual_checked_count = int(edited_df["manual_review"].sum())
                st.caption(
                    "Manual review checked rows: "
                    f"{manual_checked_count} / {len(edited_df)} "
                    "(manual review flags do not change automatic outlier removal)."
                )

            if panel_configs[idx]["remove_outliers"]:
                st.write("Outlier removal comparison (before vs after)")
                if result["compare_df"] is not None:
                    st.dataframe(result["compare_df"], use_container_width=True, hide_index=True)
                st.caption(
                    "Removal applied: "
                    f"{'Yes' if result['removal_applied'] else 'No'} "
                    f"(candidate rows={result['removed_count']})"
                )

    st.subheader("Export")
    st.caption("합성(Grid) Figure와 패널별 Figure를 각각 다운로드할 수 있습니다.")

    export_top_cols = st.columns([1.2, 1.0])
    with export_top_cols[0]:
        export_label = st.selectbox(
            "Format",
            options=["PNG", "PDF", "SVG"],
            index=0,
            help="내보낼 파일 포맷을 선택합니다.",
        )
    with export_top_cols[1]:
        png_dpi = int(
            st.number_input(
                "PNG DPI",
                min_value=300,
                max_value=1200,
                value=300,
                step=50,
                disabled=(export_label != "PNG"),
                help=(
                    "PNG 해상도(DPI)입니다. PNG 선택 시 적용됩니다. "
                    "고해상도 인쇄가 필요하면 600~1200을 사용하세요."
                ),
            )
        )

    export_format = export_label.lower()

    st.markdown("**Grid figure (all panels)**")
    filename_key = f"export_grid_filename_{export_format}"
    if filename_key not in st.session_state:
        st.session_state[filename_key] = _default_export_filename(
            panel_configs, export_format
        )

    filename = st.text_input(
        "Grid filename",
        key=filename_key,
        help="전체 패널이 합쳐진 Figure 파일명입니다.",
    )
    filename = (
        filename.strip()
        if filename
        else _default_export_filename(panel_configs, export_format)
    )
    if not filename.lower().endswith(f".{export_format}"):
        filename = f"{filename}.{export_format}"
    filename = _sanitize_filename(filename.rsplit(".", 1)[0]) + f".{export_format}"

    export_bytes, mime = export_figure_bytes(fig, export_format, png_dpi=png_dpi)
    st.download_button(
        label="Download grid figure",
        data=export_bytes,
        file_name=filename,
        mime=mime,
        key=f"download_grid_{export_format}_{png_dpi}",
    )

    st.markdown("**Individual panel figures**")
    panel_download_cols = 2 if panel_count <= 4 else 3
    panel_dl_columns = st.columns(panel_download_cols)

    for idx, (panel_cfg, panel_result) in enumerate(zip(panel_configs, panel_results)):
        panel_filename_key = f"export_panel_{idx}_filename_{export_format}"
        default_panel_name = _default_panel_export_filename(idx, panel_cfg, export_format)
        if panel_filename_key not in st.session_state:
            st.session_state[panel_filename_key] = default_panel_name

        single_panel_cfg = panel_result.get("plot_cfg", panel_cfg)
        panel_fig = create_single_panel_figure(single_panel_cfg, panel_result)
        panel_bytes, panel_mime = export_figure_bytes(
            panel_fig,
            export_format,
            png_dpi=png_dpi,
        )
        plt.close(panel_fig)

        with panel_dl_columns[idx % panel_download_cols]:
            panel_filename = st.text_input(
                f"Panel {idx + 1} filename",
                key=panel_filename_key,
                help="개별 패널 Figure 파일명입니다.",
            )
            panel_filename = panel_filename.strip() if panel_filename else default_panel_name
            if not panel_filename.lower().endswith(f".{export_format}"):
                panel_filename = f"{panel_filename}.{export_format}"
            panel_filename = (
                _sanitize_filename(panel_filename.rsplit(".", 1)[0]) + f".{export_format}"
            )

            st.download_button(
                label=f"Download Panel {idx + 1}",
                data=panel_bytes,
                file_name=panel_filename,
                mime=panel_mime,
                key=f"download_panel_{idx}_{export_format}_{png_dpi}",
            )

    plt.close(fig)


if __name__ == "__main__":
    main()

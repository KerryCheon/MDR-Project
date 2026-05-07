# Utility for the notebook
# Jakob Balkovec

# desc: prints a neat dashboard of metrics for regression tasks,
#       with a focus on interpretability and quick insights.
#       It includes standard metrics like R², MAE, RMSE,
#       as well as bias and a custom ubrmse metric.
#       The dashboard is designed to be visually appealing
#       and informative, making it easy to assess model performance at a glance.

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from IPython.display import display, HTML

def _ubrmse(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    yt = y_true - np.mean(y_true)
    yp = y_pred - np.mean(y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))

def _safe_corr(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])

def metrics_dashboard(y_true, y_pred, name="Model Evaluation", prefix="", show=True, return_dict=False, compact=True, show_quantiles=True):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    err = y_true - y_pred
    ae = np.abs(err)

    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    ubrmse = float(_ubrmse(y_true, y_pred))
    bias = float(np.mean(err))
    corr = float(_safe_corr(y_true, y_pred))
    q_err = np.quantile(err, [0.05, 0.25, 0.50, 0.75, 0.95])
    med_ae = float(np.median(ae))

    out = {"name": name, "n": len(y_true), "r2": r2, "mae": mae, "rmse": rmse, "bias": bias}
    if not show: return out if return_dict else None

    bg_main = "#1e293b"
    bg_inner = "rgba(15, 23, 42, 0.4)"
    border = "rgba(255, 255, 255, 0.1)"
    text_main = "#f8fafc"
    text_dim = "#94a3b8"

    def get_color(val, metric_type="pos"):
        if metric_type == "pos":
            if val >= 0.8: return "#22c55e"
            if val >= 0.6: return "#eab308"
            return "#ef4444"
        else:
            if abs(val) < 0.05: return "#22c55e"
            if abs(val) < 0.15: return "#eab308"
            return "#ef4444"

    def _micro_bar(label, val, color):
        pct = max(0, min(100, val * 100)) if "R" in label or "Corr" in label else 50
        return f"""
        <div style="flex: 1; min-width: 180px; margin-bottom: 8px;">
            <div style="display: flex; justify-content: space-between; font-family: monospace; font-size: 11px; margin-bottom: 4px;">
                <span style="color:{text_dim}">{label}</span>
                <span style="color:{color}; font-weight: bold;">{val:.3f}</span>
            </div>
            <div style="height: 4px; background: rgba(255,255,255,0.1); border-radius: 2px; overflow: hidden;">
                <div style="width: {pct}%; height: 100%; background: {color}; box-shadow: 0 0 8px {color}66;"></div>
            </div>
        </div>
        """

    header_html = f"""
    <div style="background: linear-gradient(145deg, {bg_main}, #0f172a); border: 1px solid {border}; border-radius: 16px; padding: 24px; color: {text_main}; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica;">
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 20px; border-bottom: 1px solid {border}; padding-bottom: 12px;">
            <div style="font-size: 20px; font-weight: 800; letter-spacing: -0.5px;">{name} <span style="font-size: 12px; font-weight: 400; color: {text_dim}; margin-left: 8px;">{prefix}</span></div>
            <div style="font-family: monospace; color: {text_dim}; font-size: 12px;">N = {len(y_true):,}</div>
        </div>

        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
            <div><div style="color: {text_dim}; font-size: 10px; font-weight: 700; text-transform: uppercase;">R-Squared</div><div style="font-size: 24px; font-weight: 900; color: {get_color(r2, 'pos')}">{r2:.4f}</div></div>
            <div><div style="color: {text_dim}; font-size: 10px; font-weight: 700; text-transform: uppercase;">RMSE</div><div style="font-size: 24px; font-weight: 900;">{rmse:.4f}</div></div>
            <div><div style="color: {text_dim}; font-size: 10px; font-weight: 700; text-transform: uppercase;">ubRMSE</div><div style="font-size: 24px; font-weight: 900;">{ubrmse:.4f}</div></div>
            <div><div style="color: {text_dim}; font-size: 10px; font-weight: 700; text-transform: uppercase;">Bias</div><div style="font-size: 24px; font-weight: 900; color: {get_color(bias, 'bias')}">{bias:+.4f}</div></div>
        </div>

        <div style="margin-top: 25px; display: flex; flex-wrap: wrap; gap: 20px; background: {bg_inner}; padding: 15px; border-radius: 12px;">
            {_micro_bar("R² Accuracy", r2, get_color(r2, 'pos'))}
            {_micro_bar("Pearson Corr", corr, get_color(corr, 'pos'))}
            {_micro_bar("Residual Bias", 1 - abs(bias), get_color(bias, 'bias'))}
        </div>

        {f'''
        <div style="margin-top: 15px; padding: 10px 5px; font-size: 11px; font-family: ui-monospace, monospace; border-top: 1px solid {border};">
            <span style="color: {text_dim}; font-weight: bold;">ERR QUANTILES [5/25/50/75/95]:</span>
            <span style="color: #60a5fa; margin-left: 8px;">{" &nbsp; ".join([f"{x:+.3f}" for x in q_err])}</span>
            <span style="margin-left: 20px; color: {text_dim};">REF (MED|ERR|): {med_ae:.4f}</span>
        </div>
        ''' if show_quantiles else ""}

        <div style="margin-top: 10px; display: grid; grid-template-columns: repeat(3, 1fr); font-size: 11px; font-family: monospace; color: {text_dim}; opacity: 0.8;">
            <div>MAE: <span style="color: {text_main}">{mae:.5f}</span></div>
            <div>MED |ERR|: <span style="color: {text_main}">{med_ae:.5f}</span></div>
            <div>STD(ERR): <span style="color: {text_main}">{np.std(err):.5f}</span></div>
        </div>
    </div>
    """

    display(HTML(header_html))
    return out if return_dict else None

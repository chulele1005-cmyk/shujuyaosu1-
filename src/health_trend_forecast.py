# -*- coding: utf-8 -*-
"""
==============================================================================
人群健康趋势分析与异常预警 — 任务二 (4.2)
功能: 时序预测 + 异常检测 + 趋势变点识别
数据: OWID COVID China / WHO China NCD 时序
==============================================================================
"""

import os, sys, json, warnings
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.random.seed(42)

OUTPUT_DIR = Path("./outputs")
(OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "report").mkdir(parents=True, exist_ok=True)

log_lines = []

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}")
    log_lines.append(f"[{ts}] {msg}")


# ======================================================================
# Step 1: 加载时序数据
# ======================================================================
def load_timeseries_data():
    log("加载时序数据...")

    # 尝试1: OWID中国COVID数据
    owid_china = Path("./data/anhui/owid_china/owid_covid_china.parquet")
    owid_raw = Path("./data/raw/covid_public_health/owid_covid_compact.csv")

    if owid_china.exists():
        df = pd.read_parquet(owid_china)
        log(f"OWID China COVID: {df.shape}")
    elif owid_raw.exists():
        df_raw = pd.read_csv(owid_raw, low_memory=False)
        df = df_raw[df_raw['country'].astype(str).str.contains('China', case=False, na=False)].copy()
        log(f"OWID China (from raw): {df.shape}")
    else:
        log("无中国时序数据, 生成演示数据", "WARN")
        return generate_demo_data()

    # 找日期和指标列
    date_col = 'date' if 'date' in df.columns else df.columns[0]
    val_cols = [c for c in df.columns if any(k in c.lower() for k in
                ['new_cases', 'total_cases', 'new_deaths', 'confirmed'])]

    if val_cols:
        val_col = val_cols[0]
    else:
        val_col = df.select_dtypes(include=[np.number]).columns[-1]

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col, val_col])
    df = df.sort_values(date_col)
    df = df.reset_index(drop=True)

    log(f"时序数据: {len(df)}行, date={date_col}, value={val_col}")
    log(f"时间范围: {df[date_col].min()} ~ {df[date_col].max()}")
    return df, date_col, val_col


def generate_demo_data():
    """生成中国区域健康趋势演示数据"""
    log("生成中国安徽省健康趋势演示数据...")
    # 模拟安徽省月度慢性病发病率数据
    dates = pd.date_range("2018-01-01", periods=96, freq="ME")
    # 趋势 + 季节性 + 噪声
    trend = np.linspace(100, 115, 96)  # 缓慢上升趋势
    seasonal = 8 * np.sin(np.arange(96) * 2 * np.pi / 12)  # 年周期
    noise = np.random.randn(96) * 3
    values = trend + seasonal + noise

    df = pd.DataFrame({'date': dates, 'incidence_rate': values.clip(min=0)})
    log(f"演示数据: {len(df)}行, 2018-2025年月度慢病发病率")
    return df, 'date', 'incidence_rate'


# ======================================================================
# Step 2: 移动平均趋势预测
# ======================================================================
def forecast_with_ma(df, date_col, val_col, horizon=12):
    """简单移动平均 + 趋势外推预测"""
    log(f"\n--- 趋势预测 (MA + Holt-Winters风格) ---")

    series = df[val_col].values
    dates = df[date_col].values

    # 计算移动平均
    ma_3m = pd.Series(series).rolling(3).mean()
    ma_12m = pd.Series(series).rolling(12).mean()

    # 简单线性外推 (基于最近12个月的趋势)
    recent = series[-12:]
    x = np.arange(12)
    slope, intercept = np.polyfit(x, recent, 1)
    future_x = np.arange(12, 12 + horizon)
    forecast_values = slope * future_x + intercept
    forecast_values = np.maximum(forecast_values, 0)

    # 计算置信区间 (基于历史残差)
    residuals = recent - (slope * x + intercept)
    std_resid = np.std(residuals)
    upper = forecast_values + 1.96 * std_resid
    lower = np.maximum(forecast_values - 1.96 * std_resid, 0)

    log(f"  趋势斜率: {slope:.3f}/月")
    log(f"  预测未来{horizon}个月, 范围: [{forecast_values[0]:.1f} ~ {forecast_values[-1]:.1f}]")

    # 绘图
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, series, color='#3498db', alpha=0.6, linewidth=0.8, label='Historical')
    ax.plot(dates, ma_12m, color='#2c3e50', linewidth=2, label='12-Month MA')
    future_dates = pd.date_range(dates[-1], periods=horizon+1, freq='ME')[1:]
    ax.plot(future_dates, forecast_values, color='#e74c3c', linewidth=2,
            linestyle='--', label=f'Forecast ({horizon} months)')
    ax.fill_between(future_dates, lower, upper, alpha=0.15, color='#e74c3c',
                     label='95% CI')
    ax.set_title('Health Trend Forecast — Anhui Province (Simulated)', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Incidence Rate')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "trend_forecast.png", dpi=150, bbox_inches='tight')
    plt.close()
    log("趋势预测图: outputs/figures/trend_forecast.png")

    return forecast_values, slope


# ======================================================================
# Step 3: CUSUM 异常检测
# ======================================================================
def cusum_anomaly_detection(df, val_col, threshold=3.0):
    """CUSUM 异常检测"""
    log(f"\n--- CUSUM 异常检测 (threshold={threshold}σ) ---")

    series = df[val_col].dropna().values
    n = len(series)
    if n < 10:
        return []

    mean, std = np.mean(series), np.std(series)
    if std == 0:
        return []

    cusum_pos = np.zeros(n)
    cusum_neg = np.zeros(n)
    anomalies = []
    for i in range(1, n):
        cusum_pos[i] = max(0, cusum_pos[i-1] + (series[i] - mean) / std - 0.5)
        cusum_neg[i] = min(0, cusum_neg[i-1] + (series[i] - mean) / std + 0.5)
        if cusum_pos[i] > threshold:
            anomalies.append({"index": i, "type": "upward_spike", "value": float(series[i])})
        if cusum_neg[i] < -threshold:
            anomalies.append({"index": i, "type": "downward_drop", "value": float(series[i])})

    log(f"  检测到 {len(anomalies)} 个异常点")

    # 绘图
    dates = df['date'].values if 'date' in df.columns else np.arange(n)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    axes[0].plot(dates, series, alpha=0.7, linewidth=0.8, color='#3498db')
    # 标记异常点
    for a in anomalies:
        color = '#e74c3c' if a['type'] == 'upward_spike' else '#e67e22'
        axes[0].scatter(dates[a['index']], a['value'], color=color, s=50, zorder=5)
    axes[0].set_title('Time Series with Anomalies')

    axes[1].plot(cusum_pos, color='#e74c3c', label='CUSUM+')
    axes[1].axhline(y=threshold, color='red', linestyle='--', alpha=0.5)
    axes[1].set_title('CUSUM Positive')

    axes[2].plot(cusum_neg, color='#2ecc71', label='CUSUM-')
    axes[2].axhline(y=-threshold, color='green', linestyle='--', alpha=0.5)
    axes[2].set_title('CUSUM Negative')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "cusum_anomalies.png", dpi=150, bbox_inches='tight')
    plt.close()
    log("CUSUM异常检测图: outputs/figures/cusum_anomalies.png")

    return anomalies


# ======================================================================
# Step 4: 趋势分解
# ======================================================================
def decompose_trend(df, date_col, val_col):
    """简单趋势分解: 趋势 + 季节 + 残差"""
    log("\n--- 趋势分解 ---")
    series = df[val_col].values
    dates = df[date_col].values

    # 12月移动平均作为趋势
    trend = pd.Series(series).rolling(12, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
    detrended = series - trend

    # 季节性 (按月平均)
    months = pd.to_datetime(dates).month.values
    seasonal = np.zeros(len(series))
    for m in range(1, 13):
        mask = months == m
        if mask.sum() > 0:
            seasonal[mask] = detrended[mask].mean()
    residual = detrended - seasonal

    # 绘图
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(dates, series, color='#3498db', linewidth=0.8)
    axes[0].set_title('Original')
    axes[1].plot(dates, trend, color='#2c3e50', linewidth=2)
    axes[1].set_title('Trend (12-month MA)')
    axes[2].plot(dates, seasonal, color='#2ecc71', linewidth=0.8)
    axes[2].set_title('Seasonal (Monthly)')
    axes[3].plot(dates, residual, color='#e74c3c', linewidth=0.5)
    axes[3].set_title('Residual')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "trend_decomposition.png", dpi=150, bbox_inches='tight')
    plt.close()
    log("趋势分解图: outputs/figures/trend_decomposition.png")


# ======================================================================
# Step 5: 报告
# ======================================================================
def generate_report(df, anomalies, slope, horizon):
    n_anomalies = len(anomalies)
    up = sum(1 for a in anomalies if a['type'] == 'upward_spike')
    down = sum(1 for a in anomalies if a['type'] == 'downward_drop')

    report = f"""# 人群健康趋势分析与异常预警报告

**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**任务:** 4.2 人群健康趋势分析与异常预警

## 数据概览

| 指标 | 数值 |
|------|------|
| 数据点数 | {len(df)} |
| 异常点总数 | {n_anomalies} |
| 向上异常 | {up} |
| 向下异常 | {down} |
| 趋势斜率 | {slope:.4f}/月 |
| 预测期 | {horizon} 个月 |

## 分析结论

本分析基于中国区域健康趋势数据，使用移动平均平滑 + 线性外推法预测未来趋势，
结合 CUSUM 算法检测异常波动。趋势分解展示了长期趋势、季节性波动和随机残差三个分量。

## 输出文件

- outputs/figures/trend_forecast.png — 趋势预测图
- outputs/figures/cusum_anomalies.png — CUSUM异常检测图
- outputs/figures/trend_decomposition.png — 趋势分解图
"""
    (OUTPUT_DIR / "report" / "health_trend_report.md").write_text(report, encoding="utf-8")
    log("报告: outputs/report/health_trend_report.md")


# ======================================================================
# Main
# ======================================================================
def main():
    log("╔══════════════════════════════════════════════════════════╗")
    log("║   任务4.2: 人群健康趋势分析与异常预警                   ║")
    log("╚══════════════════════════════════════════════════════════╝")

    df, date_col, val_col = load_timeseries_data()
    forecast, slope = forecast_with_ma(df, date_col, val_col)
    anomalies = cusum_anomaly_detection(df, val_col)
    decompose_trend(df, date_col, val_col)
    generate_report(df, anomalies, slope, 12)

    log("\n✅ 趋势分析与异常预警完成!")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)
    main()

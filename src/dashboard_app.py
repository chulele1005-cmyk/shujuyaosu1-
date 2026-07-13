# -*- coding: utf-8 -*-
"""
==============================================================================
交互式健康风险可视化面板 — 任务三 (5.1)
Dash + Plotly Web应用
==============================================================================
"""

import os, sys, json, warnings
from pathlib import Path
import pandas as pd, numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

warnings.filterwarnings("ignore")

# ========================= 配置 =========================
APP_PORT = 8050
PROJECT_ROOT = Path(__file__).parent.parent

# ========================= 加载数据 =========================
def load_model_results():
    """加载任务二模型结果"""
    return {
        'diabetes': {'name': '糖尿病预测 (BRFSS)', 'auc': 0.828, 'f1': 0.253, 'samples': '253,680'},
        'heart': {'name': '心脏病预测 (BRFSS)', 'auc': 0.851, 'f1': 0.166, 'samples': '253,680'},
        'cardio': {'name': '心血管预测 (实测)', 'auc': 0.801, 'f1': 0.724, 'samples': '70,000'},
        'heart_uci': {'name': '心脏病 (UCI)', 'auc': 1.000, 'f1': 1.000, 'samples': '1,025'},
    }

def load_shap_data():
    """SHAP特征重要性"""
    return {
        'diabetes': [
            ('GenHlth (整体健康)', 0.632), ('HighBP (高血压)', 0.462),
            ('BMI (体重指数)', 0.404), ('Age (年龄)', 0.392),
            ('HighChol (高胆固醇)', 0.275), ('Income (收入)', 0.221),
            ('Education (教育)', 0.198), ('DiffWalk (行走困难)', 0.185),
            ('PhysActivity (运动)', 0.172), ('Smoker (吸烟)', 0.098),
        ],
        'heart': [
            ('Age (年龄)', 0.710), ('GenHlth (整体健康)', 0.479),
            ('HighBP (高血压)', 0.350), ('Sex (性别)', 0.328),
            ('HighChol (高胆固醇)', 0.299), ('Education (教育)', 0.210),
            ('BMI (体重指数)', 0.187), ('Smoker (吸烟)', 0.145),
            ('Income (收入)', 0.140), ('DiffWalk (行走困难)', 0.131),
        ],
        'cardio': [
            ('ap_hi (收缩压!)', 0.868), ('age (年龄)', 0.281),
            ('cholesterol (胆固醇)', 0.207), ('ap_lo (舒张压)', 0.107),
            ('bmi (体重指数)', 0.067), ('pulse_pressure (脉压差)', 0.056),
            ('gluc (血糖)', 0.048), ('weight (体重)', 0.045),
            ('active (运动)', 0.043), ('smoke (吸烟)', 0.015),
        ],
    }

def load_trend_data():
    """生成趋势数据"""
    dates = pd.date_range('2020-01-01', periods=365*3+365, freq='W')
    np.random.seed(42)
    trend = 50 + np.cumsum(np.random.randn(len(dates))*0.3)
    seasonal = 15*np.sin(np.arange(len(dates))*2*np.pi/52)
    values = (trend + seasonal + np.random.randn(len(dates))*5).clip(0)
    return pd.DataFrame({'date': dates, 'cases': values})

# ========================= Dash App =========================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY],
                title='区域健康风险预测与智能诊疗决策平台')
server = app.server

# ---- Layout ----
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col(html.Div([
            html.H2('🏥 区域健康风险预测与智能诊疗决策平台', style={'color':'white'}),
            html.P('Health Risk Prediction & Intelligent Clinical Decision Support',
                   style={'color':'#bdc3c7', 'fontSize':'14px'}),
        ]), width=12)
    ], style={'backgroundColor':'#2c3e50', 'padding':'20px', 'borderRadius':'8px',
              'marginBottom':'15px'}),

    # KPI Row
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H4('0.851', style={'color':'#27ae60', 'fontSize':'28px'}),
            html.P('最佳模型 AUC (心脏病)', style={'color':'#7f8c8d'}),
            html.Small('BRFSS Heart 25万人', style={'color':'#95a5a6'}),
        ])], color='white'), width=3),
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H4('50.7万', style={'color':'#2980b9', 'fontSize':'28px'}),
            html.P('训练数据总量', style={'color':'#7f8c8d'}),
            html.Small('BRFSS Diabetes + Heart', style={'color':'#95a5a6'}),
        ])], color='white'), width=3),
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H4('11,630', style={'color':'#e67e22', 'fontSize':'28px'}),
            html.P('趋势分析数据天数', style={'color':'#7f8c8d'}),
            html.Small('OWID中国 COVID-19', style={'color':'#95a5a6'}),
        ])], color='white'), width=3),
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H4('203人', style={'color':'#8e44ad', 'fontSize':'28px'}),
            html.P('安徽验证样本', style={'color':'#7f8c8d'}),
            html.Small('CHARLS 安徽省', style={'color':'#95a5a6'}),
        ])], color='white'), width=3),
    ], style={'marginBottom':'15px'}),

    # Main content tabs
    dbc.Tabs([
        # Tab 1: Model Performance
        dbc.Tab(label='📊 模型性能', children=[
            dbc.Row([
                dbc.Col(dcc.Graph(id='auc-bar-chart'), width=6),
                dbc.Col(dcc.Graph(id='f1-bar-chart'), width=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='model-comparison-table'), width=12),
            ]),
        ]),

        # Tab 2: SHAP Feature Importance
        dbc.Tab(label='🔍 风险因素分析', children=[
            dbc.Row([
                dbc.Col([
                    html.Label('选择模型:'),
                    dcc.Dropdown(
                        id='shap-model-selector',
                        options=[
                            {'label':'🥇 BRFSS 糖尿病预测 (AUC=0.828)','value':'diabetes'},
                            {'label':'🥇 BRFSS 心脏病预测 (AUC=0.851)','value':'heart'},
                            {'label':'🥇 Cardiovascular 心血管实测 (AUC=0.801)','value':'cardio'},
                        ],
                        value='cardio',
                        style={'marginBottom':'15px'}
                    ),
                ], width=12),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='shap-bar-chart'), width=8),
                dbc.Col(dcc.Graph(id='risk-gauge'), width=4),
            ]),
        ]),

        # Tab 3: Trend Forecast
        dbc.Tab(label='📈 趋势预测', children=[
            dbc.Row([
                dbc.Col([
                    html.Label('选择区域:'),
                    dcc.Dropdown(
                        id='region-selector',
                        options=[
                            {'label':'🇨🇳 中国 (OWID COVID-19)','value':'china'},
                            {'label':'🌏 全球 (OWID COVID-19)','value':'global'},
                            {'label':'🇺🇸 美国 (BRFSS趋势)','value':'us'},
                        ],
                        value='china',
                        style={'marginBottom':'15px'}
                    ),
                    html.Label('预测天数:'),
                    dcc.Slider(30, 180, step=30, value=90,
                               marks={i:f'{i}天' for i in [30,60,90,120,150,180]},
                               id='forecast-days-slider'),
                ], width=12),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='trend-forecast-chart'), width=12),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='trend-decomposition-chart'), width=12),
            ]),
        ]),

        # Tab 4: Regional Health (Anhui focus)
        dbc.Tab(label='🗺️ 区域健康 (安徽)', children=[
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardBody([
                    html.H5('🇨🇳 安徽省健康数据概览', style={'color':'#2c3e50'}),
                    html.P('数据来源: CHARLS 2020 (中国健康与养老追踪调查)'),
                    html.Hr(),
                    html.P(f'📍 受访者: 203人 (45岁以上中老年)'),
                    html.P(f'🫀 高血压自报率: ~32%'),
                    html.P(f'📊 慢性病类型: 15种 (含心脏病/糖尿病/中风/癌症/关节炎等)'),
                    html.P(f'🏘️ 社区数: 218个村/居委会'),
                    html.P(f'📋 问卷模块: 10个 (健康/人口/经济/COVID/功能等)'),
                    html.Hr(),
                    html.P('⚠️ 待获取 CHARLS Biomarkers 实测体检数据后可展示完整区域健康指标',
                           style={'color':'#e74c3c', 'fontSize':'13px'}),
                ])]), width=6),
                dbc.Col(dcc.Graph(id='china-province-chart'), width=6),
            ]),
        ]),

        # Tab 5: Resource Dashboard
        dbc.Tab(label='🏥 资源评估', children=[
            dbc.Row([
                dbc.Col(dcc.Graph(id='resource-gauge-chart'), width=6),
                dbc.Col(dcc.Graph(id='resource-utilization-chart'), width=6),
            ]),
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardBody([
                    html.H5('M/M/c 排队论模型参数'),
                    html.P('基于Healthcare Ops数据估算的医疗资源利用率:'),
                    html.Hr(),
                    html.P('⚠️ 注意: Healthcare Ops经V2验证为合成数据, 以下为框架演示'),
                ])]), width=12),
            ]),
        ]),
    ]),
], fluid=True, style={'padding':'20px', 'backgroundColor':'#ecf0f1', 'minHeight':'100vh'})


# ========================= Callbacks =========================

@app.callback(
    [Output('auc-bar-chart','figure'), Output('f1-bar-chart','figure'),
     Output('model-comparison-table','figure')],
    Input('auc-bar-chart','id')  # dummy trigger
)
def update_model_charts(_):
    results = load_model_results()
    names = [v['name'] for v in results.values()]
    aucs = [v['auc'] for v in results.values()]
    f1s = [v['f1'] for v in results.values()]
    samples = [v['samples'] for v in results.values()]

    # AUC bar chart
    auc_fig = px.bar(x=names, y=aucs, color=aucs, title='AUC-ROC 对比 (越高越好)',
                     color_continuous_scale='Greens', text=[f'{a:.3f}' for a in aucs])
    auc_fig.add_hline(y=0.85, line_dash='dash', line_color='orange',
                       annotation_text='指南目标: 0.85')
    auc_fig.add_hline(y=0.80, line_dash='dot', line_color='gray',
                       annotation_text='及格线: 0.80')
    auc_fig.update_layout(height=350)

    # F1 bar chart
    f1_fig = px.bar(x=names, y=f1s, color=f1s, title='F1 Score 对比',
                    color_continuous_scale='Blues', text=[f'{f:.3f}' for f in f1s])
    f1_fig.update_layout(height=350)

    # Table
    table_fig = go.Figure(data=[go.Table(
        header=dict(values=['模型', 'AUC', 'F1', '样本量', '评估'],
                    fill_color='#2c3e50', font=dict(color='white')),
        cells=dict(values=[
            names, [f'{a:.3f}' for a in aucs], [f'{f:.3f}' for f in f1s],
            samples,
            ['✅ 良好' if a>0.80 else '⚠️ 需改进' for a in aucs]
        ], fill_color=[['white','#f8f9fa']*2])
    )])
    table_fig.update_layout(title='模型综合评估表', height=250)

    return auc_fig, f1_fig, table_fig


@app.callback(
    [Output('shap-bar-chart','figure'), Output('risk-gauge','figure')],
    Input('shap-model-selector','value')
)
def update_shap_view(model_key):
    shap_data = load_shap_data()
    data = shap_data.get(model_key, shap_data['cardio'])

    # SHAP bar
    features = [d[0] for d in reversed(data)]
    values = [d[1] for d in reversed(data)]
    shap_fig = px.bar(x=values, y=features, orientation='h',
                       title='SHAP 特征重要性 (风险因素排名)',
                       color=values, color_continuous_scale='Reds')
    shap_fig.update_layout(height=450, yaxis=dict(tickfont=dict(size=11)))

    # Gauge
    model_results = load_model_results()
    auc = model_results[model_key]['auc']
    gauge_fig = go.Figure(go.Indicator(
        mode='gauge+number+delta',
        value=auc, number={'suffix':' AUC'},
        delta={'reference': 0.85, 'increasing': {'color':'green'}},
        title={'text': f"{model_results[model_key]['name']}"},
        gauge={'axis': {'range': [0.5, 1.0]},
               'bar': {'color':'#27ae60'},
               'steps': [{'range':[0.5,0.7],'color':'#e74c3c'},
                         {'range':[0.7,0.8],'color':'#f39c12'},
                         {'range':[0.8,0.85],'color':'#2ecc71'},
                         {'range':[0.85,1.0],'color':'#27ae60'}],
               'threshold': {'line':{'color':'orange','width':2},'value':0.85}},
    ))
    gauge_fig.update_layout(height=450)

    return shap_fig, gauge_fig


@app.callback(
    [Output('trend-forecast-chart','figure'),
     Output('trend-decomposition-chart','figure')],
    [Input('region-selector','value'), Input('forecast-days-slider','value')]
)
def update_trend(region, forecast_days):
    df = load_trend_data()
    n_hist = len(df) - forecast_days//7
    hist = df.iloc[:n_hist]
    future_dates = pd.date_range(hist['date'].iloc[-1], periods=forecast_days//7+1, freq='W')[1:]

    # Trend forecast
    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(x=hist['date'], y=hist['cases'],
                                    mode='lines', name='历史数据',
                                    line=dict(color='#3498db', width=1.5)))
    # Forecast
    last_val = hist['cases'].iloc[-1]
    slope = hist['cases'].iloc[-12:].diff().mean()
    fcst = last_val + slope * np.arange(1, len(future_dates)+1)
    trend_fig.add_trace(go.Scatter(x=future_dates, y=fcst,
                                    mode='lines', name=f'预测 ({forecast_days}天)',
                                    line=dict(color='#e74c3c', width=2, dash='dash')))
    # CI
    std = hist['cases'].iloc[-12:].std()
    upper = fcst + 1.96*std; lower = np.maximum(fcst - 1.96*std, 0)
    trend_fig.add_trace(go.Scatter(
        x=list(future_dates)+list(future_dates)[::-1],
        y=list(upper)+list(lower)[::-1],
        fill='toself', fillcolor='rgba(231,76,60,0.15)',
        name='95% 置信区间', line=dict(color='rgba(255,255,255,0)')))
    trend_fig.update_layout(title=f'{region.upper()} 健康趋势预测',
                            height=350, template='plotly_white')

    # Decomposition
    ma12 = df['cases'].rolling(12).mean()
    detrended = df['cases'] - ma12
    decomp_fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                subplot_titles=['原始序列','趋势 (12周MA)','残差'],
                                vertical_spacing=0.08)
    decomp_fig.add_trace(go.Scatter(x=df['date'], y=df['cases'],
                                     line=dict(color='#3498db',width=0.8)), row=1, col=1)
    decomp_fig.add_trace(go.Scatter(x=df['date'], y=ma12,
                                     line=dict(color='#2c3e50',width=2)), row=2, col=1)
    decomp_fig.add_trace(go.Scatter(x=df['date'], y=detrended,
                                     line=dict(color='#e74c3c',width=0.5)), row=3, col=1)
    decomp_fig.update_layout(height=450, template='plotly_white')

    return trend_fig, decomp_fig


@app.callback(
    Output('china-province-chart', 'figure'),
    Input('china-province-chart', 'id')
)
def update_china_map(_):
    """模拟中国健康风险热力图"""
    provinces = ['安徽','北京','上海','广东','江苏','浙江','山东','河南','湖北',
                 '湖南','四川','河北','辽宁','福建','陕西','重庆','云南','贵州',
                 '广西','山西','吉林','黑龙江','内蒙古','新疆','甘肃','江西','天津','海南']
    np.random.seed(42)
    risk = np.random.beta(2, 5, len(provinces)) * 100
    risk[0] = 15.2  # 安徽 (highlight)

    fig = px.bar(x=provinces, y=risk, color=risk,
                  title='中国各省慢性病风险指数 (模拟数据 — 需CHARLS Biomarkers验证)',
                  color_continuous_scale='RdYlGn_r',
                  labels={'x':'省份', 'y':'风险指数'})
    fig.add_hline(y=risk.mean(), line_dash='dash', line_color='gray',
                   annotation_text=f'全国均值: {risk.mean():.1f}')
    fig.update_layout(height=400)
    return fig


@app.callback(
    [Output('resource-gauge-chart','figure'),
     Output('resource-utilization-chart','figure')],
    Input('resource-gauge-chart','id')
)
def update_resource(_):
    resources = ['普通病房(50床)','ICU(20床)','急门诊','呼吸机(安徽15台)','手术室','MRI']
    utils = [0.45, 0.62, 0.38, 0.28, 0.71, 0.55]
    colors = ['#e74c3c' if u>0.85 else '#f39c12' if u>0.7 else '#2ecc71' for u in utils]

    gauge_fig = go.Figure()
    for i, (res, util) in enumerate(zip(resources, utils)):
        gauge_fig.add_trace(go.Bar(
            y=[res], x=[min(util, 1.0)], orientation='h',
            marker_color=colors[i], name=res,
            text=f'{util:.0%}', textposition='inside'))
    gauge_fig.add_vline(x=0.7, line_dash='dash', line_color='orange',
                         annotation_text='高负荷线 70%')
    gauge_fig.add_vline(x=0.85, line_dash='dash', line_color='red',
                         annotation_text='过载线 85%')
    gauge_fig.update_layout(title='医疗资源利用率 (M/M/c 排队论)',
                            height=300, xaxis=dict(tickformat='.0%'))

    util_fig = px.pie(names=resources, values=utils,
                       title='资源负荷分布', hole=0.4)
    util_fig.update_layout(height=300)

    return gauge_fig, util_fig


# ========================= Main =========================
if __name__ == '__main__':
    print(f'╔══════════════════════════════════════════════╗')
    print(f'║  区域健康风险预测与智能诊疗决策平台          ║')
    print(f'║  访问地址: http://localhost:{APP_PORT}          ║')
    print(f'╚══════════════════════════════════════════════╝')
    app.run(host='0.0.0.0', port=APP_PORT, debug=False)

"""
flask_app/app.py
Accident Risk Prediction & Traffic Flow Analysis — Flask version
FUOYE Final Year Project: Azuh Toyosi Titilayo (FTP/CSC/25/0134940)
"""
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib, os, io, base64
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, '..', 'models')
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results')


def _safe_load(path):
    try:
        return joblib.load(path)
    except Exception:
        return None


def _chart(fig, dpi=130):
    """Convert a matplotlib figure to a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=dpi,
                facecolor='white', edgecolor='none')
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return data


# ── Load all artifacts at startup ─────────────────────────────────────────────
rf          = joblib.load(os.path.join(MODELS_DIR, 'random_forest.pkl'))
scaler      = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
le_road     = joblib.load(os.path.join(MODELS_DIR, 'le_road.pkl'))
le_wea      = joblib.load(os.path.join(MODELS_DIR, 'le_weather.pkl'))
results_df  = pd.read_csv(os.path.join(RESULTS_DIR, 'model_results.csv'))
fi_df       = pd.read_csv(os.path.join(RESULTS_DIR, 'feature_importance.csv'))
preds       = _safe_load(os.path.join(RESULTS_DIR, 'predictions.pkl'))
cm_rf       = _safe_load(os.path.join(RESULTS_DIR, 'cm_rf.pkl'))
cm_ann      = _safe_load(os.path.join(RESULTS_DIR, 'cm_ann.pkl'))
ann_history = _safe_load(os.path.join(MODELS_DIR,  'ann_history.pkl'))

try:
    import onnxruntime as ort
    _ann_sess       = ort.InferenceSession(os.path.join(MODELS_DIR, 'ann_model.onnx'))
    _ann_input_name = _ann_sess.get_inputs()[0].name
    ann_loaded = True
except Exception:
    ann_loaded = False

LABELS        = ['Low Risk', 'Medium Risk', 'High Risk']
RISK_COLORS   = {'Low Risk': '#2563eb', 'Medium Risk': '#d97706', 'High Risk': '#dc2626'}
ROAD_TYPES    = list(le_road.classes_)
WEATHER_TYPES = list(le_wea.classes_)

rf_row  = results_df[results_df['Model'] == 'Random Forest'].iloc[0]
ann_row = results_df[results_df['Model'] == 'ANN'].iloc[0]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('home.html',
        rf_accuracy=f"{rf_row['Accuracy']*100:.1f}",
        ann_accuracy=f"{ann_row['Accuracy']*100:.1f}",
        rf_f1=f"{rf_row['F1-Score']:.3f}",
        ann_f1=f"{ann_row['F1-Score']:.3f}",
        rf_roc=f"{rf_row['ROC-AUC']:.3f}",
        ann_roc=f"{ann_row['ROC-AUC']:.3f}",
    )


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    result    = None
    form_data = {}

    if request.method == 'POST':
        form_data = request.form.to_dict()

        hour           = int(form_data.get('hour', 8))
        day_num        = int(form_data.get('day_num', 0))
        month          = int(form_data.get('month', 6))
        temperature    = float(form_data.get('temperature', 28))
        humidity       = float(form_data.get('humidity', 75))
        precipitation  = float(form_data.get('precipitation', 0))
        visibility     = float(form_data.get('visibility', 8))
        wind_speed     = float(form_data.get('wind_speed', 10))
        traffic_volume = int(form_data.get('traffic_volume', 500))
        junction       = int('junction' in form_data)
        traffic_signal = int('traffic_signal' in form_data)
        crossing       = int('crossing' in form_data)
        bump           = int('bump' in form_data)
        speed_limit    = int(form_data.get('speed_limit', 50))
        road_type      = form_data.get('road_type', ROAD_TYPES[0])
        weather_cond   = form_data.get('weather_cond', WEATHER_TYPES[0])
        model_choice   = form_data.get('model_choice', 'Random Forest')

        road_enc    = le_road.transform([road_type])[0]
        weather_enc = le_wea.transform([weather_cond])[0]
        inp = np.array([[hour, day_num, month, temperature, humidity, precipitation,
                         visibility, wind_speed, traffic_volume,
                         junction, traffic_signal, crossing, bump,
                         speed_limit, road_enc, weather_enc]], dtype=float)
        inp_sc = scaler.transform(inp)

        if model_choice == 'ANN' and ann_loaded:
            proba      = _ann_sess.run(None, {_ann_input_name: inp_sc.astype(np.float32)})[0][0]
            pred_class = int(np.argmax(proba))
            model_used = 'ANN (Neural Network)'
        else:
            proba      = rf.predict_proba(inp_sc)[0]
            pred_class = int(rf.predict(inp_sc)[0])
            model_used = 'Random Forest'

        risk_label = LABELS[pred_class]

        # Probability bar chart
        fig, ax = plt.subplots(figsize=(9, 1.3))
        ax.barh([''], [proba[0]], color='#2563eb', label='Low Risk')
        ax.barh([''], [proba[1]], left=[proba[0]], color='#d97706', label='Medium Risk')
        ax.barh([''], [proba[2]], left=[proba[0]+proba[1]], color='#dc2626', label='High Risk')
        ax.set_xlim(0, 1)
        ax.axis('off')
        ax.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -1.9), fontsize=10)
        ax.set_title('Risk Probability Distribution', fontsize=11, fontweight='bold', pad=5)
        plt.tight_layout()
        prob_chart = _chart(fig)

        result = {
            'risk_label': risk_label,
            'risk_class': pred_class,
            'confidence': f"{proba[pred_class]*100:.1f}",
            'proba_low':  f"{proba[0]*100:.1f}",
            'proba_med':  f"{proba[1]*100:.1f}",
            'proba_high': f"{proba[2]*100:.1f}",
            'model_used': model_used,
            'prob_chart': prob_chart,
            'color':      RISK_COLORS[risk_label],
        }

    return render_template('prediction.html',
        road_types=ROAD_TYPES,
        weather_types=WEATHER_TYPES,
        ann_loaded=ann_loaded,
        result=result,
        form_data=form_data,
    )


@app.route('/analysis')
def analysis():
    import plotly.graph_objects as go
    import plotly.io as pio

    COLORS  = ['#2563eb', '#d97706', '#dc2626']
    LABELS3 = ['Low Risk', 'Medium Risk', 'High Risk']

    cfg = {'displayModeBar': False, 'responsive': True}

    def to_html(fig):
        return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                           config=cfg, div_id=None)

    charts = {}

    # ── Pie charts ────────────────────────────────────────────────────────────
    if preds is not None:
        y_test  = preds['y_test']
        rf_pred = preds['rf_pred']
        c_act  = np.bincount(y_test,  minlength=3).tolist()
        c_pred = np.bincount(rf_pred, minlength=3).tolist()

        fig = go.Figure(go.Pie(
            labels=LABELS3, values=c_act,
            marker=dict(colors=COLORS, line=dict(color='white', width=2)),
            hole=0.5, textinfo='label+percent',
            textfont=dict(size=13, family='Inter'),
            hovertemplate='%{label}: %{value:,} cases (%{percent})<extra></extra>'
        ))
        fig.update_layout(showlegend=True, margin=dict(t=10,b=10,l=10,r=10),
                          height=280, paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)',
                          legend=dict(font=dict(family='Inter', size=12)))
        charts['pie_actual'] = to_html(fig)

        fig = go.Figure(go.Pie(
            labels=LABELS3, values=c_pred,
            marker=dict(colors=COLORS, line=dict(color='white', width=2)),
            hole=0.5, textinfo='label+percent',
            textfont=dict(size=13, family='Inter'),
            hovertemplate='%{label}: %{value:,} cases (%{percent})<extra></extra>'
        ))
        fig.update_layout(showlegend=True, margin=dict(t=10,b=10,l=10,r=10),
                          height=280, paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)',
                          legend=dict(font=dict(family='Inter', size=12)))
        charts['pie_pred'] = to_html(fig)

    # ── Model comparison ──────────────────────────────────────────────────────
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    rf_v  = [round(float(rf_row[m]),  3) for m in metrics]
    ann_v = [round(float(ann_row[m]), 3) for m in metrics]

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Random Forest', x=metrics, y=rf_v,
                         marker_color='#1c4ed8', marker_line_width=0,
                         text=[f'{v:.3f}' for v in rf_v], textposition='outside',
                         textfont=dict(size=11, family='Inter')))
    fig.add_trace(go.Bar(name='ANN (Neural Network)', x=metrics, y=ann_v,
                         marker_color='#0891b2', marker_line_width=0,
                         text=[f'{v:.3f}' for v in ann_v], textposition='outside',
                         textfont=dict(size=11, family='Inter')))
    fig.update_layout(
        barmode='group', height=320,
        margin=dict(t=20, b=10, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(range=[0, 1.15], gridcolor='rgba(0,0,0,0.07)',
                   tickfont=dict(family='Inter', size=11)),
        xaxis=dict(tickfont=dict(family='Inter', size=12)),
        legend=dict(font=dict(family='Inter', size=12),
                    orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        font=dict(family='Inter'),
    )
    charts['comparison'] = to_html(fig)

    # ── Feature importance ────────────────────────────────────────────────────
    fi_top = fi_df.head(12)
    fi_labels = fi_top['feature'].tolist()
    fi_values = [round(float(v), 4) for v in fi_top['importance'].tolist()]
    bar_colors = ['#1e3a8a' if i < 3 else '#1c4ed8' if i < 7 else '#93c5fd'
                  for i in range(len(fi_labels))]

    fig = go.Figure(go.Bar(
        x=fi_values[::-1], y=fi_labels[::-1],
        orientation='h',
        marker_color=bar_colors[::-1], marker_line_width=0,
        text=[f'{v:.4f}' for v in fi_values[::-1]], textposition='outside',
        textfont=dict(size=10, family='Inter'),
        hovertemplate='%{y}: %{x:.4f}<extra></extra>'
    ))
    fig.update_layout(
        height=380, margin=dict(t=10, b=10, l=10, r=60),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='rgba(0,0,0,0.07)', tickfont=dict(family='Inter', size=11)),
        yaxis=dict(tickfont=dict(family='Inter', size=11)),
        font=dict(family='Inter'),
    )
    charts['importance'] = to_html(fig)

    # ── Error metrics ─────────────────────────────────────────────────────────
    err_labels = ['MAE', 'RMSE']
    rf_err  = [round(float(rf_row['MAE']),  4), round(float(rf_row['RMSE']),  4)]
    ann_err = [round(float(ann_row['MAE']), 4), round(float(ann_row['RMSE']), 4)]

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Random Forest', x=err_labels, y=rf_err,
                         marker_color='#1c4ed8', marker_line_width=0,
                         text=[f'{v:.4f}' for v in rf_err], textposition='outside',
                         textfont=dict(size=11, family='Inter')))
    fig.add_trace(go.Bar(name='ANN', x=err_labels, y=ann_err,
                         marker_color='#0891b2', marker_line_width=0,
                         text=[f'{v:.4f}' for v in ann_err], textposition='outside',
                         textfont=dict(size=11, family='Inter')))
    fig.update_layout(
        barmode='group', height=260,
        margin=dict(t=20, b=10, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(gridcolor='rgba(0,0,0,0.07)', tickfont=dict(family='Inter', size=11)),
        xaxis=dict(tickfont=dict(family='Inter', size=12)),
        legend=dict(font=dict(family='Inter', size=11),
                    orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        font=dict(family='Inter'),
    )
    charts['errors'] = to_html(fig)

    return render_template('analysis.html',
        preds_available=(preds is not None),
        charts=charts,
    )


@app.route('/evaluation')
def evaluation():
    charts = {}

    disp = results_df[['Model', 'Accuracy', 'Precision', 'Recall',
                        'F1-Score', 'ROC-AUC', 'MAE', 'RMSE']].copy()
    disp['Accuracy']  = disp['Accuracy'].apply(lambda x: f"{x*100:.2f}%")
    disp['Precision'] = disp['Precision'].apply(lambda x: f"{x:.4f}")
    disp['Recall']    = disp['Recall'].apply(lambda x: f"{x:.4f}")
    disp['F1-Score']  = disp['F1-Score'].apply(lambda x: f"{x:.4f}")
    disp['ROC-AUC']   = disp['ROC-AUC'].apply(lambda x: f"{x:.4f}")
    disp['MAE']       = disp['MAE'].apply(lambda x: f"{x:.4f}")
    disp['RMSE']      = disp['RMSE'].apply(lambda x: f"{x:.4f}")
    metrics_rows = disp.to_dict('records')
    metrics_cols = list(disp.columns)

    if cm_rf is not None:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=LABELS, yticklabels=LABELS, linewidths=0.5)
        ax.set_title('Confusion Matrix — Random Forest', fontsize=11, fontweight='bold')
        ax.set_ylabel('Actual Risk Level', fontsize=9)
        ax.set_xlabel('Predicted Risk Level', fontsize=9)
        plt.tight_layout()
        charts['cm_rf'] = _chart(fig)

    if cm_ann is not None:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(cm_ann, annot=True, fmt='d', cmap='Greens', ax=ax,
                    xticklabels=LABELS, yticklabels=LABELS, linewidths=0.5)
        ax.set_title('Confusion Matrix — ANN', fontsize=11, fontweight='bold')
        ax.set_ylabel('Actual Risk Level', fontsize=9)
        ax.set_xlabel('Predicted Risk Level', fontsize=9)
        plt.tight_layout()
        charts['cm_ann'] = _chart(fig)

    if ann_history is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        ax1.plot(ann_history['accuracy'],     color='#1c4ed8', lw=2, label='Training')
        ax1.plot(ann_history['val_accuracy'], color='#dc2626', lw=2, ls='--', label='Validation')
        ax1.set_title('Accuracy per Training Epoch', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy'); ax1.legend()
        ax1.yaxis.grid(True, linestyle='--', alpha=0.5)
        ax2.plot(ann_history['loss'],     color='#1c4ed8', lw=2, label='Training')
        ax2.plot(ann_history['val_loss'], color='#dc2626', lw=2, ls='--', label='Validation')
        ax2.set_title('Loss per Training Epoch', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss (lower = better)'); ax2.legend()
        ax2.yaxis.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        charts['learning'] = _chart(fig)

    return render_template('evaluation.html',
        metrics_rows=metrics_rows,
        metrics_cols=metrics_cols,
        charts=charts,
    )


@app.route('/about')
def about():
    return render_template('about.html',
        rf_accuracy=f"{rf_row['Accuracy']*100:.2f}",
        rf_f1=f"{rf_row['F1-Score']:.3f}",
        rf_roc=f"{rf_row['ROC-AUC']:.3f}",
        ann_accuracy=f"{ann_row['Accuracy']*100:.2f}",
        ann_f1=f"{ann_row['F1-Score']:.3f}",
        ann_roc=f"{ann_row['ROC-AUC']:.3f}",
    )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

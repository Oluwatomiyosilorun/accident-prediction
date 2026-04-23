"""
app.py — Accident Risk Prediction & Traffic Flow Analysis
FUOYE Final Year Project: Azuh Toyosi Titilayo (FTP/CSC/25/0134940)
Run with:  streamlit run app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import joblib, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Accident Risk Prediction — FUOYE",
                   page_icon="🚦", layout="wide",
                   initial_sidebar_state="expanded")

MODELS_DIR  = os.path.join(os.path.dirname(__file__), 'models')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

@st.cache_resource
def load_artifacts():
    rf      = joblib.load(f'{MODELS_DIR}/random_forest.pkl')
    scaler  = joblib.load(f'{MODELS_DIR}/scaler.pkl')
    le_road = joblib.load(f'{MODELS_DIR}/le_road.pkl')
    le_wea  = joblib.load(f'{MODELS_DIR}/le_weather.pkl')
    feats   = joblib.load(f'{MODELS_DIR}/feature_names.pkl')
    results = pd.read_csv(f'{RESULTS_DIR}/model_results.csv')
    fi      = pd.read_csv(f'{RESULTS_DIR}/feature_importance.csv')
    preds   = joblib.load(f'{RESULTS_DIR}/predictions.pkl')
    cm_rf   = joblib.load(f'{RESULTS_DIR}/cm_rf.pkl')
    cm_ann  = joblib.load(f'{RESULTS_DIR}/cm_ann.pkl')
    history = joblib.load(f'{MODELS_DIR}/ann_history.pkl')
    return rf, scaler, le_road, le_wea, feats, results, fi, preds, cm_rf, cm_ann, history

try:
    import tensorflow as tf; tf.get_logger().setLevel('ERROR')
    from tensorflow.keras.models import load_model
    ann = load_model(f'{MODELS_DIR}/ann_model.keras', compile=False)
    ann_loaded = True
except Exception:
    ann_loaded = False

rf, scaler, le_road, le_wea, FEATURES, results_df, fi_df, preds, cm_rf, cm_ann, ann_history = load_artifacts()

LABELS      = ['Low Risk', 'Medium Risk', 'High Risk']
RISK_COLORS = {'Low Risk': '#2E75B6', 'Medium Risk': '#FFC000', 'High Risk': '#C00000'}
RISK_EMOJI  = {'Low Risk': '🟢', 'Medium Risk': '🟡', 'High Risk': '🔴'}
ROAD_TYPES    = list(le_road.classes_)
WEATHER_TYPES = list(le_wea.classes_)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚦 Traffic Risk System")
    st.markdown("**FUOYE Final Year Project**")
    st.markdown("Azuh Toyosi Titilayo")
    st.markdown("`FTP/CSC/25/0134940`")
    st.divider()
    page = st.radio("Navigate", [
        "🏠 Home",
        "🔮 Risk Prediction",
        "📈 Traffic Flow Analysis",
        "📊 Model Evaluation",
        "ℹ️ About"])
    st.divider()
    st.caption("Supervisor: Mr. Onadokun")
    st.caption("Dept. of Computer Science, FUOYE — 2025")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🚦 Accident Risk Prediction & Traffic Flow Analysis")
    st.markdown("#### An Intelligent Traffic Management System for Nigerian Urban Roads")
    st.divider()

    st.markdown("""
    > **Nigeria records over 13,000 road traffic crashes and 5,000 fatalities every year.**
    > The majority of these accidents are preventable — they occur under predictable conditions
    > such as heavy rainfall, peak-hour congestion, and high-speed corridors.
    > Yet traditional traffic management relies on fixed signal timers that cannot adapt,
    > and officers must respond to incidents *after* they happen.
    >
    > **This system changes that.** By analysing road conditions, weather, and traffic data
    > in real time, it predicts accident risk *before* incidents occur — giving traffic
    > controllers the intelligence to act proactively.
    """)

    st.divider()

    # ── How it works ──
    st.markdown("### How the System Works")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        **Step 1 — Enter Conditions**

        A traffic officer inputs the current road and weather situation: time of day, weather,
        traffic volume, road type, and infrastructure details.
        """)
    with col2:
        st.markdown("""
        **Step 2 — AI Analysis**

        The system passes the inputs through a trained machine learning model
        (Random Forest or ANN) that learned patterns from 15,000 traffic records.
        """)
    with col3:
        st.markdown("""
        **Step 3 — Risk Level Output**

        A risk level is returned — **Low**, **Medium**, or **High** — along with
        a confidence score and probability breakdown.
        """)
    with col4:
        st.markdown("""
        **Step 4 — Recommended Action**

        The system suggests an appropriate response: standard monitoring,
        increased patrol, or immediate deployment of officers.
        """)

    st.divider()

    # ── System performance summary ──
    st.markdown("### System Performance at a Glance")
    rf_row  = results_df[results_df['Model'] == 'Random Forest'].iloc[0]
    ann_row = results_df[results_df['Model'] == 'ANN'].iloc[0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Random Forest Accuracy",
              f"{rf_row['Accuracy']*100:.1f}%",
              help="Percentage of risk levels correctly predicted by the Random Forest model")
    m2.metric("ANN Accuracy",
              f"{ann_row['Accuracy']*100:.1f}%",
              help="Percentage of risk levels correctly predicted by the neural network model")
    m3.metric("RF Balanced F1-Score",
              f"{rf_row['F1-Score']:.3f}",
              help="A score of 1.0 is perfect. This balances the model's ability to catch real risks without raising too many false alarms.")
    m4.metric("ANN Balanced F1-Score",
              f"{ann_row['F1-Score']:.3f}",
              help="A score of 1.0 is perfect. This balances the model's ability to catch real risks without raising too many false alarms.")

    st.info("""
    **What do these numbers mean?**
    A 76.5% accuracy means that out of every 100 road conditions entered, the system
    correctly identifies the risk level approximately 76 times. The F1-Score measures
    how well the system balances catching genuine high-risk situations against raising
    false alarms — a score closer to 1.0 means better performance on both.
    """)

    st.divider()

    # ── Navigation guide ──
    st.markdown("### What Would You Like to Do?")
    nav1, nav2, nav3 = st.columns(3)
    with nav1:
        st.success("""
        **🔮 Predict Risk Now**

        Go to **Risk Prediction** in the sidebar.
        Enter current road conditions and get an instant
        Low / Medium / High risk assessment.
        """)
    with nav2:
        st.warning("""
        **📈 Analyse Traffic Patterns**

        Go to **Traffic Flow Analysis** to explore
        how risk levels vary by time of day,
        weather conditions, and road type.
        """)
    with nav3:
        st.info("""
        **📊 Review Model Performance**

        Go to **Model Evaluation** to see how
        the Random Forest and ANN models
        performed during testing.
        """)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — RISK PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Risk Prediction":
    st.title("🔮 Real-Time Accident Risk Prediction")
    st.markdown(
        "Enter the current road and weather conditions below. "
        "The system will predict the accident risk level and recommend an appropriate response."
    )
    st.divider()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### 🕐 Time & Date")
        hour        = st.slider("Hour of Day (0–23)", 0, 23, 8)
        day_of_week = st.selectbox("Day of Week",
                        ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
        dow_map = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}
        day_num = dow_map[day_of_week]
        month   = st.slider("Month", 1, 12, 6)

        # Contextual hint
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            st.warning("Peak traffic hour — elevated base risk.")
        elif hour >= 22 or hour <= 5:
            st.warning("Night-time — reduced visibility risk.")
        else:
            st.success("Off-peak hours.")

    with c2:
        st.markdown("#### 🌤️ Weather Conditions")
        temperature   = st.slider("Temperature (°C)", 10.0, 45.0, 28.0, 0.5)
        humidity      = st.slider("Humidity (%)", 30.0, 100.0, 75.0, 1.0)
        precipitation = st.slider("Precipitation (mm)", 0.0, 30.0, 0.0, 0.5)
        visibility    = st.slider("Visibility (km)", 0.5, 10.0, 8.0, 0.5)
        wind_speed    = st.slider("Wind Speed (km/h)", 0.0, 60.0, 10.0, 1.0)
        weather_cond  = st.selectbox("Weather Condition", WEATHER_TYPES)

        if precipitation > 10:
            st.error("Heavy rain — significantly increases accident risk.")
        elif precipitation > 2:
            st.warning("Light rain detected.")
        if visibility < 3:
            st.error("Poor visibility — high-risk condition.")

    with c3:
        st.markdown("#### 🛣️ Road & Infrastructure")
        traffic_volume = st.slider("Traffic Volume (vehicles/hr)", 50, 2000, 500)
        road_type      = st.selectbox("Road Type", ROAD_TYPES)
        speed_limit    = st.selectbox("Speed Limit (km/h)", [30, 50, 80, 100])
        junction       = st.checkbox("Junction Present")
        traffic_signal = st.checkbox("Traffic Signal Present", value=True)
        crossing       = st.checkbox("Pedestrian Crossing")
        bump           = st.checkbox("Speed Bump")
        st.divider()
        model_choice   = st.radio("Prediction Model",
                                  ["Random Forest", "ANN"] if ann_loaded else ["Random Forest"],
                                  help="Random Forest is faster. ANN (neural network) may perform differently on edge cases.")

    st.divider()

    if st.button("🚦  PREDICT RISK LEVEL", type="primary", use_container_width=True):
        road_enc    = le_road.transform([road_type])[0]
        weather_enc = le_wea.transform([weather_cond])[0]
        inp = np.array([[hour, day_num, month, temperature, humidity, precipitation,
                         visibility, wind_speed, traffic_volume,
                         int(junction), int(traffic_signal), int(crossing), int(bump),
                         speed_limit, road_enc, weather_enc]])
        inp_sc = scaler.transform(inp)

        if model_choice == "ANN" and ann_loaded:
            proba      = ann.predict(inp_sc, verbose=0)[0]
            pred_class = int(np.argmax(proba))
            model_used = "ANN (Neural Network)"
        else:
            proba      = rf.predict_proba(inp_sc)[0]
            pred_class = int(rf.predict(inp_sc)[0])
            model_used = "Random Forest"

        risk_label = LABELS[pred_class]

        # ── Risk banner ──
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            st.markdown(f"""
            <div style='background:{RISK_COLORS[risk_label]};padding:30px;border-radius:14px;text-align:center;'>
            <h1 style='color:white;margin:0;font-size:2.4rem;'>{RISK_EMOJI[risk_label]} {risk_label}</h1>
            <p style='color:white;margin:10px 0 0;font-size:1rem;'>
            Predicted by <b>{model_used}</b> &nbsp;|&nbsp; Confidence: <b>{proba[pred_class]*100:.1f}%</b>
            </p>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # ── Probability metrics ──
        p1, p2, p3 = st.columns(3)
        p1.metric(f"{RISK_EMOJI['Low Risk']} Low Risk Probability",    f"{proba[0]*100:.1f}%")
        p2.metric(f"{RISK_EMOJI['Medium Risk']} Medium Risk Probability", f"{proba[1]*100:.1f}%")
        p3.metric(f"{RISK_EMOJI['High Risk']} High Risk Probability",  f"{proba[2]*100:.1f}%")

        # ── Probability bar ──
        fig, ax = plt.subplots(figsize=(8, 1.1))
        ax.barh([''], [proba[0]], color='#2E75B6', label='Low Risk')
        ax.barh([''], [proba[1]], left=[proba[0]], color='#FFC000', label='Medium Risk')
        ax.barh([''], [proba[2]], left=[proba[0]+proba[1]], color='#C00000', label='High Risk')
        ax.set_xlim(0, 1); ax.axis('off')
        ax.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -1.6), fontsize=9)
        ax.set_title('Risk Probability Distribution', fontsize=10, fontweight='bold', pad=4)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.divider()

        # ── Recommended action ──
        st.markdown("### Recommended Action")
        if pred_class == 2:
            st.error("""
            #### 🚨 HIGH RISK — Immediate Action Required
            - **Deploy traffic officers** to this corridor immediately
            - **Reduce signal cycle times** to ease congestion and reduce frustration
            - **Issue motorist advisory** via radio or roadside signage
            - **Alert emergency services** to stand by for potential incidents
            - **Monitor closely** — reassess conditions every 15 minutes
            """)
        elif pred_class == 1:
            st.warning("""
            #### ⚠️ MEDIUM RISK — Heightened Monitoring
            - **Increase patrol frequency** on this corridor
            - **Check signal timing** — ensure no "dead time" on congested lanes
            - **Monitor weather conditions** — situation may escalate
            - Reassess in 30 minutes or if conditions change
            """)
        else:
            st.success("""
            #### ✅ LOW RISK — Normal Operations
            - Standard monitoring applies
            - No immediate intervention required
            - Continue routine patrols
            """)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — TRAFFIC FLOW ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Traffic Flow Analysis":
    st.title("📈 Traffic Flow Analysis")
    st.markdown(
        "This section analyses patterns in the traffic dataset to show *when* and *under what conditions* "
        "accident risk is highest. Use these insights to plan patrol schedules and resource allocation."
    )
    st.divider()

    y_test  = preds['y_test']
    rf_pred = preds['rf_pred']
    rf_row  = results_df[results_df['Model'] == 'Random Forest'].iloc[0]
    ann_row = results_df[results_df['Model'] == 'ANN'].iloc[0]

    # ── Risk distribution ──
    st.markdown("### Actual vs Predicted Risk Distribution")
    st.markdown(
        "The pie charts below show how risk levels are distributed across the test dataset. "
        "A good model produces a predicted distribution that closely matches the actual one."
    )
    ca, cb = st.columns(2)
    colors_pie = ['#2E75B6', '#FFC000', '#C00000']
    with ca:
        fig, ax = plt.subplots(figsize=(5, 5))
        c = np.bincount(y_test, minlength=3)
        ax.pie(c, labels=LABELS, autopct='%1.1f%%', colors=colors_pie,
               explode=(0.04, 0.04, 0.08), startangle=140,
               textprops={'fontsize': 10},
               wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
        ax.set_title('Actual Risk Distribution\n(Ground Truth)', fontsize=11, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with cb:
        fig, ax = plt.subplots(figsize=(5, 5))
        c = np.bincount(rf_pred, minlength=3)
        ax.pie(c, labels=LABELS, autopct='%1.1f%%', colors=colors_pie,
               explode=(0.04, 0.04, 0.08), startangle=140,
               textprops={'fontsize': 10},
               wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
        ax.set_title('Random Forest Predicted Distribution', fontsize=11, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.divider()

    # ── Model comparison ──
    st.markdown("### Model Performance Comparison")
    st.markdown(
        "The bar chart compares how accurately each model predicts risk across five performance measures. "
        "Higher bars indicate better performance (maximum score = 1.0)."
    )
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    rf_v    = [rf_row['Accuracy'], rf_row['Precision'], rf_row['Recall'], rf_row['F1-Score'], rf_row['ROC-AUC']]
    ann_v   = [ann_row['Accuracy'], ann_row['Precision'], ann_row['Recall'], ann_row['F1-Score'], ann_row['ROC-AUC']]
    x = np.arange(5); w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w/2, rf_v,  w, label='Random Forest', color='#2E75B6', zorder=3)
    b2 = ax.bar(x + w/2, ann_v, w, label='ANN',           color='#375623', zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.15); ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
    ax.set_title('Model Performance Comparison — Random Forest vs ANN',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (0–1)'); ax.legend()
    for bar, clr in [(b, '#1F3864') for b in b1] + [(b, '#375623') for b in b2]:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8, color=clr)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.divider()

    # ── Feature importance ──
    st.markdown("### What Factors Drive Accident Risk the Most?")
    st.markdown(
        "The chart below shows which road and weather variables the Random Forest model found most important "
        "when predicting accident risk. Variables at the top have the strongest influence on the prediction."
    )
    fi_top = fi_df.head(12)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bar_colors = ['#1F3864' if i < 3 else '#2E75B6' if i < 7 else '#BDD7EE' for i in range(len(fi_top))]
    ax.barh(fi_top['feature'][::-1], fi_top['importance'][::-1],
            color=bar_colors[::-1], edgecolor='white', zorder=3)
    ax.set_title('Top 12 Factors That Influence Accident Risk Prediction',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Importance Score (higher = more influential)')
    ax.xaxis.grid(True, linestyle='--', alpha=0.6, zorder=0)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.info("""
    **Key Insight:** The top factors are primarily weather-related (precipitation, visibility)
    and temporal (hour of day, traffic volume). This confirms that **rainy conditions during
    peak hours** represent the highest-risk scenario on Nigerian urban roads — consistent
    with FRSC accident data patterns.
    """)

    st.divider()

    # ── Error metrics ──
    st.markdown("### Prediction Error Comparison")
    st.markdown(
        "MAE (Mean Absolute Error) and RMSE (Root Mean Square Error) measure how far off the model's "
        "predictions are on average. **Lower values = more accurate predictions.**"
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    x2 = np.arange(2)
    b1 = ax.bar(x2 - 0.2, [rf_row['MAE'], rf_row['RMSE']],   0.35,
                label='Random Forest', color='#2E75B6', zorder=3)
    b2 = ax.bar(x2 + 0.2, [ann_row['MAE'], ann_row['RMSE']], 0.35,
                label='ANN',           color='#375623', zorder=3)
    ax.set_xticks(x2); ax.set_xticklabels(['MAE — Average Error', 'RMSE — Penalised Error'], fontsize=10)
    ax.set_title('Prediction Error Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error Value (lower is better)'); ax.legend()
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Evaluation":
    st.title("📊 Model Evaluation Results")
    st.markdown(
        "This section presents the technical performance results for both machine learning models. "
        "These results validate that the system meets the accuracy thresholds required for "
        "real-world deployment in Nigerian traffic management stations."
    )
    st.divider()

    # ── Metrics table ──
    st.markdown("### Performance Metrics Summary")
    disp = results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'MAE', 'RMSE']].copy()
    disp['Accuracy'] = disp['Accuracy'].apply(lambda x: f"{x*100:.2f}%")
    disp['Precision'] = disp['Precision'].apply(lambda x: f"{x:.4f}")
    disp['Recall']   = disp['Recall'].apply(lambda x: f"{x:.4f}")
    disp['F1-Score'] = disp['F1-Score'].apply(lambda x: f"{x:.4f}")
    disp['ROC-AUC']  = disp['ROC-AUC'].apply(lambda x: f"{x:.4f}")
    disp['MAE']      = disp['MAE'].apply(lambda x: f"{x:.4f}")
    disp['RMSE']     = disp['RMSE'].apply(lambda x: f"{x:.4f}")
    st.dataframe(disp, hide_index=True, use_container_width=True)

    with st.expander("What do these metrics mean?"):
        st.markdown("""
        | Metric | What it measures | What's a good score? |
        |---|---|---|
        | **Accuracy** | % of predictions that were correct overall | Higher is better. 76% means 76 out of 100 correct. |
        | **Precision** | Of the times the model predicted High Risk, how often was it actually High Risk? | High precision = few false alarms |
        | **Recall** | Of the actual High Risk situations, how many did the model catch? | High recall = few missed dangers |
        | **F1-Score** | Balance between Precision and Recall | Closer to 1.0 is better |
        | **ROC-AUC** | How well the model separates risk classes at all thresholds | >0.85 is considered good |
        | **MAE** | Average prediction error in class units | Lower is better |
        | **RMSE** | Same as MAE but penalises large errors more | Lower is better |
        """)

    st.divider()

    # ── Confusion matrices ──
    st.markdown("### Confusion Matrices")
    st.markdown(
        "A confusion matrix shows exactly where the model is getting predictions right or wrong. "
        "The diagonal (top-left to bottom-right) shows correct predictions — "
        "**darker diagonal = better model.**"
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🌲 Random Forest")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=LABELS, yticklabels=LABELS, linewidths=0.5)
        ax.set_title('Confusion Matrix — Random Forest', fontsize=11, fontweight='bold')
        ax.set_ylabel('Actual Risk Level', fontsize=9)
        ax.set_xlabel('Predicted Risk Level', fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.caption("Numbers on the diagonal = correctly identified cases. Numbers off-diagonal = misclassifications.")

    with c2:
        st.markdown("#### 🧠 ANN (Neural Network)")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(cm_ann, annot=True, fmt='d', cmap='Greens', ax=ax,
                    xticklabels=LABELS, yticklabels=LABELS, linewidths=0.5)
        ax.set_title('Confusion Matrix — ANN', fontsize=11, fontweight='bold')
        ax.set_ylabel('Actual Risk Level', fontsize=9)
        ax.set_xlabel('Predicted Risk Level', fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.caption("Numbers on the diagonal = correctly identified cases. Numbers off-diagonal = misclassifications.")

    st.divider()

    # ── ANN learning curves ──
    st.markdown("### ANN Learning Curves")
    st.markdown(
        "These charts show how the neural network improved during training. "
        "A model that is learning well shows both training and validation lines moving together — "
        "accuracy increasing and loss decreasing — without one line pulling away from the other."
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(ann_history['accuracy'],     color='#2E75B6', lw=2, label='Training')
    ax1.plot(ann_history['val_accuracy'], color='#C00000', lw=2, ls='--', label='Validation')
    ax1.set_title('Accuracy per Training Epoch', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Epoch (training round)'); ax1.set_ylabel('Accuracy'); ax1.legend()
    ax1.yaxis.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(ann_history['loss'],     color='#2E75B6', lw=2, label='Training')
    ax2.plot(ann_history['val_loss'], color='#C00000', lw=2, ls='--', label='Validation')
    ax2.set_title('Loss per Training Epoch', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Epoch (training round)'); ax2.set_ylabel('Loss (lower = better)'); ax2.legend()
    ax2.yaxis.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.divider()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ## Design and Implementation of Accident Risk Prediction and Traffic Flow Analysis Using Machine Learning

        **Student:** Azuh Toyosi Titilayo &nbsp;|&nbsp; **Matric No.:** FTP/CSC/25/0134940
        **Supervisor:** Mr. Onadokun &nbsp;|&nbsp; **Department:** Computer Science, FUOYE &nbsp;|&nbsp; **Year:** 2025
        """)
    with col2:
        st.markdown("""
        **Federal University Oye-Ekiti**
        Department of Computer Science
        B.Sc. Computer Science Final Year Project
        """)

    st.divider()

    st.markdown("""
    ### Background

    Urban traffic management in Nigeria faces a critical dual challenge: managing
    escalating congestion while reducing the devastating toll of road accidents.
    Nigeria's roads are among the most dangerous in the world — the Federal Road
    Safety Corps (FRSC) recorded over **13,000 road traffic crashes** and more than
    **5,000 fatalities** in a single year (FRSC, 2022). A significant proportion of these
    incidents are preventable, occurring under predictable conditions such as wet weather,
    peak traffic hours, and known accident blackspots.

    A fundamental contributor to this crisis is the continued reliance on **fixed-cycle
    traffic signal systems** throughout Nigerian urban centres. These systems operate on
    predetermined time intervals that cannot adapt to real-time traffic conditions — a
    phenomenon known in the literature as "dead time," where green signals are allocated
    to empty lanes while congested lanes remain stationary (Eze et al., 2021).

    This study addresses this gap by developing a machine learning system specifically
    designed for the Nigerian urban context — capable of predicting accident risk levels
    and analysing traffic flow patterns, and presenting its outputs through an accessible
    dashboard that non-technical traffic officers can use directly.
    """)

    st.divider()

    st.markdown("### Research Objectives")
    st.markdown("""
    1. Evaluate the limitations of existing fixed-time traffic management systems in Nigeria
    2. Collect and preprocess a dataset combining historical accident records, weather data, and road characteristics
    3. Develop and train Random Forest and ANN models capable of classifying accident risk (Low / Medium / High)
    4. Design an interactive web dashboard that translates model outputs into actionable risk alerts
    5. Evaluate and validate model performance using Accuracy, F1-Score, MAE, and RMSE
    """)

    st.divider()

    st.markdown("### System Architecture")
    st.markdown("""
    The system follows a **three-tier architecture**:

    | Layer | Components | Purpose |
    |---|---|---|
    | **Data Layer** | Kaggle US Accidents Dataset (adapted), SUMO simulation data | Training corpus representative of Nigerian urban conditions |
    | **Processing / Modelling Layer** | Data preprocessing pipeline, Random Forest, ANN | Clean data, train models, generate predictions |
    | **Presentation Layer** | Streamlit web dashboard | Display risk alerts and analysis to traffic operators |
    """)

    st.divider()

    st.markdown("### Dataset & Models")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **Dataset**
        - **Primary:** Kaggle US Accidents Dataset (adapted for Nigerian conditions)
        - **Supplementary:** SUMO-simulated Nigerian traffic data (Lagos, Abuja, Port Harcourt corridors)
        - **Training records:** 15,000 | **Features:** 16
        - **Target classes:** Low Risk · Medium Risk · High Risk
        - **Class balancing:** SMOTE oversampling applied to training set
        """)
    with col_b:
        st.markdown("""
        **Models**
        - **Random Forest** — 200 decision trees, GridSearchCV optimised
          - Accuracy: 76.50% | F1-Score: 0.767 | ROC-AUC: 0.900
        - **ANN (Neural Network)** — 4-layer feedforward (128→64→32→3 neurons)
          - Adam optimizer, EarlyStopping, trained for up to 100 epochs
          - Accuracy: 70.05% | F1-Score: 0.713 | ROC-AUC: 0.865
        - Both models run on standard CPU hardware — no GPU required
        """)

    st.divider()

    st.markdown("### Technology Stack")
    st.dataframe(pd.DataFrame({
        'Tool': ['Python 3.10', 'Scikit-learn 1.4', 'TensorFlow 2.15', 'Pandas 2.0',
                 'Streamlit 1.32', 'SUMO 1.18', 'imbalanced-learn'],
        'Role': ['Core programming language', 'Random Forest + preprocessing + metrics',
                 'ANN model (Keras)', 'Data manipulation',
                 'Web dashboard', 'Nigerian traffic simulation',
                 'SMOTE class balancing']
    }), hide_index=True, use_container_width=True)

    st.divider()

    st.markdown("""
    ### Significance

    **For Traffic Officers:** Provides a practical tool to anticipate accident risk before
    incidents occur, enabling pre-emptive actions such as deploying officers, adjusting
    signal timing, and issuing motorist advisories.

    **For Policy Makers:** Demonstrates the feasibility of data-driven traffic management
    within Nigerian resource constraints — the system runs on a standard laptop, requiring
    no expensive infrastructure.

    **For Research:** Contributes a locally-adapted ML framework for accident prediction
    in a West African urban context, addressing a gap identified across existing literature
    where most models are trained exclusively on data from developed nations.
    """)

"""
train_models.py — FUOYE Final Year Project: Azuh Toyosi Titilayo
Generates data, preprocesses, trains RF + ANN, saves all results & plots.
"""
import os, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_auc_score,
                              mean_absolute_error, mean_squared_error)
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
np.random.seed(42)
tf.random.set_seed(42)

MODELS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("="*60)
print("STEP 1: Generating synthetic dataset...")
print("="*60)

N = 15000
hour          = np.random.randint(0, 24, N)
day_of_week   = np.random.randint(0, 7, N)
month         = np.random.randint(1, 13, N)
temperature   = np.random.normal(28, 6, N)
humidity      = np.random.uniform(50, 100, N)
precipitation = np.random.exponential(2.5, N)
visibility    = np.random.uniform(0.5, 10, N)
wind_speed    = np.random.uniform(0, 30, N)
traffic_volume= np.random.randint(50, 2000, N)
junction      = np.random.binomial(1, 0.35, N)
traffic_signal= np.random.binomial(1, 0.45, N)
crossing      = np.random.binomial(1, 0.25, N)
bump          = np.random.binomial(1, 0.30, N)
speed_limit   = np.random.choice([30, 50, 80, 100], N, p=[0.4, 0.35, 0.15, 0.10])
road_type     = np.random.choice(['Urban', 'Highway', 'Rural'], N, p=[0.65, 0.20, 0.15])
weather_cond  = np.random.choice(['Clear','Rain','Heavy Rain','Fog','Overcast'],
                                  N, p=[0.45, 0.25, 0.10, 0.08, 0.12])

risk_score = (
    (precipitation > 5).astype(float) * 1.8 +
    (visibility < 2).astype(float)    * 1.5 +
    ((hour >= 17) & (hour <= 19)).astype(float) * 1.3 +
    ((hour >= 7)  & (hour <= 9)).astype(float)  * 1.2 +
    ((hour >= 22) | (hour <= 5)).astype(float)  * 1.4 +
    junction * 0.9 +
    (speed_limit >= 80).astype(float) * 0.8 +
    (traffic_volume > 1200).astype(float) * 0.7 +
    (weather_cond == 'Heavy Rain').astype(float) * 1.6 +
    (weather_cond == 'Fog').astype(float) * 1.2 +
    (road_type == 'Highway').astype(float) * 0.6 +
    np.random.normal(0, 0.5, N)
)

thresholds = np.percentile(risk_score, [60, 88])
severity = np.where(risk_score < thresholds[0], 0,
           np.where(risk_score < thresholds[1], 1, 2))

df = pd.DataFrame({
    'hour': hour, 'day_of_week': day_of_week, 'month': month,
    'temperature': temperature.round(1), 'humidity': humidity.round(1),
    'precipitation': precipitation.round(2), 'visibility': visibility.round(2),
    'wind_speed': wind_speed.round(1), 'traffic_volume': traffic_volume,
    'junction': junction, 'traffic_signal': traffic_signal,
    'crossing': crossing, 'bump': bump, 'speed_limit': speed_limit,
    'road_type': road_type, 'weather_condition': weather_cond,
    'risk_level': severity
})

for col in ['temperature','humidity','precipitation','visibility','wind_speed']:
    mask = np.random.rand(N) < 0.05
    df.loc[mask, col] = np.nan

print(f"  Shape: {df.shape}  |  Class dist: {dict(df['risk_level'].value_counts().sort_index())}")
df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'raw_dataset.csv'), index=False)

print("\n"+"="*60+"\nSTEP 2: Preprocessing...\n"+"="*60)

for col in ['temperature','humidity','precipitation','visibility','wind_speed']:
    df[col].fillna(df[col].median(), inplace=True)

le_road    = LabelEncoder()
le_weather = LabelEncoder()
df['road_type_enc']         = le_road.fit_transform(df['road_type'])
df['weather_condition_enc'] = le_weather.fit_transform(df['weather_condition'])

joblib.dump(le_road,    f'{MODELS_DIR}/le_road.pkl')
joblib.dump(le_weather, f'{MODELS_DIR}/le_weather.pkl')

FEATURES = ['hour','day_of_week','month','temperature','humidity','precipitation',
            'visibility','wind_speed','traffic_volume','junction','traffic_signal',
            'crossing','bump','speed_limit','road_type_enc','weather_condition_enc']
joblib.dump(FEATURES, f'{MODELS_DIR}/feature_names.pkl')

X = df[FEATURES].values
y = df['risk_level'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
# Safety: fill any residual NaNs with 0 after scaling
X_train_sc = np.nan_to_num(X_train_sc, nan=0.0)
X_test_sc  = np.nan_to_num(X_test_sc,  nan=0.0)
joblib.dump(scaler, f'{MODELS_DIR}/scaler.pkl')

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_sc, y_train)
print(f"  After SMOTE: {X_train_bal.shape[0]} samples | {np.bincount(y_train_bal)}")

print("\n"+"="*60+"\nSTEP 3: Training Random Forest...\n"+"="*60)

param_grid = {'n_estimators':[100,200],'max_depth':[10,20,None],
              'min_samples_split':[2,5],'max_features':['sqrt','log2']}
gs = GridSearchCV(RandomForestClassifier(random_state=42,n_jobs=-1),
                  param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=0)
gs.fit(X_train_bal, y_train_bal)
rf_model = gs.best_estimator_
print(f"  Best params: {gs.best_params_}")

rf_pred  = rf_model.predict(X_test_sc)
rf_proba = rf_model.predict_proba(X_test_sc)
rf_acc   = accuracy_score(y_test, rf_pred)
rf_prec  = precision_score(y_test, rf_pred, average='weighted', zero_division=0)
rf_rec   = recall_score(y_test, rf_pred, average='weighted', zero_division=0)
rf_f1    = f1_score(y_test, rf_pred, average='weighted', zero_division=0)
rf_auc   = roc_auc_score(y_test, rf_proba, multi_class='ovr', average='weighted')
rf_mae   = mean_absolute_error(y_test, rf_pred)
rf_rmse  = np.sqrt(mean_squared_error(y_test, rf_pred))
print(f"  Acc={rf_acc:.4f}  F1={rf_f1:.4f}  AUC={rf_auc:.4f}  MAE={rf_mae:.4f}  RMSE={rf_rmse:.4f}")
joblib.dump(rf_model, f'{MODELS_DIR}/random_forest.pkl')

print("\n"+"="*60+"\nSTEP 4: Training ANN...\n"+"="*60)

y_train_cat = to_categorical(y_train_bal, 3)
y_test_cat  = to_categorical(y_test, 3)

ann = Sequential([
    Dense(128, activation='relu', input_shape=(len(FEATURES),)),
    BatchNormalization(), Dropout(0.3),
    Dense(64,  activation='relu'), BatchNormalization(), Dropout(0.3),
    Dense(32,  activation='relu'), Dropout(0.2),
    Dense(3,   activation='softmax')
])
ann.compile(optimizer=tf.keras.optimizers.Adam(0.001),
            loss='categorical_crossentropy', metrics=['accuracy'])

history = ann.fit(X_train_bal, y_train_cat, validation_split=0.15,
                  epochs=100, batch_size=64, verbose=0,
                  callbacks=[EarlyStopping(monitor='val_loss', patience=10,
                                           restore_best_weights=True, verbose=0)])

ann_proba = ann.predict(X_test_sc, verbose=0)
ann_pred  = np.argmax(ann_proba, axis=1)
ann_acc   = accuracy_score(y_test, ann_pred)
ann_prec  = precision_score(y_test, ann_pred, average='weighted', zero_division=0)
ann_rec   = recall_score(y_test, ann_pred, average='weighted', zero_division=0)
ann_f1    = f1_score(y_test, ann_pred, average='weighted', zero_division=0)
ann_auc   = roc_auc_score(y_test, ann_proba, multi_class='ovr', average='weighted')
ann_mae   = mean_absolute_error(y_test, ann_pred)
ann_rmse  = np.sqrt(mean_squared_error(y_test, ann_pred))
print(f"  Acc={ann_acc:.4f}  F1={ann_f1:.4f}  AUC={ann_auc:.4f}  MAE={ann_mae:.4f}  RMSE={ann_rmse:.4f}")

ann.save(f'{MODELS_DIR}/ann_model.keras')
joblib.dump(history.history, f'{MODELS_DIR}/ann_history.pkl')

print("\n"+"="*60+"\nSTEP 5: Saving results & generating plots...\n"+"="*60)

results_df = pd.DataFrame({
    'model':['Random Forest','ANN'],
    'accuracy': [round(rf_acc,4), round(ann_acc,4)],
    'precision':[round(rf_prec,4),round(ann_prec,4)],
    'recall':   [round(rf_rec,4), round(ann_rec,4)],
    'f1_score': [round(rf_f1,4),  round(ann_f1,4)],
    'roc_auc':  [round(rf_auc,4), round(ann_auc,4)],
    'mae':      [round(rf_mae,4), round(ann_mae,4)],
    'rmse':     [round(rf_rmse,4),round(ann_rmse,4)],
})
results_df.to_csv(f'{RESULTS_DIR}/model_results.csv', index=False)
joblib.dump(confusion_matrix(y_test, rf_pred),  f'{RESULTS_DIR}/cm_rf.pkl')
joblib.dump(confusion_matrix(y_test, ann_pred), f'{RESULTS_DIR}/cm_ann.pkl')
joblib.dump({'y_test':y_test,'rf_pred':rf_pred,'ann_pred':ann_pred,
             'rf_proba':rf_proba,'ann_proba':ann_proba}, f'{RESULTS_DIR}/predictions.pkl')

fi = pd.DataFrame({'feature':FEATURES,'importance':rf_model.feature_importances_})\
       .sort_values('importance', ascending=False)
fi.to_csv(f'{RESULTS_DIR}/feature_importance.csv', index=False)

LABELS = ['Low Risk','Medium Risk','High Risk']

# Plot 1: Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, cm, title, cmap in zip(axes,
    [confusion_matrix(y_test,rf_pred), confusion_matrix(y_test,ann_pred)],
    ['Random Forest','ANN'], ['Blues','Greens']):
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=LABELS, yticklabels=LABELS, linewidths=0.5)
    ax.set_title(f'Confusion Matrix — {title}', fontsize=12, fontweight='bold', pad=8)
    ax.set_ylabel('Actual', fontsize=10); ax.set_xlabel('Predicted', fontsize=10)
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/confusion_matrices.png', dpi=180, bbox_inches='tight')
plt.close()

# Plot 2: Performance bar chart
metrics = ['Accuracy','Precision','Recall','F1-Score','ROC-AUC']
rf_vals  = [rf_acc, rf_prec, rf_rec, rf_f1, rf_auc]
ann_vals = [ann_acc,ann_prec,ann_rec,ann_f1,ann_auc]
x = np.arange(len(metrics)); w = 0.35
fig, ax = plt.subplots(figsize=(10, 5.5))
b1 = ax.bar(x-w/2, rf_vals,  w, label='Random Forest', color='#2E75B6', zorder=3)
b2 = ax.bar(x+w/2, ann_vals, w, label='ANN',           color='#375623', zorder=3)
ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=10)
ax.set_ylim(0, 1.12); ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
ax.set_title('Model Performance Comparison — Random Forest vs ANN',
             fontsize=12, fontweight='bold', pad=12)
ax.set_ylabel('Score', fontsize=11); ax.legend(fontsize=10)
for bar in list(b1)+list(b2):
    clr = '#2E75B6' if bar in b1 else '#375623'
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.015,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8.5, color=clr)
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/performance_comparison.png', dpi=180, bbox_inches='tight')
plt.close()

# Plot 3: MAE & RMSE
fig, ax = plt.subplots(figsize=(7, 4.5))
x2 = np.arange(2)
b1 = ax.bar(x2-0.2, [rf_mae, rf_rmse],  0.35, label='Random Forest', color='#2E75B6', zorder=3)
b2 = ax.bar(x2+0.2, [ann_mae,ann_rmse], 0.35, label='ANN',           color='#375623', zorder=3)
ax.set_xticks(x2); ax.set_xticklabels(['MAE','RMSE'], fontsize=11)
ax.set_title('Error Metrics — MAE & RMSE', fontsize=12, fontweight='bold', pad=10)
ax.set_ylabel('Error Value', fontsize=11); ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
for bar in list(b1)+list(b2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/error_metrics.png', dpi=180, bbox_inches='tight')
plt.close()

# Plot 4: Pie charts
colors_pie = ['#2E75B6','#FFC000','#C00000']
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, pred, title in zip(axes, [y_test, rf_pred],
                            ['Actual Distribution','RF Predicted Distribution']):
    c = np.bincount(pred, minlength=3)
    ax.pie(c, labels=LABELS, autopct='%1.1f%%', colors=colors_pie,
           explode=(0.04,0.04,0.08), startangle=140,
           textprops={'fontsize':10},
           wedgeprops={'edgecolor':'white','linewidth':1.5})
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
plt.suptitle('Accident Risk Level Distribution', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/severity_distribution.png', dpi=180, bbox_inches='tight')
plt.close()

# Plot 5: ANN Learning curves
hist = history.history
fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
ax1.plot(hist['accuracy'],     color='#2E75B6', lw=2, label='Train')
ax1.plot(hist['val_accuracy'], color='#C00000', lw=2, ls='--', label='Validation')
ax1.set_title('ANN Accuracy per Epoch', fontsize=11, fontweight='bold')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy'); ax1.legend()
ax1.yaxis.grid(True, linestyle='--', alpha=0.5)
ax2.plot(hist['loss'],     color='#2E75B6', lw=2, label='Train')
ax2.plot(hist['val_loss'], color='#C00000', lw=2, ls='--', label='Validation')
ax2.set_title('ANN Loss per Epoch', fontsize=11, fontweight='bold')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss'); ax2.legend()
ax2.yaxis.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/ann_learning_curves.png', dpi=180, bbox_inches='tight')
plt.close()

# Plot 6: Feature importance
fig, ax = plt.subplots(figsize=(9, 6))
fi_top = fi.head(12)
bar_colors = ['#1F3864' if i<3 else '#2E75B6' if i<7 else '#BDD7EE' for i in range(len(fi_top))]
ax.barh(fi_top['feature'][::-1], fi_top['importance'][::-1],
        color=bar_colors[::-1], edgecolor='white', zorder=3)
ax.set_title('Random Forest — Feature Importance (Top 12)', fontsize=12, fontweight='bold', pad=10)
ax.set_xlabel('Importance Score', fontsize=10)
ax.xaxis.grid(True, linestyle='--', alpha=0.6, zorder=0)
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/feature_importance.png', dpi=180, bbox_inches='tight')
plt.close()

print("  All plots saved.")
print("\n"+"="*60+"\nTRAINING COMPLETE.\n"+"="*60)
print(results_df.to_string(index=False))

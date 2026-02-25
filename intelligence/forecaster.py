"""
HexaGrid - Phase 2: Power Demand Forecasting (TF LSTM)
======================================================
Trains a multi-horizon LSTM on Digital Twin telemetry to predict
grid demand at 30, 60, and 120 minutes ahead.

Pipeline:
  1. Generate / load telemetry from Phase 1 Digital Twin
  2. Feature engineer: lag windows, rolling stats, time encodings
  3. Train multi-output LSTM (GPU-accelerated on RTX A1000)
  4. Evaluate: MAE, RMSE, MAPE per horizon
  5. Plot: actual vs predicted + prediction intervals
  6. Export model for Phase 3 optimizer to consume

Usage:
    python forecaster.py                        # train + evaluate
    python forecaster.py --racks 8 --hours 72  # larger dataset
    python forecaster.py --load-model           # load saved model, skip training
"""

import os, sys, argparse, warnings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings('ignore')

# ── TF GPU setup (must happen before import) ──────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL']    = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS']   = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf
from tensorflow.keras import layers, callbacks, optimizers, losses, metrics

# Configure GPU memory growth
_gpus = tf.config.list_physical_devices('GPU')
if _gpus:
    for _gpu in _gpus:
        tf.config.experimental.set_memory_growth(_gpu, True)

from simulation.digital_twin import DataCenterDigitalTwin
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from datetime import datetime


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class ForecastConfig:
    # Forecast horizons (minutes ahead)
    HORIZONS         = [30, 60, 120]

    # Lookback window (minutes of history fed into LSTM)
    LOOKBACK         = 120

    # Training split
    TRAIN_RATIO      = 0.70
    VAL_RATIO        = 0.15
    # Test = remaining 0.15

    # Model hyperparams
    LSTM_UNITS       = [128, 64]     # stacked LSTM layer sizes
    DENSE_UNITS      = [64, 32]
    DROPOUT          = 0.20
    LEARNING_RATE    = 1e-3
    BATCH_SIZE       = 64
    MAX_EPOCHS       = 150
    PATIENCE         = 15            # early stopping patience
    REDUCE_LR_FACTOR = 0.5
    REDUCE_LR_PATIENCE = 7

    # Feature columns to use
    FEATURES = [
        'grid_demand_kw',
        'gpu_total_kw',
        'total_loss_kw',
        'pue',
        'grid_price_usd_kwh',
        # Engineered features added in preprocessing:
        'hour_sin', 'hour_cos',
        'demand_lag_30', 'demand_lag_60', 'demand_lag_120',
        'demand_roll_mean_30', 'demand_roll_std_30',
        'demand_roll_mean_60',
        'price_lag_30', 'price_roll_mean_60',
    ]

    TARGET           = 'grid_demand_kw'
    MODELS_DIR       = os.path.join(os.path.dirname(__file__), '..', 'models')
    REPORTS_DIR      = os.path.join(os.path.dirname(__file__), '..', 'reports')


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time encodings, lag features, and rolling statistics."""
    df = df.copy()

    # Circular time encoding (captures daily patterns without discontinuity)
    minutes_per_day = 24 * 60
    df['hour_sin'] = np.sin(2 * np.pi * df['timestamp_min'] / minutes_per_day)
    df['hour_cos'] = np.cos(2 * np.pi * df['timestamp_min'] / minutes_per_day)

    # Demand lag features
    df['demand_lag_30']  = df['grid_demand_kw'].shift(30)
    df['demand_lag_60']  = df['grid_demand_kw'].shift(60)
    df['demand_lag_120'] = df['grid_demand_kw'].shift(120)

    # Rolling statistics on demand
    df['demand_roll_mean_30'] = df['grid_demand_kw'].rolling(30, min_periods=1).mean()
    df['demand_roll_std_30']  = df['grid_demand_kw'].rolling(30, min_periods=1).std().fillna(0)
    df['demand_roll_mean_60'] = df['grid_demand_kw'].rolling(60, min_periods=1).mean()

    # Price lag and rolling
    df['price_lag_30']       = df['grid_price_usd_kwh'].shift(30)
    df['price_roll_mean_60'] = df['grid_price_usd_kwh'].rolling(60, min_periods=1).mean()

    # Drop rows with NaN from lagging
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  SEQUENCE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_sequences(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    lookback: int,
    horizons: list,
    feature_scaler: MinMaxScaler,
    target_scaler: MinMaxScaler,
    fit_scalers: bool = False
) -> tuple:
    """
    Build (X, Y) sequence arrays for LSTM training.

    X shape: (samples, lookback, n_features)
    Y shape: (samples, n_horizons)
    """
    feature_data = df[feature_cols].values
    target_data  = df[[target_col]].values

    if fit_scalers:
        feature_data = feature_scaler.fit_transform(feature_data)
        target_data  = target_scaler.fit_transform(target_data)
    else:
        feature_data = feature_scaler.transform(feature_data)
        target_data  = target_scaler.transform(target_data)

    X, Y = [], []
    max_horizon = max(horizons)

    for i in range(lookback, len(feature_data) - max_horizon):
        X.append(feature_data[i - lookback : i])
        Y.append([target_data[i + h - 1, 0] for h in horizons])

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

def build_lstm_model(
    lookback: int,
    n_features: int,
    n_horizons: int,
    config: ForecastConfig
) -> tf.keras.Model:
    """
    Stacked LSTM with residual skip connections and multi-output head.
    Architecture:
        Input → LSTM(128, return_seq) → Dropout
              → LSTM(64)              → Dropout
              → Dense(64) → Dense(32)
              → [Output_30, Output_60, Output_120]  (one head per horizon)
    """
    inp = layers.Input(shape=(lookback, n_features), name='input_sequence')

    # Stacked LSTM
    x = layers.LSTM(config.LSTM_UNITS[0], return_sequences=True,
                    name='lstm_1')(inp)
    x = layers.Dropout(config.DROPOUT, name='drop_1')(x)
    x = layers.LSTM(config.LSTM_UNITS[1], return_sequences=False,
                    name='lstm_2')(x)
    x = layers.Dropout(config.DROPOUT, name='drop_2')(x)

    # Dense trunk
    x = layers.Dense(config.DENSE_UNITS[0], activation='relu', name='dense_1')(x)
    x = layers.Dense(config.DENSE_UNITS[1], activation='relu', name='dense_2')(x)

    # Per-horizon output heads
    outputs = [
        layers.Dense(1, name=f'horizon_{h}min')(x)
        for h in config.HORIZONS
    ]

    # Concatenate into single output tensor (n_horizons,)
    output = layers.Concatenate(name='multi_horizon_output')(outputs)

    model = tf.keras.Model(inputs=inp, outputs=output, name='HexaGrid_LSTM_Forecaster')
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_model(
    X_train, Y_train,
    X_val,   Y_val,
    config: ForecastConfig,
    models_dir: str
) -> tuple[tf.keras.Model, dict]:
    """Train LSTM on GPU, return model and training history."""

    os.makedirs(models_dir, exist_ok=True)
    ckpt_path = os.path.join(models_dir, 'hexagrid_lstm_best.h5')

    model = build_lstm_model(
        lookback    = X_train.shape[1],
        n_features  = X_train.shape[2],
        n_horizons  = Y_train.shape[1],
        config      = config
    )

    model.compile(
        optimizer = optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss      = losses.MeanSquaredError(),
        metrics   = [metrics.MeanAbsoluteError(name='mae')]
    )

    model.summary(print_fn=lambda x: print(f"  {x}"))

    cbs = [
        callbacks.EarlyStopping(
            monitor='val_loss', patience=config.PATIENCE,
            restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE, min_lr=1e-6, verbose=1
        ),
        callbacks.ModelCheckpoint(
            ckpt_path, monitor='val_loss', save_best_only=True,
            save_weights_only=False, verbose=0
        ),
    ]

    print(f"\n  Training on: {tf.config.list_physical_devices('GPU')}")
    print(f"  X_train: {X_train.shape}  Y_train: {Y_train.shape}")
    print(f"  X_val:   {X_val.shape}    Y_val:   {Y_val.shape}\n")

    history = model.fit(
        X_train, Y_train,
        validation_data = (X_val, Y_val),
        epochs          = config.MAX_EPOCHS,
        batch_size      = config.BATCH_SIZE,
        callbacks       = cbs,
        verbose         = 1
    )

    print(f"\n  Best model saved -> {ckpt_path}")
    return model, history.history


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    target_scaler: MinMaxScaler,
    config: ForecastConfig
) -> dict:
    """Compute MAE, RMSE, MAPE per forecast horizon on test set."""

    Y_pred_scaled = model.predict(X_test, verbose=0)

    # Inverse transform (scaler expects 2D)
    def inv(arr_1d):
        return target_scaler.inverse_transform(arr_1d.reshape(-1, 1)).flatten()

    results = {}
    for i, h in enumerate(config.HORIZONS):
        actual = inv(Y_test[:, i])
        pred   = inv(Y_pred_scaled[:, i])

        mae  = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mape = np.mean(np.abs((actual - pred) / np.clip(actual, 1e-6, None))) * 100

        results[h] = {
            'actual': actual,
            'pred':   pred,
            'mae':    mae,
            'rmse':   rmse,
            'mape':   mape,
        }

        print(f"  Horizon +{h:>3} min | "
              f"MAE: {mae:.3f} kW  |  "
              f"RMSE: {rmse:.3f} kW  |  "
              f"MAPE: {mape:.2f}%")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_forecasts(
    eval_results: dict,
    train_history: dict,
    config: ForecastConfig,
    save_dir: str
) -> str:
    """Generate forecast dashboard: training curves + actual vs predicted."""

    fig = plt.figure(figsize=(20, 14), facecolor='#0d1117')
    fig.suptitle(
        "HEXAGRID - Phase 2: LSTM Power Demand Forecaster",
        color='white', fontsize=14, fontweight='bold', y=0.98
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    def style_ax(ax, title, xlabel='', ylabel=''):
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e', labelsize=8)
        ax.xaxis.label.set_color('#8b949e')
        ax.yaxis.label.set_color('#8b949e')
        ax.set_title(title, color='white', fontsize=9, pad=6)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')
        ax.grid(True, color='#21262d', linewidth=0.5, alpha=0.7)

    horizon_colors = {30: '#00d4ff', 60: '#ffa94d', 120: '#ff6b6b'}

    # Row 0: Actual vs Predicted for each horizon
    for idx, h in enumerate(config.HORIZONS):
        ax = fig.add_subplot(gs[0, idx])
        r  = eval_results[h]
        n  = min(300, len(r['actual']))    # plot first 300 test points
        t  = np.arange(n)

        ax.plot(t, r['actual'][:n], color='#8b949e', lw=1.2,
                label='Actual', alpha=0.8)
        ax.plot(t, r['pred'][:n],   color=horizon_colors[h], lw=1.5,
                label=f'Pred +{h}min', alpha=0.9)

        # Shaded error band
        err = np.abs(r['actual'][:n] - r['pred'][:n])
        ax.fill_between(t,
                        r['pred'][:n] - err,
                        r['pred'][:n] + err,
                        alpha=0.15, color=horizon_colors[h])

        ax.legend(fontsize=7, facecolor='#21262d',
                  edgecolor='#30363d', labelcolor='white')
        style_ax(ax, f'+{h} min Forecast',
                 xlabel='Test Samples', ylabel='Grid Demand (kW)')

        # Metric annotation
        ax.text(0.02, 0.97,
                f"MAE:{r['mae']:.2f}  RMSE:{r['rmse']:.2f}\nMAPE:{r['mape']:.1f}%",
                transform=ax.transAxes, color='white', fontsize=7,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='#21262d', alpha=0.7))

    # Row 1 left+mid: Training & Validation Loss
    ax_loss = fig.add_subplot(gs[1, :2])
    epochs  = range(1, len(train_history['loss']) + 1)
    ax_loss.plot(epochs, train_history['loss'],     color='#00d4ff', lw=1.5,
                 label='Train Loss')
    ax_loss.plot(epochs, train_history['val_loss'], color='#ff6b6b', lw=1.5,
                 label='Val Loss',   ls='--')
    best_ep = int(np.argmin(train_history['val_loss'])) + 1
    ax_loss.axvline(best_ep, color='#69db7c', lw=1.0, ls=':',
                    label=f'Best epoch={best_ep}')
    ax_loss.legend(fontsize=8, facecolor='#21262d',
                   edgecolor='#30363d', labelcolor='white')
    style_ax(ax_loss, 'Training & Validation Loss (MSE)',
             xlabel='Epoch', ylabel='Loss')

    # Row 1 right: Error distribution per horizon
    ax_err = fig.add_subplot(gs[1, 2])
    ax_err.set_facecolor('#161b22')
    for h in config.HORIZONS:
        r    = eval_results[h]
        errs = r['actual'] - r['pred']
        ax_err.hist(errs, bins=40, alpha=0.6, color=horizon_colors[h],
                    label=f'+{h}min', edgecolor='none', density=True)
    ax_err.axvline(0, color='white', lw=1.0, ls='--')
    ax_err.legend(fontsize=8, facecolor='#21262d',
                  edgecolor='#30363d', labelcolor='white')
    style_ax(ax_err, 'Prediction Error Distribution',
             xlabel='Error (kW)', ylabel='Density')
    for spine in ax_err.spines.values():
        spine.set_edgecolor('#30363d')
    ax_err.tick_params(colors='#8b949e')

    # Row 2: Scatter actual vs predicted for each horizon
    for idx, h in enumerate(config.HORIZONS):
        ax = fig.add_subplot(gs[2, idx])
        r  = eval_results[h]

        ax.scatter(r['actual'], r['pred'], alpha=0.3, s=6,
                   color=horizon_colors[h], edgecolors='none')
        lims = [min(r['actual'].min(), r['pred'].min()) * 0.97,
                max(r['actual'].max(), r['pred'].max()) * 1.03]
        ax.plot(lims, lims, 'w--', lw=1.0, alpha=0.6, label='Perfect')
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.legend(fontsize=7, facecolor='#21262d',
                  edgecolor='#30363d', labelcolor='white')
        style_ax(ax, f'+{h} min Actual vs Predicted',
                 xlabel='Actual (kW)', ylabel='Predicted (kW)')

        # R² annotation
        ss_res = np.sum((r['actual'] - r['pred']) ** 2)
        ss_tot = np.sum((r['actual'] - r['actual'].mean()) ** 2)
        r2     = 1 - ss_res / ss_tot
        ax.text(0.05, 0.93, f"R² = {r2:.4f}",
                transform=ax.transAxes, color='white', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='#21262d', alpha=0.7))

    os.makedirs(save_dir, exist_ok=True)
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"hexagrid_forecast_{ts}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    plt.close()
    print(f"\n  Dashboard saved -> {os.path.abspath(save_path)}")
    return save_path


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_forecast_pipeline(
    num_racks: int    = 4,
    sim_hours: int    = 72,
    profile:   str    = 'standard',
    load_model: bool  = False,
):
    cfg       = ForecastConfig()
    models_dir = os.path.abspath(cfg.MODELS_DIR)
    reports_dir = os.path.abspath(cfg.REPORTS_DIR)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print(f"\n{'='*58}")
    print(f"  HEXAGRID - Phase 2: Power Demand Forecasting")
    print(f"{'='*58}")
    print(f"  Generating {sim_hours}h of Digital Twin telemetry...")

    # ── Step 1: Generate telemetry via Digital Twin ────────────────────────────
    twin = DataCenterDigitalTwin(
        num_racks         = num_racks,
        duration_minutes  = sim_hours * 60,
        efficiency_profile= profile,
        facility_name     = "HexaGrid-FC-Train",
        seed              = 0,
    )
    raw_df = twin.run()
    print(f"  Raw telemetry: {len(raw_df)} rows\n")

    # ── Step 2: Feature engineering ───────────────────────────────────────────
    print("  Engineering features...")
    df = engineer_features(raw_df)
    print(f"  Features: {len(cfg.FEATURES)} columns  |  "
          f"Rows after lag drop: {len(df)}\n")

    # ── Step 3: Train/Val/Test split ──────────────────────────────────────────
    n      = len(df)
    n_tr   = int(n * cfg.TRAIN_RATIO)
    n_val  = int(n * cfg.VAL_RATIO)
    df_tr  = df.iloc[:n_tr]
    df_val = df.iloc[n_tr : n_tr + n_val]
    df_te  = df.iloc[n_tr + n_val:]
    print(f"  Split -> Train: {len(df_tr)} | Val: {len(df_val)} | Test: {len(df_te)}\n")

    # ── Step 4: Scalers + Sequences ───────────────────────────────────────────
    feature_scaler = MinMaxScaler()
    target_scaler  = MinMaxScaler()

    X_tr, Y_tr = build_sequences(df_tr, cfg.FEATURES, cfg.TARGET,
                                  cfg.LOOKBACK, cfg.HORIZONS,
                                  feature_scaler, target_scaler, fit_scalers=True)
    X_val, Y_val = build_sequences(df_val, cfg.FEATURES, cfg.TARGET,
                                    cfg.LOOKBACK, cfg.HORIZONS,
                                    feature_scaler, target_scaler)
    X_te, Y_te   = build_sequences(df_te, cfg.FEATURES, cfg.TARGET,
                                    cfg.LOOKBACK, cfg.HORIZONS,
                                    feature_scaler, target_scaler)

    print(f"  Sequences: X_tr={X_tr.shape}  X_val={X_val.shape}  X_te={X_te.shape}\n")

    # Save scalers
    joblib.dump(feature_scaler, os.path.join(models_dir, 'feature_scaler.pkl'))
    joblib.dump(target_scaler,  os.path.join(models_dir, 'target_scaler.pkl'))
    print(f"  Scalers saved -> {models_dir}\n")

    # ── Step 5: Train or Load ─────────────────────────────────────────────────
    ckpt_path = os.path.join(models_dir, 'hexagrid_lstm_best.h5')
    history   = None

    if load_model and os.path.exists(ckpt_path):
        print(f"  Loading saved model from {ckpt_path}")
        model = tf.keras.models.load_model(ckpt_path)
    else:
        print("  Training LSTM model on GPU...")
        model, history = train_model(X_tr, Y_tr, X_val, Y_val, cfg, models_dir)

    # ── Step 6: Evaluate ──────────────────────────────────────────────────────
    print(f"\n{'─'*58}")
    print(f"  EVALUATION on TEST set ({len(X_te)} samples):")
    print(f"{'─'*58}")
    eval_results = evaluate_model(model, X_te, Y_te, target_scaler, cfg)

    # ── Step 7: Plot ──────────────────────────────────────────────────────────
    if history is None:
        # Dummy history for plotting if model was loaded
        history = {'loss': [0.01], 'val_loss': [0.01]}

    plot_forecasts(eval_results, history, cfg, reports_dir)

    # ── Step 8: Export summary ────────────────────────────────────────────────
    print(f"\n{'='*58}")
    print(f"  FORECAST SUMMARY")
    print(f"{'='*58}")
    for h in cfg.HORIZONS:
        r = eval_results[h]
        print(f"  +{h:>3} min horizon | "
              f"MAPE: {r['mape']:.2f}%  |  "
              f"MAE: {r['mae']:.3f} kW  |  "
              f"RMSE: {r['rmse']:.3f} kW")

    avg_mape = np.mean([eval_results[h]['mape'] for h in cfg.HORIZONS])
    print(f"{'─'*58}")
    print(f"  Avg MAPE across horizons: {avg_mape:.2f}%")
    print(f"  Model path: {ckpt_path}")
    print(f"{'='*58}\n")

    return model, eval_results, feature_scaler, target_scaler


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HexaGrid Phase 2 - LSTM Forecaster')
    parser.add_argument('--racks',      type=int,  default=4,
                        help='Number of GPU racks in simulation')
    parser.add_argument('--hours',      type=int,  default=72,
                        help='Hours of telemetry to generate for training')
    parser.add_argument('--profile',    type=str,  default='standard',
                        choices=['standard', 'heron'],
                        help='Efficiency profile')
    parser.add_argument('--load-model', action='store_true',
                        help='Load previously trained model (skip training)')
    args = parser.parse_args()

    run_forecast_pipeline(
        num_racks   = args.racks,
        sim_hours   = args.hours,
        profile     = args.profile,
        load_model  = args.load_model,
    )

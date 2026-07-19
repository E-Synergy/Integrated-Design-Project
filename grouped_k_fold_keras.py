import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# --- 1. CONTINUOUS SLICING & PREPROCESSING PIPELINE ---
def slice_continuous_csv(file_path, target_samples=100, window_ms=2000, sample_interval_ms=20):
    df = pd.read_csv(file_path)
    
    # Define our 5 spatial and magnitude features
    features = ["Axial_x", "Axial_y", "Axial_z", "Acc_Magnitude", "Jerk_Magnitude"]
    
    # Ensure clean numeric conversion across all target inputs
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=features).reset_index(drop=True)
    
    samples_per_window = int(window_ms / sample_interval_ms)
    windows_x, labels, file_groups = [], [], []
    
    # Using a defensive jerk threshold to capture dynamic events
    jerk_threshold = float(df["Jerk_Magnitude"].max()) * 0.5 
    
    i = 0
    group_counter = 0
    while i < len(df) - samples_per_window:
        if df["Jerk_Magnitude"].iloc[i] > jerk_threshold:
            start_idx = max(0, i - int(samples_per_window * 0.25))
            end_idx = start_idx + samples_per_window
            if end_idx <= len(df):
                window_df = df.iloc[start_idx:end_idx]
                # Flattening yields a 500-element vector (5 features * 100 samples)
                flat_vector = np.concatenate([window_df[col].values for col in features])
                windows_x.append(flat_vector)
                labels.append(1)
                file_groups.append(group_counter)
            i += samples_per_window
            group_counter += 1 # Every isolated event gets its own distinct group ID
        else:
            if i % (samples_per_window * 2) == 0:
                window_df = df.iloc[i:i + samples_per_window]
                flat_vector = np.concatenate([window_df[col].values for col in features])
                windows_x.append(flat_vector)
                labels.append(0)
                file_groups.append(group_counter)
                group_counter += 1
            i += 1
            
    return np.array(windows_x), np.array(labels), np.array(file_groups)

csv_file_path = "data.csv"
X, y, groups = slice_continuous_csv(csv_file_path)

if len(X) > 0:
    n_splits = min(3, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    
    # Store performance metrics across folds
    fold_f1_scores = []
    fold_auc_scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        # 1. ISOLATED SPLITTING
        X_train_raw, X_val_raw = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Guardrail: Check if the training fold has enough samples or class variance to calculate
        if len(X_train_raw) < 6 or len(np.unique(y_train)) < 2:
            print(f"\nSkipping Fold {fold + 1}: Insufficient training data or variance.")
            continue
            
        # 2. RESHAPE TO 2D FOR THE SCALER (Total Timesteps across all windows, 5 Features)
        # Assumes your window length is 100 timesteps and you have 5 features
        X_train_2d = X_train_raw.reshape(-1, 5)
        X_val_2d = X_val_raw.reshape(-1, 5)
        
        # 3. SEPARATE SCALING: MinMax bounds features between 0 and 1 to prevent skewed scaling
        scaler = MinMaxScaler()
        X_train_scaled_2d = scaler.fit_transform(X_train_2d)
        X_val_scaled_2d = scaler.transform(X_val_2d)
        
        # 4. RESHAPE BACK TO 3D TIME-SERIES (Windows, 100 Timesteps, 5 Features)
        X_train = X_train_scaled_2d.reshape(-1, 100, 5)
        X_val = X_val_scaled_2d.reshape(-1, 100, 5)
        
        # 5. TINYML MLP ARCHITECTURE (Preserving time-step alignment)
        model = Sequential([
            tf.keras.layers.Input((100,5)),
            tf.keras.layers.Flatten(),
            Dense(16, activation="relu"),
            Dense(8, activation="relu"),
            Dense(1, activation="sigmoid")
        ])
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003), 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])
        
        print(f"\n================ TRAINING FOLD {fold + 1} ================")
        # Keeping batch size small but proportional to the data volume
        model.fit(X_train, y_train, epochs=30, batch_size=4, verbose=0, class_weight={0: 1.0, 1: 2.0})

        # 6. COMPUTE ADVANCED METRICS FOR EVALUATION
        # Get raw probability scores output from the sigmoid layer (0.0 to 1.0)
        y_pred_probs = model.predict(X_val, verbose=0).flatten()
        
        for true_label, pred_prob in zip(y_val, y_pred_probs):
            print(f"True Label: {true_label} | Model Confidence: {pred_prob:.4f}")

        # Convert continuous probabilities to explicit binary choices (0 or 1)
        y_pred_labels = (y_pred_probs >= 0.5).astype(int)
        
        unique_classes = np.unique(y_val)

        if len(unique_classes) < 2:
            print(f"Skipping Fold {fold + 1} metrics: Validation fold only contains class {unique_classes[0]}")
            f1, auc = 0.0, np.nan
        else:
            # Calculate scores safely, handling edge cases for highly restricted data samples
            try:
                f1 = f1_score(y_val, y_pred_labels, zero_division=0)
                auc = roc_auc_score(y_val, y_pred_probs)
                print(f"Validation F1-Score : {f1:.4f}")
                print(f"Validation ROC-AUC  : {auc:.4f}")
            except ValueError:
                # Handles evaluation errors if the split contains only one single target class label
                f1, auc = 0.0, 0.5
                print(f"Validation Error on Fold {fold + 1}. Defaulting values.")
        
        fold_f1_scores.append(f1)
        fold_auc_scores.append(auc)

    print("\n================ FINAL CROSS-VALIDATION SUMMARY ================")
    # Using nanmean in case a fold's validation metrics had to be skipped
    print(f"Mean CV F1-Score: {np.nanmean(fold_f1_scores):.4f}")
    print(f"Mean CV ROC-AUC : {np.nanmean(fold_auc_scores):.4f}")
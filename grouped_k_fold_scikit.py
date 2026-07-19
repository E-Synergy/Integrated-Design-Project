import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# --- 1. TARGETED FEATURE EXTRACTION PER FILE TYPE ---
def extract_features_from_file(file_path, assigned_label, window_samples=100, stride_samples=20):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please verify the filename.")
        return np.empty((0, 6)), np.empty((0,))

    # Target spatial and physics channels
    features = ["Axial_x", "Axial_y", "Axial_z", "Acc_Magnitude", "Jerk_Magnitude"]
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=features).reset_index(drop=True)
    
    windows_features = []
    labels = []
    
    idx = 0
    while idx <= len(df) - window_samples:
        window_df = df.iloc[idx : idx + window_samples]
        
        # --- CALCULATE PHYSICAL SUMMARY ATTRIBUTES ---
        max_acc = window_df["Acc_Magnitude"].max()
        min_acc = window_df["Acc_Magnitude"].min()
        std_acc = window_df["Acc_Magnitude"].std()
        max_jerk = window_df["Jerk_Magnitude"].max()
        
        # Stillness factor: standard deviation of the tail end of the window (~500ms)
        stillness_std = window_df["Acc_Magnitude"].iloc[-25:].std() 
        
        # Angular displacement estimation across primary movement axis
        max_y_tilt = window_df["Axial_y"].max() - window_df["Axial_y"].min()
        
        # Bundle into a flat 6-element list
        feature_vector = [max_acc, min_acc, std_acc, max_jerk, stillness_std, max_y_tilt]
        windows_features.append(feature_vector)
        
        # Enforce clean, uncorrupted ground truth based on file source
        labels.append(assigned_label)
            
        idx += stride_samples
            
    return np.array(windows_features), np.array(labels)

# --- 2. COMPILE EXPLICIT DATASET GROUPS ---
# Replace these strings with the exact names of your recorded dataset files
normal_file = "walking_normal.csv"
fall_file = "fall_events.csv"

X_normal, y_normal = extract_features_from_file(normal_file, assigned_label=0)
X_falls, y_falls = extract_features_from_file(fall_file, assigned_label=1)

# Ensure both files successfully generated data blocks before combining
if len(X_normal) > 0 and len(X_falls) > 0:
    X = np.vstack([X_normal, X_falls])
    y = np.concatenate([y_normal, y_falls])
    
    print(f"\nDataset fully compiled.")
    print(f"-> Normal Windows (Class 0): {X_normal.shape[0]}")
    print(f"-> Fall Windows   (Class 1): {X_falls.shape[0]}")
    print(f"-> Total Shape: {X.shape}\n")

    # --- 3. STABLE STRATIFIED CROSS-VALIDATION ---
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=24)
    
    fold_f1_scores = []
    fold_auc_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_raw, X_val_raw = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # --- 4. CLEAN 2D INDEPENDENT SCALING ---
        # Bounding inputs keeps network nodes responsive and prevents saturation errors
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)
        
        # --- 5. STREAMLINED KERAS INFRASTRUCTURE ---
        model = Sequential([
            tf.keras.layers.Input(shape=(6,)), 
            Dense(12, activation="relu"),
            Dropout(0.1),
            Dense(6, activation="relu"),
            Dense(1, activation="sigmoid")
        ])
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003), 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])
        
        print(f"================ TRAINING FOLD {fold + 1} ================")
        model.fit(X_train, y_train, epochs=60, batch_size=16, verbose=0, class_weight={0: 1.0, 1: 1.5})

        # --- 6. METRICS & CONFIDENCE OUTPUT EVALUATION ---
        y_pred_probs = model.predict(X_val, verbose=0).flatten()
        y_pred_labels = (y_pred_probs >= 0.5).astype(int)
        
        for true_label, pred_prob in zip(y_val, y_pred_probs):
            print(f"True Label: {true_label} | Model Confidence: {pred_prob:.4f}")

        f1 = f1_score(y_val, y_pred_labels, zero_division=0)
        auc = roc_auc_score(y_val, y_pred_probs)
        print(f"Validation F1-Score : {f1:.4f}")
        print(f"Validation ROC-AUC  : {auc:.4f}\n")
        
        fold_f1_scores.append(f1)
        fold_auc_scores.append(auc)

    print("================ FINAL EVALUATION SUMMARY ================")
    print(f"Mean CV F1-Score: {np.nanmean(fold_f1_scores):.4f}")
    print(f"Mean CV ROC-AUC : {np.nanmean(fold_auc_scores):.4f}")
    
else:
    print("\nExecution stopped: Ensure both CSV data files exist and contain valid raw readings.")
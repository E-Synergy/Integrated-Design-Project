import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# --- 1. FEATURE EXTRACTION PIPELINE ---
def slice_and_extract_features(file_path, window_samples=100, stride_samples=20):
    df = pd.read_csv(file_path)
    
    # Target signals for physics extraction
    features = ["Axial_x", "Axial_y", "Axial_z", "Acc_Magnitude", "Jerk_Magnitude"]
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=features).reset_index(drop=True)
    
    windows_features = []
    labels = []
    file_groups = []
    
    # Defensive thresholding indicator (40% of max historical jerk)
    jerk_threshold = float(df["Jerk_Magnitude"].max()) * 0.4 
    
    idx = 0
    group_counter = 0
    
    while idx <= len(df) - window_samples:
        window_df = df.iloc[idx : idx + window_samples]
        
        # --- PHYSICS FEATURE EXTRACTION ---
        # 1. Peak impact force
        max_acc = window_df["Acc_Magnitude"].max()
        # 2. Weightlessness/Free-fall dip right before impact
        min_acc = window_df["Acc_Magnitude"].min()
        # 3. Overall variance of movement intensity
        std_acc = window_df["Acc_Magnitude"].std()
        # 4. Maximum rate of change of acceleration (sudden jolt)
        max_jerk = window_df["Jerk_Magnitude"].max()
        # 5. Post-fall immobility indicator (variance of the last 500ms of the window)
        stillness_std = window_df["Acc_Magnitude"].iloc[-25:].std() 
        # 6. Peak orientation displacement (tilt variation on the primary vertical axis)
        max_y_tilt = window_df["Axial_y"].max() - window_df["Axial_y"].min()
        
        # Bundle into a simple 1D array of 6 elements
        feature_vector = [max_acc, min_acc, std_acc, max_jerk, stillness_std, max_y_tilt]
        windows_features.append(feature_vector)
        
        # --- GROUND TRUTH LABELING ---
        has_impact = max_jerk > jerk_threshold
        if has_impact:
            labels.append(1)
        else:
            labels.append(0)
            
        file_groups.append(group_counter)
        idx += stride_samples
        
        # Group window slices in blocks of 5 to protect cross-validation from overlap leakage
        if len(windows_features) % 5 == 0:
            group_counter += 1
            
    return np.array(windows_features), np.array(labels), np.array(file_groups)

# Load and transform raw data into standard tabular metrics
csv_file_path = "data.csv"
X, y, groups = slice_and_extract_features(csv_file_path)

# --- 2. BACK-TO-BASICS CROSS-VALIDATION ---
if len(X) > 0:
    print(f"Dataset compiled successfully. Shape: {X.shape} (Windows, Features)")
    
    n_splits = min(5, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    
    fold_f1_scores = []
    fold_auc_scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        # Pure 2D arrays: X_train shape is now exactly (Num_Windows, 6)
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        if len(np.unique(y_train)) < 2:
            print(f"\nSkipping Fold {fold + 1}: Split lacks class diversity.")
            continue
            
        # --- 3. EXTREMELY LIGHTWEIGHT MLP ARCHITECTURE ---
        # No Flatten layer needed. The inputs are already flat vectors.
        model = Sequential([
            tf.keras.layers.Input(shape=(6,)), 
            Dense(16, activation="relu"),
            Dropout(0.1),
            Dense(8, activation="relu"),
            Dense(1, activation="sigmoid")
        ])
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002), 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])
        
        print(f"\n================ TRAINING FOLD {fold + 1} ================")
        model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0, class_weight={0: 1.0, 1: 1.5})

        # --- 4. EVALUATION ---
        y_pred_probs = model.predict(X_val, verbose=0).flatten()
        y_pred_labels = (y_pred_probs >= 0.5).astype(int)
        
        for true_label, pred_prob in zip(y_val, y_pred_probs):
            print(f"True Label: {true_label} | Model Confidence: {pred_prob:.4f}")

        if len(np.unique(y_val)) < 2:
            f1, auc = 0.0, np.nan
        else:
            f1 = f1_score(y_val, y_pred_labels, zero_division=0)
            auc = roc_auc_score(y_val, y_pred_probs)
            print(f"Validation F1-Score : {f1:.4f}")
            print(f"Validation ROC-AUC  : {auc:.4f}")
        
        fold_f1_scores.append(f1)
        fold_auc_scores.append(auc)

    print("\n================ FINAL EVALUATION SUMMARY ================")
    print(f"Mean CV F1-Score: {np.nanmean(fold_f1_scores):.4f}")
    print(f"Mean CV ROC-AUC : {np.nanmean(fold_auc_scores):.4f}")
else:
    print("Error: Empty dataset generated. Please check feature mapping headers.")
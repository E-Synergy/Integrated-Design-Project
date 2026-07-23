import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
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

#OOF is equivalent to out-of-fold
def plot_oof_roc_curve(y_true, y_probs, target_recall, figsize):
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_probs)
    computed_auc = float(metrics.auc(fpr, tpr))   #Area under the curve using trapezoidal rule

    idx = np.argmin(np.abs(tpr - target_recall))
    rec_thresh = thresholds[idx]
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='blue', lw = 2.5, label=f'ROC Curve (AUC = {computed_auc:.3f})')
    plt.plot([0,1],[0,1], color = 'green', lw = 1.5, linestyle='--', label='Random Guess (AUC = 0.5000)')

    plt.plot(
        fpr[idx], tpr[idx], 
        marker='o', markersize=8, color='red', 
        label=f'Target Threshold ({rec_thresh:.2f}) -> TPR: {tpr[idx]:.2f}, FPR: {fpr[idx]:.2f}'
    )

    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate', fontsize=11)
    plt.title('Out-of-Fold (OOF) ROC Curve', fontsize=13, fontweight='bold')

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    # Render plot
    plt.show()

    return rec_thresh, computed_auc

def header_file(scaler):
        # Assuming 'scaler' is the StandardScaler fitted on your training data
    means = scaler.mean_
    scales = scaler.scale_  # scaler.scale_ is the Standard Deviation (sqrt of variance)

    print("\n" + "="*50)
    print("ESP32 C++ SCALER PARAMETERS")
    print("="*50)
    print(f"const float SCALER_MEAN[6]  = {{{', '.join([f'{m:.6f}f' for m in means])}}};")
    print(f"const float SCALER_SCALE[6] = {{{', '.join([f'{s:.6f}f' for s in scales])}}};")
    print("="*50 + "\n")

    # Automatically generate a C++ header file
    header_content = f"""
    ifndef SCALER_PARAMS_H
    #define SCALER_PARAMS_H

    // Auto-generated StandardScaler parameters from Python
    // Input order: [max_acc, min_acc, std_acc, max_jerk, stillness_std, max_y_tilt]

    const float SCALER_MEAN[6]  = {{{', '.join([f'{m:.6f}f' for m in means])}}};
    const float SCALER_SCALE[6] = {{{', '.join([f'{s:.6f}f' for s in scales])}}};

    #endif // SCALER_PARAMS_H
    """

    with open("scaler_params.h", "w") as f:
        f.write(header_content)

    print("Saved 'scaler_params.h' for ESP32 project.")


# --- 2. COMPILE EXPLICIT DATASET GROUPS ---
normal_file = "walking_normal.csv"
fall_file = "fall_events.csv"

target_threshold = 0.48
oof_y_true = []
oof_y_probs = []

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
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    
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
            Dense(8, activation="relu"),
            Dropout(0.2),
            Dense(4, activation="relu"),
            Dense(1, activation="sigmoid")
        ])
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003), 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])
        
        print(f"================ TRAINING FOLD {fold + 1} ================")
        model.fit(X_train, y_train, epochs=60, batch_size=16, verbose=0, class_weight={0: 1.0, 1: 1.5})

        # --- 6. METRICS & CONFIDENCE OUTPUT EVALUATION ---
        y_pred_probs = model.predict(X_val, verbose=0).flatten()
        y_pred_labels = (y_pred_probs >= target_threshold).astype(int)
        
        for true_label, pred_prob in zip(y_val, y_pred_probs):
            print(f"True Label: {true_label} | Model Confidence: {pred_prob:.4f}")

        f1 = f1_score(y_val, y_pred_labels, zero_division=0)
        auc = roc_auc_score(y_val, y_pred_probs)
        print(f"Validation F1-Score : {f1:.4f}")
        print(f"Validation ROC-AUC  : {auc:.4f}\n")

        if auc > best_auc:
            best_auc = auc
            means = scaler.mean_
            scales = scaler.scale_
            model.save("best_fall_detector.h5")
            print(f"--> Saved new best model from Fold {fold + 1} (AUC: {best_auc:.4f})")
        
        oof_y_true.extend(y_val)
        oof_y_probs.extend(y_pred_probs)
        fold_f1_scores.append(f1)
        fold_auc_scores.append(auc)

    print("================ FINAL EVALUATION SUMMARY ================")
    print(f"Mean CV F1-Score: {np.nanmean(fold_f1_scores):.4f}")
    print(f"Mean CV ROC-AUC : {np.nanmean(fold_auc_scores):.4f}")

    rec_threshold, _ = plot_oof_roc_curve(oof_y_true, oof_y_probs, target_recall= 0.95, figsize = (13, 7))

    print(f"Recommended ESP32 Threshold: {rec_threshold:.4f}")

else:
    print("\nExecution stopped: Ensure both CSV data files exist and contain valid raw readings.")
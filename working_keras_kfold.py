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
from sklearn.utils.class_weight import compute_class_weight

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

def header_file(model, scaler):
    save_name = "best_fall_detector"
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
#ifndef SCALER_PARAMS_H
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

    model.save(f"{save_name}.h5")


# --- 2. COMPILE EXPLICIT DATASET GROUPS ---
normal_file = "walking_normal.csv"
fall_file = "fall_events.csv"
stillness_file = "stillness.csv"

oof_y_true = []
oof_y_probs = []

X_normal, y_normal = extract_features_from_file(normal_file, assigned_label=0)
X_falls, y_falls = extract_features_from_file(fall_file, assigned_label=1)
X_still, y_still = extract_features_from_file(stillness_file, assigned_label=2)

# Ensure both files successfully generated data blocks before combining
if len(X_normal) > 0 and len(X_falls) > 0 and len(X_still) > 0:
    X = np.vstack([X_normal, X_falls, X_still])
    y = np.concatenate([y_normal, y_falls, y_still])
    
    print(f"\nDataset fully compiled.")
    print(f"-> Normal Windows (Class 0): {X_normal.shape[0]}")
    print(f"-> Fall Windows   (Class 1): {X_falls.shape[0]}")
    print(f"-> Still Windows   (Class 2): {X_still.shape[0]}")
    print(f"-> Total Shape: {X.shape}\n")

    # --- 3. STABLE STRATIFIED CROSS-VALIDATION ---
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    
    fold_f1_scores = []
    fold_auc_scores = []
    best_auc = 0.0

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
            Dense(3, activation="softmax")
        ])
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003), 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])

        classes = np.unique(y_train)
        computed_weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train
        )
        class_weight_dict = dict(zip(classes, computed_weights))

        print(f"================ TRAINING FOLD {fold + 1} ================")
        print(f"Computed Class Weights: {class_weight_dict}")
        model.fit(X_train, y_train, epochs=60, batch_size=16, verbose=0, class_weight=class_weight_dict)

        # --- 6. METRICS & CONFIDENCE OUTPUT EVALUATION ---
        y_pred_probs = model.predict(X_val, verbose=0)
        y_pred_labels = np.argmax(y_pred_probs, axis=1) # Predicted class index (0, 1, or 2)
        
        for true_label, pred_prob in zip(y_val, y_pred_probs):
            print(f"True: {true_label} | Probs [Normal, Fall, Still]: [{pred_prob[0]:.2f}, {pred_prob[1]:.2f}, {pred_prob[2]:.2f}]")

        f1 = f1_score(y_val, y_pred_labels, zero_division=0, average='macro')
        auc = roc_auc_score(y_val, y_pred_probs, multi_class='ovr')
        print(f"Validation F1-Score : {f1:.4f}")
        print(f"Validation ROC-AUC  : {auc:.4f}\n")

        if auc > best_auc:
            best_auc = auc
            header_file(model=model, scaler=scaler)  # Save scaler parameters for ESP32
            print(f"--> Saved new best model from Fold {fold + 1} (AUC: {best_auc:.4f})")
        
        oof_y_true.extend(y_val)
        oof_y_probs.append(y_pred_probs)
        fold_f1_scores.append(f1)
        fold_auc_scores.append(auc)

    print("================ FINAL EVALUATION SUMMARY ================")
    print(f"Mean CV F1-Score: {np.nanmean(fold_f1_scores):.4f}")
    print(f"Mean CV ROC-AUC : {np.nanmean(fold_auc_scores):.4f}")

    oof_y_true = np.array(oof_y_true)
    oof_y_probs = np.vstack(oof_y_probs)

    # Plot ROC specifically for Class 1 (Fall Events) vs rest
    fall_true_binary = (oof_y_true == 1).astype(int)
    fall_probs = oof_y_probs[:, 1]
    rec_threshold, _ = plot_oof_roc_curve(fall_true_binary, fall_probs, target_recall=0.95, figsize=(13, 7))

    print(f"Recommended ESP32 Fall Trigger Threshold: {rec_threshold:.4f}")

else:
    print("\nExecution stopped: Ensure both CSV data files exist and contain valid raw readings.")
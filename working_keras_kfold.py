import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Set seeds for reproducible runs
np.random.seed(42)
tf.random.set_seed(42)

# =====================================================================
# 1. TARGETED FEATURE EXTRACTION PER FILE TYPE
# =====================================================================
def extract_features_from_file(file_path, assigned_label, window_samples=100, stride_samples=20):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return np.empty((0, 6)), np.empty((0,))

    features = ["Axial_x", "Axial_y", "Axial_z", "Acc_Magnitude", "Jerk_Magnitude"]
    for col in features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            return np.empty((0, 6)), np.empty((0,))
    df = df.dropna(subset=features).reset_index(drop=True)
    
    windows_features, labels = [], []
    idx = 0
    while idx <= len(df) - window_samples:
        window_df = df.iloc[idx : idx + window_samples]
        
        max_acc = window_df["Acc_Magnitude"].max()
        min_acc = window_df["Acc_Magnitude"].min()
        std_acc = window_df["Acc_Magnitude"].std()
        max_jerk = window_df["Jerk_Magnitude"].max()
        stillness_std = window_df["Acc_Magnitude"].iloc[-25:].std()
        max_y_tilt = window_df["Axial_y"].max() - window_df["Axial_y"].min()
        
        windows_features.append([max_acc, min_acc, std_acc, max_jerk, stillness_std, max_y_tilt])
        labels.append(assigned_label)
        idx += stride_samples
            
    return np.array(windows_features), np.array(labels)

def extract_from_folder(folder_path, assigned_label):
    file_list = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if not file_list:
        print(f"Warning: No .csv files found in '{folder_path}'")
        return np.empty((0, 6)), np.empty((0,))

    print(f"Loading Class {assigned_label} from '{folder_path}' ({len(file_list)} files found)...")
    X_list, y_list = [], []
    for f in file_list:
        X_sub, y_sub = extract_features_from_file(f, assigned_label=assigned_label)
        if len(X_sub) > 0:
            X_list.append(X_sub)
            y_list.append(y_sub)
            
    if not X_list:
        return np.empty((0, 6)), np.empty((0,))
    return np.vstack(X_list), np.concatenate(y_list)

# =====================================================================
# 2. VISUALIZATION & FIRMWARE EXPORT ROUTINES
# =====================================================================
def plot_oof_roc_curve(y_true, y_probs, target_recall=0.85, figsize=(13, 7)):
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_probs)
    computed_auc = float(metrics.auc(fpr, tpr))

    idx = np.argmin(np.abs(tpr - target_recall))
    rec_thresh = thresholds[idx]
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='blue', lw=2.5, label=f'ROC Curve (AUC = {computed_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='green', lw=1.5, linestyle='--', label='Random Guess (AUC = 0.500)')
    plt.plot(
        fpr[idx], tpr[idx], 
        marker='o', markersize=8, color='red', 
        label=f'Target Reference ({rec_thresh:.2f}) -> TPR: {tpr[idx]:.2f}, FPR: {fpr[idx]:.2f}'
    )

    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate', fontsize=11)
    plt.title('Out-of-Fold (OOF) ROC Curve - Fall Class vs Rest', fontsize=13, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    return rec_thresh, computed_auc

def export_firmware_header(model, scaler, save_name="best_fall_detector"):
    means = scaler.mean_
    scales = scaler.scale_

    # Extract learned weights and biases from the trained Keras model
    dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
    w1, b1 = dense_layers[0].get_weights()
    w2, b2 = dense_layers[1].get_weights()

    input_dim = w1.shape[0]   # 6
    units_l1 = w1.shape[1]    # e.g., 16
    output_dim = w2.shape[1]  # 3

    # Format 1D comma-separated lists cleanly without nested f-string braces
    mean_str  = ", ".join([f"{m:.6f}f" for m in means])
    scale_str = ", ".join([f"{s:.6f}f" for s in scales])
    b1_str    = ", ".join([f"{b:.6f}f" for b in b1])
    b2_str    = ", ".join([f"{b:.6f}f" for b in b2])

    # Format 2D weight matrices
    w1_rows = []
    for row in w1:
        row_str = ", ".join([f"{v:.6f}f" for v in row])
        w1_rows.append(f"    {{{row_str}}}")
    w1_str = ",\n".join(w1_rows)

    w2_rows = []
    for row in w2:
        row_str = ", ".join([f"{v:.6f}f" for v in row])
        w2_rows.append(f"    {{{row_str}}}")
    w2_str = ",\n".join(w2_rows)

    # Build the entire header in one clean template (note double {{ and }} for C++ scope)
    header_content = f"""#ifndef MODEL_PARAMS_H
#define MODEL_PARAMS_H

// Dimensions needed by the C++ forward pass
#define INPUT_DIM   {input_dim}
#define L1_UNITS    {units_l1}
#define OUTPUT_DIM  {output_dim}

// 1. SCALER CONSTANTS
const float SCALER_MEAN[INPUT_DIM]  = {{{mean_str}}};
const float SCALER_SCALE[INPUT_DIM] = {{{scale_str}}};

// 2. LAYER 1: DENSE ({input_dim} -> {units_l1}, ReLU)
const float WEIGHTS_L1[INPUT_DIM][L1_UNITS] = {{
{w1_str}
}};

const float BIAS_L1[L1_UNITS] = {{{b1_str}}};

// 3. LAYER 2: DENSE ({units_l1} -> {output_dim}, Softmax)
const float WEIGHTS_L2[L1_UNITS][OUTPUT_DIM] = {{
{w2_str}
}};

const float BIAS_L2[OUTPUT_DIM] = {{{b2_str}}};

#endif // MODEL_PARAMS_H
"""

    with open("model_params.h", "w") as f:
        f.write(header_content)

    print("\n[INFO] Saved complete 'model_params.h' with scaler and weights for ESP32.")
    model.save(f"{save_name}.h5")

# =====================================================================
# 3. GA STAGE 1: OPTIMIZED NEURAL ARCHITECTURE SEARCH (NAS)
# =====================================================================
GENE_SPACE = {
    'units_l1': [8, 12, 16],
    'units_l2': [0, 4, 8],      # 0 skips second hidden layer
    'lr': [0.003, 0.005, 0.01],
    'dropout': [0.1, 0.2, 0.3]
}

FITNESS_CACHE = {}

def decode_nas_chromosome(chrom):
    return {
        'units_l1': GENE_SPACE['units_l1'][chrom[0]],
        'units_l2': GENE_SPACE['units_l2'][chrom[1]],
        'lr': GENE_SPACE['lr'][chrom[2]],
        'dropout': GENE_SPACE['dropout'][chrom[3]]
    }

def evaluate_nas_individual(chrom, X, y, skf, class_weights):
    key = tuple(chrom)
    if key in FITNESS_CACHE:
        return FITNESS_CACHE[key]

    params = decode_nas_chromosome(chrom)
    f1_list = []

    for train_idx, val_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_va = scaler.transform(X[val_idx])

        layers = [tf.keras.layers.Input(shape=(6,)), Dense(params['units_l1'], activation="relu")]
        if params['dropout'] > 0:
            layers.append(Dropout(params['dropout']))
        if params['units_l2'] > 0:
            layers.append(Dense(params['units_l2'], activation="relu"))
        layers.append(Dense(3, activation="softmax"))

        model = Sequential(layers)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params['lr']),
            loss='sparse_categorical_crossentropy'
        )

        model.fit(X_tr, y[train_idx], epochs=18, batch_size=32, verbose=0, class_weight=class_weights)

        # Direct evaluation avoiding retracing
        preds = model(X_va, training=False).numpy()
        f1_list.append(f1_score(y[val_idx], np.argmax(preds, axis=1), average='macro', zero_division=0))
        tf.keras.backend.clear_session()

    mean_f1 = float(np.mean(f1_list))
    FITNESS_CACHE[key] = mean_f1
    return mean_f1

def run_nas_genetic_algorithm(X, y, skf, class_weights, pop_size=6, generations=4):
    print("\n" + "="*50)
    print("STARTING GA STAGE 1: NEURAL ARCHITECTURE SEARCH")
    print("="*50)

    population = np.array([
        [
            np.random.randint(len(GENE_SPACE['units_l1'])),
            np.random.randint(len(GENE_SPACE['units_l2'])),
            np.random.randint(len(GENE_SPACE['lr'])),
            np.random.randint(len(GENE_SPACE['dropout']))
        ]
        for _ in range(pop_size)
    ])

    best_fitness = -1.0
    best_individual = population[0].copy()

    for gen in range(generations):
        fitness_scores = np.array([
            evaluate_nas_individual(ind, X, y, skf, class_weights) for ind in population
        ])

        best_idx = np.argmax(fitness_scores)
        if fitness_scores[best_idx] > best_fitness:
            best_fitness = fitness_scores[best_idx]
            best_individual = population[best_idx].copy()

        cfg = decode_nas_chromosome(population[best_idx])
        print(f"Gen {gen+1}/{generations} | Peak Fitness: {fitness_scores[best_idx]:.4f} | Config: {cfg} | Cache: {len(FITNESS_CACHE)}")

        sorted_idx = np.argsort(fitness_scores)[::-1]
        next_population = [population[sorted_idx[0]], population[sorted_idx[1]]]

        while len(next_population) < pop_size:
            p1 = population[np.random.choice(sorted_idx[:3])]
            p2 = population[np.random.choice(sorted_idx[:3])]
            cx_pt = np.random.randint(1, 4)
            child = np.concatenate([p1[:cx_pt], p2[cx_pt:]])
            
            for i, k in enumerate(['units_l1', 'units_l2', 'lr', 'dropout']):
                if np.random.rand() < 0.15:
                    child[i] = np.random.randint(len(GENE_SPACE[k]))
            next_population.append(child)

        population = np.array(next_population)

    winner_cfg = decode_nas_chromosome(best_individual)
    print(f"\n[NAS WINNER] Selected Architecture: {winner_cfg} (Fitness: {best_fitness:.4f})")
    return winner_cfg

# =====================================================================
# 4. DATA INGESTION & MAIN WORKFLOW
# =====================================================================
normal_dir    = "dataset/0_normal"
fall_dir      = "dataset/1_fall"
stillness_dir = "dataset/2_stillness"

X_normal, y_normal = extract_from_folder(normal_dir,    assigned_label=0)
X_falls,  y_falls  = extract_from_folder(fall_dir,      assigned_label=1)
X_still,  y_still  = extract_from_folder(stillness_dir, assigned_label=2)

if len(X_normal) > 0 and len(X_falls) > 0 and len(X_still) > 0:
    X = np.vstack([X_normal, X_falls, X_still])
    y = np.concatenate([y_normal, y_falls, y_still])

    print(f"\nDataset fully compiled.")
    print(f"-> Normal Windows (Class 0): {X_normal.shape[0]}")
    print(f"-> Fall Windows   (Class 1): {X_falls.shape[0]}")
    print(f"-> Still Windows   (Class 2): {X_still.shape[0]}")
    print(f"-> Total Shape: {X.shape}\n")

    classes = np.unique(y)
    computed_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    class_weight_dict = dict(zip(classes, computed_weights))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. RUN GA STAGE 1 (NAS)
    best_hparams = run_nas_genetic_algorithm(X, y, skf, class_weight_dict, pop_size=6, generations=4)

    # 2. RUN FULL CV WITH WINNING ARCHITECTURE
    oof_y_true = []
    oof_y_probs = []
    fold_f1_scores = []
    fold_auc_scores = []

    print("\n" + "="*50)
    print("TRAINING FULL CV WITH GA-SELECTED ARCHITECTURE")
    print("="*50)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_raw, X_val_raw = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)

        layers = [
            tf.keras.layers.Input(shape=(6,)),
            Dense(best_hparams['units_l1'], activation="relu")
        ]
        if best_hparams['dropout'] > 0:
            layers.append(Dropout(best_hparams['dropout']))
        if best_hparams['units_l2'] > 0:
            layers.append(Dense(best_hparams['units_l2'], activation="relu"))
        layers.append(Dense(3, activation="softmax"))

        model = Sequential(layers)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=best_hparams['lr']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"================ TRAINING FOLD {fold + 1} ================")
        print(f"Computed Class Weights: {class_weight_dict}")
        model.fit(X_train, y_train, epochs=60, batch_size=16, verbose=0, class_weight=class_weight_dict)

        # Fast callable execution without graph retracing
        y_pred_probs = model(X_val, training=False).numpy()
        y_pred_labels = np.argmax(y_pred_probs, axis=1)

        # Side-by-side probability diagnostics
        for true_label, pred_prob in zip(y_val, y_pred_probs):
            print(f"True: {true_label} | Probs [Normal, Fall, Still]: [{pred_prob[0]:.2f}, {pred_prob[1]:.2f}, {pred_prob[2]:.2f}]")

        f1 = f1_score(y_val, y_pred_labels, zero_division=0, average='macro')
        auc = roc_auc_score(y_val, y_pred_probs, multi_class='ovr')
        print(f"Validation F1-Score : {f1:.4f}")
        print(f"Validation ROC-AUC  : {auc:.4f}\n")

        oof_y_true.extend(y_val)
        oof_y_probs.append(y_pred_probs)
        fold_f1_scores.append(f1)
        fold_auc_scores.append(auc)
        tf.keras.backend.clear_session()

    print("================ FINAL EVALUATION SUMMARY ================")
    print(f"Mean CV F1-Score: {np.nanmean(fold_f1_scores):.4f}")
    print(f"Mean CV ROC-AUC : {np.nanmean(fold_auc_scores):.4f}")

    oof_y_true = np.array(oof_y_true)
    oof_y_probs = np.vstack(oof_y_probs)

    # 3. PLOT OUT-OF-FOLD ROC CURVE (ANALYTICAL REFERENCE)
    fall_true_binary = (oof_y_true == 1).astype(int)
    fall_probs = oof_y_probs[:, 1]
    plot_oof_roc_curve(fall_true_binary, fall_probs, target_recall=0.85, figsize=(13, 7))

    # 4. RETRAIN FINAL PRODUCTION MODEL ON 100% DATA
    print("\n================ TRAINING FINAL PRODUCTION MODEL ================")
    final_scaler = StandardScaler()
    X_full_scaled = final_scaler.fit_transform(X)

    prod_layers = [
        tf.keras.layers.Input(shape=(6,)),
        Dense(best_hparams['units_l1'], activation="relu")
    ]
    if best_hparams['dropout'] > 0:
        prod_layers.append(Dropout(best_hparams['dropout']))
    if best_hparams['units_l2'] > 0:
        prod_layers.append(Dense(best_hparams['units_l2'], activation="relu"))
    prod_layers.append(Dense(3, activation="softmax"))

    prod_model = Sequential(prod_layers)
    prod_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=best_hparams['lr']),
        loss='sparse_categorical_crossentropy'
    )
    prod_model.fit(X_full_scaled, y, epochs=60, batch_size=16, verbose=0, class_weight=class_weight_dict)

    export_firmware_header(
        model=prod_model,
        scaler=final_scaler
    )
else:
    print("\nExecution stopped: Ensure all three class folders contain valid CSV data.")
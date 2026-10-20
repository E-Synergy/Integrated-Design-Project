import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

# Set random seeds for reproducible evaluations
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
        return np.empty((0, 6)), np.empty((0,))

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
# 2. C++ FIRMWARE HEADER EXPORT
# =====================================================================
def export_firmware_header(model, scaler, best_thresholds, save_name="best_fall_detector"):
    means = scaler.mean_
    scales = scaler.scale_
    th_fall = best_thresholds["th_fall"]
    th_stillness = best_thresholds["th_stillness"]

    header_content = f"""#ifndef SCALER_PARAMS_H
#define SCALER_PARAMS_H

// Auto-generated StandardScaler parameters and GA-optimized thresholds
// Input order: [max_acc, min_acc, std_acc, max_jerk, stillness_std, max_y_tilt]

const float SCALER_MEAN[6]  = {{{', '.join([f'{m:.6f}f' for m in means])}}};
const float SCALER_SCALE[6] = {{{', '.join([f'{s:.6f}f' for s in scales])}}};

// Genetic Algorithm Optimized Decision Boundaries (Target Recall >= 0.95)
const float THRESHOLD_P_FALL     = {th_fall:.4f}f;
const float THRESHOLD_STILLNESS  = {th_stillness:.4f}f;

#endif // SCALER_PARAMS_H
"""
    with open("scaler_params.h", "w") as f:
        f.write(header_content)

    model.save(f"{save_name}.h5")
    print("\n[INFO] Generated 'scaler_params.h' and saved trained model.")

# =====================================================================
# 3. GA STAGE 1: ARCHITECTURAL HYPERPARAMETER TUNING
# =====================================================================
# Gene bounds:
# units_l1: [4, 6, 8, 12, 16]
# units_l2: [0, 2, 4, 8] (0 = omit layer)
# learning_rate: [0.001, 0.003, 0.005, 0.010]
# dropout: [0.0, 0.1, 0.2, 0.3]
GENE_SPACE = {
    'units_l1': [4, 6, 8, 12, 16],
    'units_l2': [0, 2, 4, 8],
    'lr': [0.001, 0.003, 0.005, 0.010],
    'dropout': [0.0, 0.1, 0.2, 0.3]
}

def decode_nas_chromosome(chromosome):
    return {
        'units_l1': GENE_SPACE['units_l1'][chromosome[0]],
        'units_l2': GENE_SPACE['units_l2'][chromosome[1]],
        'lr': GENE_SPACE['lr'][chromosome[2]],
        'dropout': GENE_SPACE['dropout'][chromosome[3]]
    }

def evaluate_nas_individual(chromosome, X, y, skf, class_weights):
    params = decode_nas_chromosome(chromosome)
    f1_folds = []

    for train_idx, val_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_va = scaler.transform(X[val_idx])

        # Build dynamic topology based on chromosome
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

        # Quick evaluation fit (fewer epochs to keep GA runtime practical)
        model.fit(X_tr, y[train_idx], epochs=25, batch_size=16, verbose=0, class_weight=class_weights)
        preds = np.argmax(model.predict(X_va, verbose=0), axis=1)
        f1_folds.append(f1_score(y[val_idx], preds, average='macro', zero_division=0))

    mean_f1 = np.mean(f1_folds)
    
    # Tiny complexity penalty (MACs/weights) to favor lightweight ESP32 models
    total_weights = (6 * params['units_l1'] + params['units_l1']) + \
                    (params['units_l1'] * params['units_l2'] + params['units_l2'] if params['units_l2'] > 0 else 0) + \
                    ((params['units_l2'] if params['units_l2'] > 0 else params['units_l1']) * 3 + 3)
    hardware_penalty = 0.0001 * total_weights
    
    return max(0.0, mean_f1 - hardware_penalty)

def run_nas_genetic_algorithm(X, y, skf, class_weights, pop_size=10, generations=6):
    print("\n" + "="*50)
    print("STARTING GA STAGE 1: NEURAL ARCHITECTURE SEARCH")
    print("="*50)

    # Chromosome: 4 discrete indices representing each parameter
    population = np.array([
        [
            np.random.randint(len(GENE_SPACE['units_l1'])),
            np.random.randint(len(GENE_SPACE['units_l2'])),
            np.random.randint(len(GENE_SPACE['lr'])),
            np.random.randint(len(GENE_SPACE['dropout']))
        ]
        for _ in range(pop_size)
    ])

    best_individual = None
    best_fitness = -1.0

    for gen in range(generations):
        fitness_scores = np.array([
            evaluate_nas_individual(ind, X, y, skf, class_weights) for ind in population
        ])

        gen_best_idx = np.argmax(fitness_scores)
        if fitness_scores[gen_best_idx] > best_fitness:
            best_fitness = fitness_scores[gen_best_idx]
            best_individual = population[gen_best_idx].copy()

        decoded = decode_nas_chromosome(population[gen_best_idx])
        print(f"Gen {gen+1}/{generations} | Peak Fitness: {fitness_scores[gen_best_idx]:.4f} | Config: {decoded}")

        # Elite selection (keep top 2)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        next_population = [population[sorted_indices[0]], population[sorted_indices[1]]]

        # Crossover & Mutation
        while len(next_population) < pop_size:
            p1 = population[np.random.choice(sorted_indices[:4])]
            p2 = population[np.random.choice(sorted_indices[:4])]
            
            # Crossover
            cx_point = np.random.randint(1, 4)
            child = np.concatenate([p1[:cx_point], p2[cx_point:]])
            
            # Mutation (15% chance per gene)
            for g_idx, key in enumerate(['units_l1', 'units_l2', 'lr', 'dropout']):
                if np.random.rand() < 0.15:
                    child[g_idx] = np.random.randint(len(GENE_SPACE[key]))
                    
            next_population.append(child)

        population = np.array(next_population)

    optimal_config = decode_nas_chromosome(best_individual)
    print(f"\n[NAS WINNER] Selected Topology: {optimal_config} (Fitness: {best_fitness:.4f})")
    return optimal_config

# =====================================================================
# 4. GA STAGE 2: MULTI-THRESHOLD / ALARM LOGIC OPTIMIZATION
# =====================================================================
# Optimize decision boundary: [th_fall (0.30 - 0.95), th_stillness (0.01 - 0.50)]
# Goal: Maximize Precision subject to Recall >= 0.95 on fall class
def evaluate_threshold_individual(chromosome, oof_probs, oof_features, oof_y_true):
    th_fall, th_stillness = chromosome[0], chromosome[1]

    # Condition: Class 1 model probability > th_fall AND stillness feature < th_stillness
    # (High stillness_std implies continuous motion; low value implies resting on ground)
    fall_preds = ((oof_probs[:, 1] >= th_fall) & (oof_features[:, 4] <= th_stillness)).astype(int)
    fall_true = (oof_y_true == 1).astype(int)

    rec = recall_score(fall_true, fall_preds, zero_division=0)
    prec = precision_score(fall_true, fall_preds, zero_division=0)

    # Severe penalty if recall constraint (0.95) is violated
    if rec < 0.95:
        fitness = prec - 10.0 * (0.95 - rec)
    else:
        fitness = prec

    return max(0.0, fitness)

def run_threshold_genetic_algorithm(oof_probs, oof_features, oof_y_true, pop_size=30, generations=25):
    print("\n" + "="*50)
    print("STARTING GA STAGE 2: ALARM LOGIC & MULTI-THRESHOLD OPTIMIZATION")
    print("="*50)

    # Continuous chromosome: [th_fall, th_stillness]
    population = np.column_stack([
        np.random.uniform(0.35, 0.90, size=pop_size),  # th_fall
        np.random.uniform(0.02, 0.40, size=pop_size)   # th_stillness
    ])

    best_fitness = -1.0
    best_chrom = population[0]

    for gen in range(generations):
        fitness_scores = np.array([
            evaluate_threshold_individual(ind, oof_probs, oof_features, oof_y_true)
            for ind in population
        ])

        gen_best_idx = np.argmax(fitness_scores)
        if fitness_scores[gen_best_idx] > best_fitness:
            best_fitness = fitness_scores[gen_best_idx]
            best_chrom = population[gen_best_idx].copy()

        # Selection & elitism
        sorted_idx = np.argsort(fitness_scores)[::-1]
        next_population = [population[sorted_idx[0]], population[sorted_idx[1]]]

        # Arithmetic Crossover & Gaussian Mutation
        while len(next_population) < pop_size:
            p1 = population[np.random.choice(sorted_idx[:8])]
            p2 = population[np.random.choice(sorted_idx[:8])]
            
            alpha = np.random.rand()
            child = alpha * p1 + (1.0 - alpha) * p2

            # Gaussian mutation
            if np.random.rand() < 0.25:
                child[0] += np.random.normal(0, 0.05)
                child[1] += np.random.normal(0, 0.02)

            # Enforce bounding
            child[0] = np.clip(child[0], 0.30, 0.95)
            child[1] = np.clip(child[1], 0.01, 0.50)
            next_population.append(child)

        population = np.array(next_population)

    # Final metrics calculation on the winner
    final_preds = ((oof_probs[:, 1] >= best_chrom[0]) & (oof_features[:, 4] <= best_chrom[1])).astype(int)
    fall_true = (oof_y_true == 1).astype(int)
    final_rec = recall_score(fall_true, final_preds, zero_division=0)
    final_prec = precision_score(fall_true, final_preds, zero_division=0)

    print(f"\n[ALARM LOGIC WINNER]")
    print(f"-> Optimal Fall Probability Cutoff (th_fall)     : {best_chrom[0]:.4f}")
    print(f"-> Optimal Max Stillness Threshold (th_stillness): {best_chrom[1]:.4f}")
    print(f"-> Resulting Fall Precision                      : {final_prec * 100:.2f}%")
    print(f"-> Resulting Fall Recall                         : {final_rec * 100:.2f}%")

    return {"th_fall": best_chrom[0], "th_stillness": best_chrom[1]}

# =====================================================================
# 5. MAIN EXECUTION PIPELINE
# =====================================================================
if __name__ == "__main__":
    normal_dir    = "dataset/0_normal"
    fall_dir      = "dataset/1_fall"
    stillness_dir = "dataset/2_stillness"

    X_normal, y_normal = extract_from_folder(normal_dir,    assigned_label=0)
    X_falls,  y_falls  = extract_from_folder(fall_dir,      assigned_label=1)
    X_still,  y_still  = extract_from_folder(stillness_dir, assigned_label=2)

    if len(X_normal) > 0 and len(X_falls) > 0 and len(X_still) > 0:
        X = np.vstack([X_normal, X_falls, X_still])
        y = np.concatenate([y_normal, y_falls, y_still])

        classes = np.unique(y)
        computed_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
        class_weights_dict = dict(zip(classes, computed_weights))

        # Fixed K-Fold split across entire search to maintain evaluation integrity
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # -------------------------------------------------------------
        # STEP 1: RUN GA ON NAS / HYPERPARAMETERS
        # -------------------------------------------------------------
        best_hparams = run_nas_genetic_algorithm(X, y, skf, class_weights_dict, pop_size=8, generations=5)

        # -------------------------------------------------------------
        # STEP 2: FULL 5-FOLD TRAINING WITH BEST HYPERPARAMETERS (OOF GATHERING)
        # -------------------------------------------------------------
        print("\n" + "="*50)
        print("TRAINING FULL CV WITH GA-SELECTED ARCHITECTURE")
        print("="*50)

        oof_probs = np.zeros((len(y), 3))
        oof_features = np.zeros_like(X)
        fold_f1_scores = []
        fold_auc_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_val = scaler.transform(X[val_idx])

            layers = [
                tf.keras.layers.Input(shape=(6,)),
                Dense(best_hparams['units_l1'], activation="relu")
            ]
            if best_hparams['dropout'] > 0:
                layers.append(Dropout(best_hparams['dropout']))
            if best_hparams['units_l2'] > 0:
                layers.append(Dense(best_hparams['units_l2'], activation="relu"))
            layers.append(Dense(3, activation="softmax"))

            cv_model = Sequential(layers)
            cv_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=best_hparams['lr']),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            cv_model.fit(
                X_train, y[train_idx],
                epochs=60,
                batch_size=16,
                verbose=0,
                class_weight=class_weights_dict
            )

            preds = cv_model(X_val, training=False).numpy()
            oof_probs[val_idx] = preds
            oof_features[val_idx] = X[val_idx]  # preserve raw features (for stillness_std)

            fold_f1 = f1_score(y[val_idx], np.argmax(preds, axis=1), average='macro', zero_division=0)
            fold_auc = roc_auc_score(y[val_idx], preds, multi_class='ovr')
            fold_f1_scores.append(fold_f1)
            fold_auc_scores.append(fold_auc)
            print(f"Fold {fold+1} | F1: {fold_f1:.4f} | ROC-AUC: {fold_auc:.4f}")

        print(f"\nMean CV F1: {np.mean(fold_f1_scores):.4f} | Mean CV ROC-AUC: {np.mean(fold_auc_scores):.4f}")

        # -------------------------------------------------------------
        # STEP 3: RUN GA ON MULTI-THRESHOLD / ALARM LOGIC
        # -------------------------------------------------------------
        best_thresholds = run_threshold_genetic_algorithm(
            oof_probs=oof_probs,
            oof_features=oof_features,
            oof_y_true=y,
            pop_size=30,
            generations=30
        )

        # -------------------------------------------------------------
        # STEP 4: FINAL RETRAIN ON 100% DATA & EXPORT
        # -------------------------------------------------------------
        print("\n" + "="*50)
        print("RETRAINING FINAL PRODUCTION MODEL ON 100% DATA")
        print("="*50)
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
        prod_model.fit(
            X_full_scaled, y,
            epochs=60,
            batch_size=16,
            verbose=0,
            class_weight=class_weights_dict
        )

        export_firmware_header(model=prod_model, scaler=final_scaler, best_thresholds=best_thresholds)
    else:
        print("Dataset directory error. Ensure dataset folders are populated.")
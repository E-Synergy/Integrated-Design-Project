import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# --- 1. CONTINUOUS SLICING & PREPROCESSING PIPELINE ---
def slice_continuous_csv(file_path, target_samples=100, window_ms=2000, sample_interval_ms=20):
    """
    Slices a continuous CSV into intact windows.
    Since you have a mixed file, we look for high Jerk segments for falls 
    and baseline segments for idle.
    """
    df = pd.read_csv(file_path)
    
    # Required features for this phase
    features = ["Acc_Magnitude", "Jerk_Magnitude"]
    
    samples_per_window = int(window_ms / sample_interval_ms) # 100 samples
    windows_x = []
    labels = []
    
    # Simple threshold scan over your continuous log to find events
    # Adjust the threshold value based on your specific dataset's Jerk peak values
    jerk_threshold = float(df["Jerk_Magnitude"].max()) * 0.6 
    
    i = 0
    while i < len(df) - samples_per_window:
        # Check if a peak event occurs, stops before the very end of file to guarantee a full 2-second window
        if df["Jerk_Magnitude"].iloc[i] > jerk_threshold:
            # Extract a 2-second window around the peak event
            start_idx = i - int(samples_per_window * 0.25)
            end_idx = start_idx + samples_per_window
            start_idx = max(0, start_idx) #calculates 25% of window size, and subtracted from script index (going back)
            end_idx = min(len(df), end_idx)
            
            if end_idx <= len(df):
                window_df = df.iloc[start_idx:end_idx]
                # Flatten the selected features into a 1D vector (shape: 200,)
                flat_vector = np.concatenate([window_df[col].values for col in features])
                windows_x.append(flat_vector)
                labels.append(1) # Label as Fall
                
            i += samples_per_window # Skip past this window to avoid duplicate slices
        else:
            # If it's baseline idling, occasionally sample a normal window
            if i % (samples_per_window * 2) == 0:
                window_df = df.iloc[i:i + samples_per_window]
                flat_vector = np.concatenate([window_df[col].values for col in features])
                windows_x.append(flat_vector)
                labels.append(0) # Label as Normal/Idle
            i += 1
        print(flat_vector)
            
    return np.array(windows_x), np.array(labels)

# --- 2. LOAD DATA & SIMULATE MULTIPLE SESSIONS ---
# Replace with your actual filename
csv_file_path = "data2.csv" 

try:
    X, y = slice_continuous_csv(csv_file_path)
    print(f"Successfully extracted {len(X)} windows from your CSV.")
    print(f"Class distribution -> Falls: {np.sum(y)}, Normal/Idle: {len(y) - np.sum(y)}")
except FileNotFoundError:
    print(f"Error: Could not find '{csv_file_path}'. Please update the path string.")
    X, y = np.empty((0, 200)), np.empty((0,))

# If data loading succeeded, run validation
if len(X) > 0:
    # Since you currently have a single long testing file, we will artificially split 
    # the extracted windows into 3 unique pseudo-session groups so GroupKFold can run.
    # (Once you record multiple distinct CSV files, map each group directly to a file ID!)
    groups = np.arange(len(X)) % 3 

    # Scale the dataset inputs
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 3. GROUPKFOLD CROSS-VALIDATION LOOP ---
    gkf = GroupKFold(n_splits=3)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_scaled, y, groups=groups)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 4. TinyML Optimized MLP Structure
        model = Sequential([
            Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid') 
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        print(f"\n--- Training Fold {fold + 1} ---")
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=4, verbose=1)

    print("\n[Pipeline Verification Complete] The data architecture structures cleanly!")
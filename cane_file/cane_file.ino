#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include "scaler_params.h"

Adafruit_MPU6050 mpu;

// --- TIMING & BUFFER CONSTANTS ---
const long BAUD_RATE = 115200;
const unsigned long SAMPLE_INTERVAL_MS = 20; // 50 Hz sampling
const int WINDOW_SAMPLES = 100;              // 100 samples = 2.0s context
const int STRIDE_SAMPLES = 20;               // 20 samples = evaluate every 400ms

// --- ROLLING BUFFER STRUCTURE ---
struct IMUSample {
    float ax, ay, az;
    float acc_mag;
    float jerk_mag;
};

IMUSample ringBuffer[WINDOW_SAMPLES];
int bufferHead = 0;
int totalSamplesCollected = 0;
int strideCounter = 0;

unsigned long lastSampleTime = 0;
float prevAccMag = 0.0f;

// --- FSM & VOTING DEFINITIONS ---
enum CaneState {
    STATE_WALKING_MONITOR,
    STATE_FALL_SUSPECTED,
    STATE_STILLNESS_CONFIRM,
    STATE_ALARM
};

CaneState currentState = STATE_WALKING_MONITOR;

int leakyBucket = 0;
const int BUCKET_CAPACITY = 100;
const int BUCKET_TRIGGER_THRESHOLD = 70;
unsigned long stillnessTimer = 0;

// --- MODEL PREDICTION OUTPUT STRUCT ---
struct ModelOutput {
    float p_normal;
    float p_fall;
    float p_still;
};

// --- STUB FUNCTION FOR FUTURE IOT / BUZZER ACTIONS ---
void triggerAlarmActions() {
    Serial.println(">>> [ALARM ACTIVE] Emergency Fall Triggered! <<<");
    // Place future BLE notification, MQTT payload, or buzzer pin logic here
}

// =====================================================================
// NATIVE C++ FORWARD PASS (NO TFLM DEPENDENCIES)
// =====================================================================
ModelOutput predict_fall(const float raw_features[6]) {
    // 1. Z-Score Standardization
    float x_scaled[INPUT_DIM];
    for (int i = 0; i < INPUT_DIM; i++) {
        x_scaled[i] = (raw_features[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
    }

    // 2. Layer 1: Dense (6 -> L1_UNITS, ReLU)
    float h1[L1_UNITS];
    for (int j = 0; j < L1_UNITS; j++) {
        float sum = BIAS_L1[j];
        for (int i = 0; i < INPUT_DIM; i++) {
            sum += x_scaled[i] * WEIGHTS_L1[i][j];
        }
        h1[j] = (sum > 0.0f) ? sum : 0.0f; // ReLU activation
    }

    // 3. Layer 2: Output Dense (L1_UNITS -> 3, Softmax)
    float logits[OUTPUT_DIM];
    float max_logit = -1e9f;
    for (int k = 0; k < OUTPUT_DIM; k++) {
        float sum = BIAS_L2[k];
        for (int j = 0; j < L1_UNITS; j++) {
            sum += h1[j] * WEIGHTS_L2[j][k];
        }
        logits[k] = sum;
        if (logits[k] > max_logit) max_logit = logits[k];
    }

    float sum_exp = 0.0f;
    float exp_logits[OUTPUT_DIM];
    for (int k = 0; k < OUTPUT_DIM; k++) {
        exp_logits[k] = expf(logits[k] - max_logit);
        sum_exp += exp_logits[k];
    }

    ModelOutput out;
    out.p_normal = exp_logits[0] / sum_exp;
    out.p_fall   = exp_logits[1] / sum_exp;
    out.p_still  = exp_logits[2] / sum_exp;
    return out;
}

// =====================================================================
// FEATURE EXTRACTION MATCHING PYTHON WINDOW LOGIC
// =====================================================================
void extractFeatures(float features_out[6]) {
    float max_acc = -1e9f;
    float min_acc = 1e9f;
    float sum_acc = 0.0f;
    float max_jerk = -1e9f;
    float max_ay = -1e9f;
    float min_ay = 1e9f;

    for (int i = 0; i < WINDOW_SAMPLES; i++) {
        int idx = (bufferHead + i) % WINDOW_SAMPLES;
        float mag = ringBuffer[idx].acc_mag;
        float jerk = ringBuffer[idx].jerk_mag;
        float ay = ringBuffer[idx].ay;

        if (mag > max_acc) max_acc = mag;
        if (mag < min_acc) min_acc = mag;
        sum_acc += mag;

        if (jerk > max_jerk) max_jerk = jerk;
        if (ay > max_ay) max_ay = ay;
        if (ay < min_ay) min_ay = ay;
    }

    float mean_acc = sum_acc / WINDOW_SAMPLES;
    float var_acc = 0.0f;
    for (int i = 0; i < WINDOW_SAMPLES; i++) {
        int idx = (bufferHead + i) % WINDOW_SAMPLES;
        float diff = ringBuffer[idx].acc_mag - mean_acc;
        var_acc += diff * diff;
    }
    float std_acc = sqrtf(var_acc / (WINDOW_SAMPLES - 1));

    // Tail end 25 samples (~500ms) for stillness_std
    float sum_tail = 0.0f;
    for (int i = WINDOW_SAMPLES - 25; i < WINDOW_SAMPLES; i++) {
        int idx = (bufferHead + i) % WINDOW_SAMPLES;
        sum_tail += ringBuffer[idx].acc_mag;
    }
    float mean_tail = sum_tail / 25.0f;
    float var_tail = 0.0f;
    for (int i = WINDOW_SAMPLES - 25; i < WINDOW_SAMPLES; i++) {
        int idx = (bufferHead + i) % WINDOW_SAMPLES;
        float diff = ringBuffer[idx].acc_mag - mean_tail;
        var_tail += diff * diff;
    }
    float stillness_std = sqrtf(var_tail / 24.0f);
    float max_y_tilt = max_ay - min_ay;

    features_out[0] = max_acc;
    features_out[1] = min_acc;
    features_out[2] = std_acc;
    features_out[3] = max_jerk;
    features_out[4] = stillness_std;
    features_out[5] = max_y_tilt;
}

// =====================================================================
// SETUP & MAIN LOOP
// =====================================================================
void setup() {
    Serial.begin(BAUD_RATE);
    while (!Serial) delay(10);

    Wire.begin();
    if (!mpu.begin()) {
        Serial.println("Failed to find MPU6050 chip! Halting.");
        while (1) delay(10);
    }

    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

    Serial.println("MPU6050 Initialized at 50Hz.");
    Serial.println("Filling circular buffer (100 samples required)...");
}

void loop() {
    unsigned long now = millis();
    if (now - lastSampleTime < SAMPLE_INTERVAL_MS) {
        return;
    }
    lastSampleTime = now;

    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    // Acc_Magnitude in m/s^2
    float acc_mag = sqrtf(a.acceleration.x * a.acceleration.x +
                          a.acceleration.y * a.acceleration.y +
                          a.acceleration.z * a.acceleration.z);

    // Approximate Jerk Magnitude (d(Acc)/dt in units/sample)
    float jerk_mag = fabsf(acc_mag - prevAccMag);
    prevAccMag = acc_mag;

    // Push into ring buffer
    ringBuffer[bufferHead].ax = a.acceleration.x;
    ringBuffer[bufferHead].ay = a.acceleration.y;
    ringBuffer[bufferHead].az = a.acceleration.z;
    ringBuffer[bufferHead].acc_mag = acc_mag;
    ringBuffer[bufferHead].jerk_mag = jerk_mag;
    bufferHead = (bufferHead + 1) % WINDOW_SAMPLES;

    if (totalSamplesCollected < WINDOW_SAMPLES) {
        totalSamplesCollected++;
        if (totalSamplesCollected % 20 == 0) {
            Serial.printf("[Buffer Warmup] %d/%d samples cached\n", totalSamplesCollected, WINDOW_SAMPLES);
        }
        return;
    }

    // Step every 20 samples (400ms)
    strideCounter++;
    if (strideCounter < STRIDE_SAMPLES) {
        return;
    }
    strideCounter = 0;

    // Extract 6 features and run C++ forward pass
    float features[6];
    extractFeatures(features);
    ModelOutput preds = predict_fall(features);

    // Multi-Class Leaky Bucket Accumulator
    if (preds.p_fall > preds.p_normal && preds.p_fall >= 0.50f) {
        leakyBucket += 35;
    } else if (preds.p_normal > 0.50f) {
        leakyBucket -= 25;
    } else if (preds.p_still > 0.65f) {
        leakyBucket -= 10;
    }
    leakyBucket = constrain(leakyBucket, 0, BUCKET_CAPACITY);

    // Periodic Console Diagnostics
    Serial.printf("[INFERENCE] P[Norm: %.2f | Fall: %.2f | Still: %.2f] | Bucket: %d/100 | State: %d\n",
                  preds.p_normal, preds.p_fall, preds.p_still, leakyBucket, currentState);

    // Finite State Machine
    switch (currentState) {
        case STATE_WALKING_MONITOR:
            if (leakyBucket >= BUCKET_TRIGGER_THRESHOLD) {
                Serial.println("-> Fall Dynamics Detected. Checking for Stillness...");
                stillnessTimer = millis();
                currentState = STATE_STILLNESS_CONFIRM;
            }
            break;

        case STATE_STILLNESS_CONFIRM:
            // If active walking resumes, cancel alarm
            if (preds.p_normal > 0.65f) {
                Serial.println("-> Normal gait resumed. Aborting trigger.");
                leakyBucket = 0;
                currentState = STATE_WALKING_MONITOR;
                break;
            }

            // Verify cane remains static for 3 consecutive seconds
            if (millis() - stillnessTimer >= 3000) {
                currentState = STATE_ALARM;
            }
            break;

        case STATE_ALARM:
            triggerAlarmActions();
            // Stays in alarm state until hardware reset or custom reset logic
            break;
    }
}
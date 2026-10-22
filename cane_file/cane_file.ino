#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include "model_params.h"
#include <math.h>

Adafruit_MPU6050 mpu;

// --- TIMING & BUFFER CONSTANTS ---
const long BAUD_RATE = 115200;
const unsigned long SAMPLE_INTERVAL_MS = 20; // 50 Hz sampling
const int WINDOW_SAMPLES = 100;              // 100 samples = 2.0s context
const int STRIDE_SAMPLES = 20;               // 20 samples = evaluate every 400ms

// --- ROLLING BUFFER STRUCTURE ---
struct IMUSample {
    float ax, ay, az;      // in milli-G (mg)
    float acc_mag;         // in milli-G (mg)
    float jerk_mag;        // in mg/s
};

IMUSample ringBuffer[WINDOW_SAMPLES];
int bufferHead = 0;
int totalSamplesCollected = 0;
int strideCounter = 0;

unsigned long lastSampleTime = 0;
float prevAcc_mg[3] = {0.0f, 0.0f, 0.0f};
bool hasPreviousAccSample = false;
const float GRAVITY_MS2 = 9.80665f;

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
// Model confidence thresholds
const float FALL_THRESHOLD = 0.50f;
const float NORMAL_THRESHOLD = 0.65f;
const float STILL_THRESHOLD = 0.65f;

// FSM timing
const unsigned long MAX_STILLNESS_WAIT_MS = 10000UL; // 10 seconds
const unsigned long STILLNESS_REQUIRED_MS = 3000UL;  // Provisional; tune in testing

unsigned long suspicionStartTime = 0;
unsigned long stillnessAccumulatedMs = 0;
unsigned long lastStillnessUpdate = 0;

bool previousStillConfidence = false;

// Alarm and button handling
const int RESET_BUTTON_PIN = 25; // Button between GPIO25 and GND
bool alarmActionTriggered = false;

bool lastButtonReading = HIGH;
bool stableButtonState = HIGH;
unsigned long lastButtonChangeTime = 0;
const unsigned long BUTTON_DEBOUNCE_MS = 40UL;

// --- MODEL PREDICTION OUTPUT STRUCT ---
struct ModelOutput {
    float p_normal;
    float p_fall;
    float p_still;
};

// --- STUB FUNCTION FOR FUTURE IOT / BUZZER ACTIONS ---
void triggerAlarmActions() {
    Serial.println("\n>>> [ALARM ACTIVE] Emergency Fall Triggered! <<<\n");
    // Place future BLE notification, MQTT payload, or buzzer pin logic here

    
}

bool readPhysicalResetButton() {
    bool reading = digitalRead(RESET_BUTTON_PIN);
    unsigned long now = millis();

    if (reading != lastButtonReading) {
        lastButtonChangeTime = now;
        lastButtonReading = reading;
    }

    if (now - lastButtonChangeTime >= BUTTON_DEBOUNCE_MS) {
        if (reading != stableButtonState) {
            stableButtonState = reading;

            // INPUT_PULLUP: LOW means pressed.
            if (stableButtonState == LOW) {
                return true;
            }
        }
    }

    return false;
}

// Integration point for your authenticated MQTT/BLE handler.
// Return true only when an authorized reset command has been
// received and validated. Do not trust arbitrary network messages.
bool consumeAuthorizedRemoteReset() {
    // TODO: Connect to your authenticated remote-command handler.
    return false;
}

void clearFallConfirmation() {
    leakyBucket = 0;
    suspicionStartTime = 0;
    stillnessAccumulatedMs = 0;
    lastStillnessUpdate = 0;
    previousStillConfidence = false;
}

void enterAlarm() {
    currentState = STATE_ALARM;

    if (!alarmActionTriggered) {
        alarmActionTriggered = true;
        triggerAlarmActions();
    }
}

void resetFromAlarm() {
    if (currentState != STATE_ALARM) {
        return;
    }

    alarmActionTriggered = false;
    clearFallConfirmation();
    currentState = STATE_WALKING_MONITOR;

    Serial.println("-----> Alarm reset. Returning to monitoring.");
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

    Serial.printf("LOGITS -> L0 (Normal): %.2f | L1 (Fall): %.2f | L2 (Still): %.2f\n", 
              logits[0], logits[1], logits[2]);
              
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

     pinMode(RESET_BUTTON_PIN, INPUT_PULLUP);

    Wire.begin();
    if (!mpu.begin()) {
        Serial.println("Failed to find MPU6050 chip! Halting.");
        while (1) delay(10);
    }

    mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

    Serial.println("MPU6050 Initialized at 50Hz.");
    Serial.println("Filling circular buffer (100 samples required)...");
}


void loop() {
    // --------------------------------------------------
    // 1. Handle alarm reset inputs
    // --------------------------------------------------
    bool physicalResetPressed = readPhysicalResetButton();
    bool remoteResetRequested = consumeAuthorizedRemoteReset();

    if (currentState == STATE_ALARM &&
        (physicalResetPressed || remoteResetRequested)) {
        resetFromAlarm();
    }

    // --------------------------------------------------
    // 2. Maintain the sensor sampling interval (50 Hz)
    // --------------------------------------------------
unsigned long now = millis();
    if (now - lastSampleTime < SAMPLE_INTERVAL_MS) {
        return;
    }

    float dt = (float)(now - lastSampleTime) / 1000.0f;
    lastSampleTime = now;

    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    // Convert raw m/s^2 readings to milli-G (mg)
    float ax_mg = (a.acceleration.x / GRAVITY_MS2) * 1000.0f;
    float ay_mg = (a.acceleration.y / GRAVITY_MS2) * 1000.0f;
    float az_mg = (a.acceleration.z / GRAVITY_MS2) * 1000.0f;

    // 3D Acceleration Magnitude in mg
    float acc_mag = sqrtf(ax_mg * ax_mg + ay_mg * ay_mg + az_mg * az_mg);

    // 3D Resultant Jerk in mg/s matching your data logger
    float jerk_mag = 0.0f;
    if (hasPreviousAccSample && dt > 0.0f) {
        float jx = (ax_mg - prevAcc_mg[0]) / dt;
        float jy = (ay_mg - prevAcc_mg[1]) / dt;
        float jz = (az_mg - prevAcc_mg[2]) / dt;
        jerk_mag = sqrtf(jx * jx + jy * jy + jz * jz);
    } else {
        hasPreviousAccSample = true;
    }

    prevAcc_mg[0] = ax_mg;
    prevAcc_mg[1] = ay_mg;
    prevAcc_mg[2] = az_mg;

    // --------------------------------------------------
    // 3. Update the circular sensor buffer
    // --------------------------------------------------
    ringBuffer[bufferHead].ax = ax_mg;
    ringBuffer[bufferHead].ay = ay_mg;
    ringBuffer[bufferHead].az = az_mg;
    ringBuffer[bufferHead].acc_mag = acc_mag;
    ringBuffer[bufferHead].jerk_mag = jerk_mag;

    bufferHead = (bufferHead + 1) % WINDOW_SAMPLES;

    // Wait until the first complete 2-second window exists
    if (totalSamplesCollected < WINDOW_SAMPLES) {
        totalSamplesCollected++;

        if (totalSamplesCollected % 20 == 0) {
            Serial.printf(
                "[Buffer Warmup] %d/%d samples cached\n",
                totalSamplesCollected,
                WINDOW_SAMPLES
            );
        }

        return;
    }

    // Run inference once every 20 new samples (400 ms)
    strideCounter++;

    if (strideCounter < STRIDE_SAMPLES) {
        return;
    }

    strideCounter = 0;

    // --------------------------------------------------
    // 4. Extract features and run the neural network
    // --------------------------------------------------
    float features[6];
    extractFeatures(features);
    Serial.printf("EXTRACTED -> max_acc: %.2f, min_acc: %.2f, std_acc: %.2f, jerk: %.2f, still_std: %.2f, y_tilt: %.2f\n",
              features[0], features[1], features[2], features[3], features[4], features[5]);
    ModelOutput preds = predict_fall(features);

    unsigned long fsmNow = millis();

    // --------------------------------------------------
    // 5. Update leaky bucket only during normal monitoring
    // --------------------------------------------------
    if (currentState == STATE_WALKING_MONITOR) {
        if (preds.p_fall > preds.p_normal &&
            preds.p_fall >= FALL_THRESHOLD) {
            leakyBucket += 35;
        }
        else if (preds.p_normal > 0.50f) {
            leakyBucket -= 25;
        }
        else if (preds.p_still > STILL_THRESHOLD) {
            leakyBucket -= 10;
        }

        leakyBucket = constrain(
            leakyBucket,
            0,
            BUCKET_CAPACITY
        );
    }

    // --------------------------------------------------
    // 6. Print inference diagnostics
    // --------------------------------------------------
    Serial.printf(
        "[INFERENCE] P[Normal: %.2f | Fall: %.2f | Still: %.2f] "
        "| Bucket: %d/%d | State: %d\n",
        preds.p_normal,
        preds.p_fall,
        preds.p_still,
        leakyBucket,
        BUCKET_CAPACITY,
        currentState
    );

    // --------------------------------------------------
    // 7. Finite State Machine
    // --------------------------------------------------
    switch (currentState) {

        // ----------------------------------------------
        // STATE 1: Normal monitoring
        // ----------------------------------------------
        case STATE_WALKING_MONITOR:

            if (leakyBucket >= BUCKET_TRIGGER_THRESHOLD) {
                suspicionStartTime = fsmNow;
                stillnessAccumulatedMs = 0;
                lastStillnessUpdate = 0;
                previousStillConfidence = false;

                currentState = STATE_FALL_SUSPECTED;

                Serial.println(
                    "-----> Fall suspected. Waiting for stillness."
                );
            }

            break;


        // ----------------------------------------------
        // STATE 2: Fall suspected, waiting for stillness
        // ----------------------------------------------
        case STATE_FALL_SUSPECTED:

            // Cancel if confident normal activity resumes
            if (preds.p_normal > NORMAL_THRESHOLD) {
                Serial.println(
                    "\n-----> Normal activity detected. Cancelling suspicion."
                );

                clearFallConfirmation();
                currentState = STATE_WALKING_MONITOR;
                break;
            }

            // Give up if stillness is not detected within 10 seconds
            if (fsmNow - suspicionStartTime >=
                MAX_STILLNESS_WAIT_MS) {

                Serial.println(
                    "\n-----> Stillness timeout. Returning to monitoring."
                );

                clearFallConfirmation();
                currentState = STATE_WALKING_MONITOR;
                break;
            }

            // Start confirmation when stillness is detected
            if (preds.p_still > STILL_THRESHOLD) {
                stillnessAccumulatedMs = 0;
                lastStillnessUpdate = fsmNow;
                previousStillConfidence = true;

                currentState = STATE_STILLNESS_CONFIRM;

                Serial.println(
                    "\n-----> Stillness detected. Starting confirmation."
                );
            }

            break;


        // ----------------------------------------------
        // STATE 3: Confirm sustained stillness
        // ----------------------------------------------
        case STATE_STILLNESS_CONFIRM:

            // Cancel if confident normal activity resumes
            if (preds.p_normal > NORMAL_THRESHOLD) {
                Serial.println(
                    "\n-----> Normal activity resumed. Cancelling alarm."
                );

                clearFallConfirmation();
                currentState = STATE_WALKING_MONITOR;
                break;
            }

            // Overall timeout applies even if confirmation pauses
            if (fsmNow - suspicionStartTime >=
                MAX_STILLNESS_WAIT_MS) {

                Serial.println(
                    "\n-----> Maximum wait exceeded. Returning to monitoring."
                );

                clearFallConfirmation();
                currentState = STATE_WALKING_MONITOR;
                break;
            }

            if (preds.p_still > STILL_THRESHOLD) {

                // Accumulate time only across consecutive
                // stillness-positive inference windows
                if (previousStillConfidence) {
                    stillnessAccumulatedMs +=
                        fsmNow - lastStillnessUpdate;
                }

                previousStillConfidence = true;
                lastStillnessUpdate = fsmNow;

                Serial.printf(
                    "[STILLNESS] Confirmed: %lu / %lu ms\n",
                    stillnessAccumulatedMs,
                    STILLNESS_REQUIRED_MS
                );

                // Alarm after sufficient accumulated stillness
                if (stillnessAccumulatedMs >=
                    STILLNESS_REQUIRED_MS) {

                    Serial.println(
                        "\n-----> Sustained stillness confirmed. ALARM!"
                    );

                    enterAlarm();
                }

            } else {

                // Pause confirmation without losing accumulated time
                previousStillConfidence = false;
                lastStillnessUpdate = fsmNow;

                Serial.println(
                    "\n-----> Stillness interrupted. Confirmation paused."
                );
            }

            break;


        // ----------------------------------------------
        // STATE 4: Alarm latched
        // ----------------------------------------------
        case STATE_ALARM:

            // enterAlarm() triggers alarm actions only once.
            // Wait for physical or authorized remote reset.
            break;
    }
}

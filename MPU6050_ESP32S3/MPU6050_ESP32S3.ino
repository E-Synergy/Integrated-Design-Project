#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

#define SDA 8
#define SCL 9

const long BAUD_RATE = 115200;
Adafruit_MPU6050 mpu;

// Timing parameters for a stable 50Hz (20ms) polling window
unsigned long lastTime = 0;
const unsigned long sampleInterval = 20;
const float GRAVITY_MS2 = SENSORS_GRAVITY_STANDARD; // ~9.80665 m/s^2

// Global C arrays to track physical dimensions [0]=X, [1]=Y, [2]=Z
float prevAcc[3] = {0.0, 0.0, 0.0};
float currentAcc[3];
float axialJerk[3];  

// Convert m/s^2 array elements to milli-G (mg) values
void convertToMg(const float raw_ms2[3], float out_mg[3]) {
  for (int i = 0; i < 3; i++) {
    out_mg[i] = (raw_ms2[i] / GRAVITY_MS2) * 1000.0;
  }
}

// Calculate directional jerk vectors and return the resultant 3D magnitude
float calculateJerk(const float current_mg[3], const float prev_mg[3], float dt, float out_jerk[3]) {
  float sumSq = 0.0;
  for (int i = 0; i < 3; i++) {
    out_jerk[i] = (current_mg[i] - prev_mg[i]) / dt;
    sumSq += out_jerk[i] * out_jerk[i];
  }
  return sqrt(sumSq);
}

// Calculate the 3D vector resultant magnitude of acceleration
float calculateAcc(const float acc_mg[3]) {
  float sumSq = 0.0;
  for (int i = 0; i < 3; i++) {
    sumSq += acc_mg[i] * acc_mg[i];
  }
  return sqrt(sumSq);
}

// Shift array parameters forward in memory to update history
void shift_memory(float dest_array[3], const float src_array[3]) {
  for (int i = 0; i < 3; i++) {
    dest_array[i] = src_array[i];
  }
}

void mpu_settings() {
  auto AccelRange = MPU6050_RANGE_2_G;
  auto GyroRange = MPU6050_RANGE_500_DEG;
  auto Bandwidth = MPU6050_BAND_21_HZ; // Removes vibration noise

  mpu.setAccelerometerRange(AccelRange);
  mpu.setGyroRange(GyroRange);
  mpu.setFilterBandwidth(Bandwidth);

  Serial.printf("Accelerometer Range: %d\n", AccelRange);
  Serial.printf("Gyroscope Range: %d\n", GyroRange);
  Serial.printf("Bandwidth: %d\n", Bandwidth);
}

void jerk_calculation() {
  // Non-blocking execution loop locked to exactly 20ms windows
  if (millis() - lastTime >= sampleInterval) {
    float dt = (float)(millis() - lastTime) / 1000.0; // Dynamic dt delta
    lastTime = millis();

    // 1. Fetch single fresh sensor event
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp); 
    
    // 2. Wrap the dynamic variables into a temporary raw local array
    float raw[3] = {a.acceleration.x, a.acceleration.y, a.acceleration.z};
    
    // 3. Populate currentAcc array elements with mg data
    convertToMg(raw, currentAcc);
    
    // 4. Compute Vector Magnitude from current mg properties
    float accMagnitude = calculateAcc(currentAcc);
    
    // 5. Compute true jerk against frozen historical prevAcc registers
    float jerkMagnitude = calculateJerk(currentAcc, prevAcc, dt, axialJerk);
    
    // 6. Stream clean, comma-separated CSV formatting for Edge Impulse ingestion
    Serial.print(currentAcc[0]);   Serial.print(","); // X-axis mg
    Serial.print(currentAcc[1]);   Serial.print(","); // Y-axis mg
    Serial.print(currentAcc[2]);   Serial.print(","); // Z-axis mg
    Serial.print(accMagnitude);    Serial.print(","); // Total Acceleration Magnitude
    Serial.println(jerkMagnitude);                    // True Resultant Jerk Magnitude
    
    // Optional directional jerk output lines if your model requires them:
    // Serial.print(axialJerk[0]); Serial.print(",");
    // Serial.print(axialJerk[1]); Serial.print(",");
    // Serial.println(axialJerk[2]);

    // 7. Advance array states to historical registers for the next step iteration
    shift_memory(prevAcc, currentAcc);
  }
}

void setup() {
  Serial.begin(BAUD_RATE);
  while (!Serial) { delay(10); }

  Serial.println("Adafruit MPU6050 Fall Detection Framework");
  Wire.begin(SDA, SCL);

  while (!mpu.begin()) {
    Serial.println("Failed to detect MPU6050 chip over I2C!");
    delay(100);
  }
  Serial.println("MPU6050 Initialised Successfully.");
  
  mpu_settings();

  // Priming the historical array data with standard gravity to avoid a fake startup spike
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  float raw_start[3] = {a.acceleration.x, a.acceleration.y, a.acceleration.z};
  convertToMg(raw_start, prevAcc);
  
  lastTime = millis();
}

void loop() {
  jerk_calculation(); // Let the internal 20ms block smoothly pacing execution
}
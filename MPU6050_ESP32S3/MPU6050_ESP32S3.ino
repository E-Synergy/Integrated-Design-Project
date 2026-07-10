#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#define SDA 8
#define SCL 9

const long BAUD_RATE = 115200;
Adafruit_MPU6050 mpu;

unsigned long lastTime = 0;
const unsigned long sampleInterval= 20;
const float GRAVITY_MS2 = SENSORS_GRAVITY_STANDARD;

float prevAccX = 0.0;
float prevAccY = 0.0;
float prevAccZ = 0.0;

void jerk_calculation() {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp); 

  prevAccX = (a.acceleration.x / GRAVITY_MS2) * 1000.0;
  prevAccY = (a.acceleration.y / GRAVITY_MS2) * 1000.0;
  prevAccZ = (a.acceleration.z / GRAVITY_MS2) * 1000.0;

  if(millis() - lastTime >= sampleInterval) {
    float dt = (millis() - lastTime) / 1000;    //compute the small change in difference
    lastTime = millis();

    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp); 

    float accX_mg = (a.acceleration.x / GRAVITY_MS2) * 1000.0;
    float accY_mg = (a.acceleration.y / GRAVITY_MS2) * 1000.0;
    float accZ_mg = (a.acceleration.z / GRAVITY_MS2) * 1000.0;

    float accMagnitude = sqrt(accX_mg * accX_mg + accY_mg * accY_mg + accZ_mg * accZ_mg);

    float jerkX = (accX_mg - prevAccX) / dt;
    float jerkY = (accY_mg - prevAccY) / dt;
    float jerkZ = (accZ_mg - prevAccZ) / dt;
    float jerkMagnitude = sqrt(jerkX * jerkX + jerkY * jerkY + jerkZ * jerkZ);

    prevAccX = accX_mg;
    prevAccY = accY_mg;
    prevAccZ = accZ_mg;

    Serial.print(accX_mg);       Serial.print(",");
    Serial.print(accY_mg);       Serial.print(",");
    Serial.print(accZ_mg);       Serial.print(",");
    Serial.print(accMagnitude);  Serial.print(",");
    Serial.println(jerkMagnitude);
  }
}

void mpu_settings() {
  auto AccelRange = MPU6050_RANGE_2_G;
  auto GyroRange = MPU6050_RANGE_500_DEG;
  auto Bandwidth = MPU6050_BAND_21_HZ;

  mpu.setAccelerometerRange(AccelRange);
  mpu.setGyroRange(GyroRange);
  mpu.setFilterBandwidth(Bandwidth);

  Serial.printf("Accelerometer Range: %d\n", AccelRange);
  Serial.printf("Gyroscope Range: %d\n", GyroRange);
  Serial.printf("Bandwidth: %d\n", Bandwidth);
}

void setup() {
 Serial.begin(BAUD_RATE);
 while (!Serial) {
  delay(10);
 }

 Serial.println("Adafruit MPU6050 test");
 Wire.begin(SDA, SCL);

 while(!mpu.begin()) {
  Serial.println("Failed to detect");
  delay(100);
 }

  Serial.println("MPU Found");
  mpu_settings();
  jerk_calculation();
  lastTime = millis();
}

void loop() {
  jerk_calculation();
  Serial.println("");
  delay(500);

}

#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

#define SDA 8
#define SCL 9

Adafruit_MPU6050 mpu;
const long BAUD_RATE = 115200;

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

void sendMPUData() {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  // Print Acceleration (X, Y, Z)
  Serial.print(a.acceleration.x);
  Serial.print(",");
  Serial.print(a.acceleration.y);
  Serial.print(",");
  Serial.print(a.acceleration.z);
  Serial.print(",");

  // Print Rotation (X, Y, Z)
  Serial.print(g.gyro.x);
  Serial.print(",");
  Serial.print(g.gyro.y);
  Serial.print(",");
  Serial.println(g.gyro.z); // println adds the hidden newline character at the end
}

void setup() {
  Serial.begin(BAUD_RATE);
  Serial.println("Adafruit MPU6050 test");
  Wire.begin(SDA, SCL);

  while(!mpu.begin()) {
    Serial.println("Failed to detect");
    delay(100);
  }

    Serial.println("MPU Found");
    mpu_settings();
    delay(100);
}

void loop() {
  sendMPUData();
  delay(500);
}
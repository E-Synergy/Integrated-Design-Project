#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#define SDA 8
#define SCL 9

const long BAUD_RATE = 115200;
Adafruit_MPU6050 mpu;

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
  delay(100);
  
}

void loop() {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  Serial.print("Acceleration X: ");
  Serial.print(a.acceleration.x);
  Serial.print(", Y: ");
  Serial.print(a.acceleration.y);
  Serial.print(", Z: ");
  Serial.print(a.acceleration.z);
  Serial.println(" m/s^2");

  Serial.print("Rotation X: ");
  Serial.print(g.gyro.x);
  Serial.print(", Y: ");
  Serial.print(g.gyro.y);
  Serial.print(", Z: ");
  Serial.print(g.gyro.z);
  Serial.println(" rad/s");

  Serial.println("");
  delay(500);

}

# Integrated-Design-Project
Smart Walking Cane with main focus on elderly assistance.

For MPU6050 Accelerometer and IMU
- Communication Protocol Used: I2C SCL, SDA
- Libraries to install:
- Remember to configure the file `C:\Users\[CHANGE_TO_YOUR_USER]\Documents\Arduino\libraries\Adafruit_MPU6050\Adafruit_MPU6050.h`
    - Change this particular line: `#define MPU6050_DEVICE_ID 0x68` to `#define MPU6050_DEVICE_ID 0x70`
- Run the .ino file

For Python 
- Create a virtual environment for best dependency control: `python -m venv .venv`
- Run the virtual enviornment using command prompt (can be also used in VSCode); but do remember to navigate to the project directory or root folder.
    CMD: `.venv\Scripts\activate.bat`
    PowerShell: `.venv\SCripts\Activate.ps1`
- Run `python -m pip install upgrade pip setuptools wheel` in the virtual environment.
    P/S: It is indicated that we are located in the virtual environment through the `(.venv) C:\Users\[CHANGE_TO_YOUR_USER]\Downloads\Integrated-Design-Project>` command in the CLI.
- Run `pip install pip-tools`
- Run `pip-sync`
- P/S: If adding a new library, please edit accordingly in 'requirements.in' then run `pip-compile requirements.in`

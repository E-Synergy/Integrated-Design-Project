import serial
import csv

arduino_port = "COM8"
baud_rate = 115200
filename = "data.csv"

def process_sensor_data(ser, writer, file):
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8').strip()
        if line.count(',') == 5:
            data_list = line.split(',')
            writer.writerow(data_list)
            file.flush() # Force save to disk
            print(f"Logged Accel (m/s^2) & Gyro (rad/s): {data_list}")
        else:
            print(f"Ignored non-data text: {line}")

def write(ser, file):
    writer = csv.writer(file)
    headers = ["Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"]
    writer.writerow(headers)

    print(f'Start logging')

    try:
        while True:
            process_sensor_data(ser, writer, file)

    except KeyboardInterrupt:
        print("\nLogging stopped by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ser.close()
        print("Serial port closed.")

try:
    ser = serial.Serial(arduino_port, baud_rate)
    print(f'Connected to Arduino port: { arduino_port }')
except Exception as e:
    print(f'Error: { e }')
    exit()

with open(filename, "a", newline = '') as file:
    #with statement isolates memory and by having an explicit statement to close the block further on, memory leak does not occur.
    #newline parameter just to address CRLF preferences; important for csv files.
    #"a" points to append and creates new file if not exists, and preserves existing data if exists
    write(file=file, ser=ser)


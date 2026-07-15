import serial
import csv
import time 

arduino_port = "COM8"
baud_rate = 115200
filename = "data.csv"
headers = ["Date", "Time", "Accel_X", "Accel_Y", "Accel_Z", "Acc_Magnitude", "Jerk_Magnitude"]

def process_sensor_data(ser, writer, file):
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8').strip()
        if line.count(',') == (len(headers) - 3):
            data_list = line.split(',')

            current_date = time.strftime('%Y-%m-%d')
            current_time = time.strftime('%H:%M:%S')

            first_index = 0
            data_list.insert(first_index, current_time)
            data_list.insert(first_index, current_date)

            writer.writerow(data_list)
            file.flush() # Force save to disk
            print(f"{data_list}")
            return True
            
        else:
            print(f"Ignored non-data text: {line}")
            return False
    return False

def write(ser, file):
    writer = csv.writer(file)
    writer.writerow(headers)

    print(f'Start logging')

    try:
        count = 0
        max_data_points = 1000
        while True and count < max_data_points:
            is_successful = process_sensor_data(ser, writer, file)
            if(is_successful):
                count += 1
                print(f'Saved {count} of  {max_data_points}')

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

with open(filename, "r+", newline = '') as file:
    #with statement isolates memory and by having an explicit statement to close the block further on, memory leak does not occur.
    #newline parameter just to address CRLF preferences; important for csv files.
    #"a" points to append and creates new file if not exists, and preserves existing data if exists
    write(file=file, ser=ser)


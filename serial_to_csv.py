import serial
import csv
import time 
import pandas as pd
import matplotlib.pyplot as plt

arduino_port = "COM8"
baud_rate = 115200
filename = "data.csv"
headers = ["Date", "Time", "Accel_X", "Accel_Y", "Accel_Z", "Acc_Magnitude", "Jerk_Magnitude", "Axial_x", "Axial_y", "Axial_z"]
memory_dict = {}


def process_sensor_data(ser, writer, file, count):
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

            row_dict = {"Date": current_date, "Time": current_time}
            sensor_headers = headers[2:]
            # keep sensor readings as floats, date/time as settings
            for key, val in zip(sensor_headers, data_list[2:]):
                try:
                    row_dict[key] = float(val)
                except ValueError:
                    row_dict[key] = val

            memory_dict[count] = row_dict

            print(f"{data_list}")
            return True
            
        else:
            print(f"Ignored non-data text: {line}")
            return False
    return False

def process(dict):
    def plot(dataframe):
        print('\nGenerating plot...')
        # Create a figure with 3 stacked subplots sharing the X (Time) axis
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        
        # 1. Top Plot: Raw Axes
        # (Make sure these string keys perfectly match your global 'headers' list)
        ax1.plot(dataframe.index, dataframe['Accel_X'], label='X', color='red', alpha=0.7)
        ax1.plot(dataframe.index, dataframe['Accel_Y'], label='Y', color='green', alpha=0.7)
        ax1.plot(dataframe.index, dataframe['Accel_Z'], label='Z', color='blue', alpha=0.7)
        ax1.set_title('Raw Accelerometer (mg)')
        ax1.set_ylabel('mg')
        ax1.legend(loc='upper right')
        ax1.grid(True)
        
        # 2. Middle Plot: Acceleration Magnitude
        ax2.plot(dataframe.index, dataframe['Acc_Magnitude'], label='Magnitude', color='purple')
        ax2.set_title('Total Acceleration Magnitude')
        ax2.set_ylabel('mg')
        ax2.grid(True)
        
        # 3. Bottom Plot: Jerk Magnitude
        ax3.plot(dataframe.index, dataframe['Jerk_Magnitude'], label='Jerk', color='orange')
        ax3.set_title('True Jerk Magnitude (The Fall Spike)')
        ax3.set_ylabel('mg/s')
        ax3.grid(True)
        
        plt.xlabel('Time')
        plt.tight_layout()
        plt.show()
        
    integrate_plot = True

    df = pd.DataFrame.from_dict(dict, orient = 'index')
    print(df)
    datetime_strings = df ['Date'] + ' ' + df['Time']
    df['Timestamp'] = pd.to_datetime(datetime_strings)
    df.set_index('Timestamp', inplace=True)
    df.drop(columns=['Date', 'Time'], inplace=True)
    df.head()

    if integrate_plot == True:
        plot(df)

    return

def write(ser, file):
    writer = csv.writer(file)
    writer.writerow(headers)

    print(f'Start logging')

    try:
        count = 0
        max_data_points = 2000
        while True and count < max_data_points:
            is_successful = process_sensor_data(ser, writer, file, count)
            if(is_successful):
                count += 1
                print(f'Saved {count} of  {max_data_points}')
        print(memory_dict)

    except KeyboardInterrupt:
        print("\nLogging stopped by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:

        file.close()
        ser.close()
        print("Serial port closed.")
        if len(memory_dict) > 0:
            print("Processing data and generating plots.")
            find_df = process(memory_dict)
    
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


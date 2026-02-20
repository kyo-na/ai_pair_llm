import serial
try:
    ser = serial.Serial('COM3', 9600, timeout=1)
    print("Waiting for Level 13 Nano-GPT Output...")
    count = 0
    while True:
        data = ser.read()
        if data:
            val = int.from_bytes(data, 'little')
            print(f"[{count:02d}] {val}")
            count += 1
            if count == 16: print("-" * 20); count = 0
except: print("Error or Port Closed")

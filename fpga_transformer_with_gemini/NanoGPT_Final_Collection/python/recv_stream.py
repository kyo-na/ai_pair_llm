import serial, time
try:
    ser = serial.Serial('COM3', 9600)
    print("Watching Level 15 Ouroboros Evolve...")
    gen = 0
    while True:
        vec = []
        for _ in range(32): vec.append(int.from_bytes(ser.read(), 'little', signed=True))
        print(f"\n--- Gen {gen} ---")
        for v in vec:
            bar = '#' * min(abs(v), 50)
            print(f"{v:4d} | {bar}" if v>=0 else f"{v:4d} | {'-'*min(abs(v),50)}")
        gen += 1
except: print("Stopped")

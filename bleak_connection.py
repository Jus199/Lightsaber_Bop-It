import asyncio
from bleak import BleakClient, BleakScanner

# Adafruit Bluefruit UART service UUIDs
UART_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
UART_RX_CHAR_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

OUTPUT_FILE = "sword_data.csv"
buffer = ""

def handle_data(sender, data):
    global buffer
    buffer += data.decode('utf-8')
    
    # Process only complete lines
    while '\n' in buffer:
        line, buffer = buffer.split('\n', 1)
        line = line.strip()
        if line:
            print(line)
            with open(OUTPUT_FILE, 'a') as f:
                f.write(line + '\n')

async def main():
    print("Scanning for Bluefruit...")
    device = await BleakScanner.find_device_by_name("Bluefruit")
    
    if not device:
        print("Device not found")
        return

    async with BleakClient(device) as client:
        print("Connected!")
        await client.start_notify(UART_RX_CHAR_UUID, handle_data)
        
        # Write CSV header
        with open(OUTPUT_FILE, 'w') as f:
            f.write('accelX,accelY,accelZ,gyroX,gyroY,gyroZ\n')
        
        print("Receiving data... Ctrl+C to stop")
        while True:
            await asyncio.sleep(1)

asyncio.run(main())
import asyncio
from bleak import BleakScanner

async def scan():
    print("Scanning...")
    devices = await BleakScanner.discover(timeout=10.0)
    if not devices:
        print("No devices found")
    for d in devices:
        print(f"Name: {d.name}, Address: {d.address}")

asyncio.run(scan())
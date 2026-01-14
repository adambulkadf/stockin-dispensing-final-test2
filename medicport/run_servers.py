"""
Run both servers on separate ports with SHARED MEMORY:
- Port 8000: Stock-in, warehouse, inventory (main.py)
- Port 8001: Dispensing operations (dispense_server.py)

Uses hypercorn to serve multiple apps in the same process.
"""

import asyncio
import subprocess
import sys
from hypercorn.asyncio import serve
from hypercorn.config import Config


def kill_existing_processes():
    """Kill any existing processes on ports 8000 and 8001"""
    for port in [8000, 8001]:
        try:
            result = subprocess.run(
                f"lsof -ti :{port}",
                shell=True,
                capture_output=True,
                text=True
            )
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    subprocess.run(f"kill -9 {pid}", shell=True, capture_output=True)
                    print(f"Killed existing process {pid} on port {port}")
        except Exception:
            pass


async def main():
    # Import apps after process starts to share memory
    from main import app as main_app
    from dispense_server import dispense_app

    print("=" * 60)
    print("MEDICPORT DUAL SERVER STARTUP")
    print("=" * 60)
    print("Starting servers:")
    print("  - Port 8000: Stock-in, Warehouse, Inventory")
    print("  - Port 8001: Dispensing Operations")
    print("=" * 60)
    print("Press Ctrl+C to stop both servers")
    print("=" * 60 + "\n")

    # Config for port 8000
    config_8000 = Config()
    config_8000.bind = ["0.0.0.0:8000"]

    # Config for port 8001
    config_8001 = Config()
    config_8001.bind = ["0.0.0.0:8001"]

    # Run both servers concurrently
    await asyncio.gather(
        serve(main_app, config_8000),
        serve(dispense_app, config_8001)
    )


if __name__ == "__main__":
    kill_existing_processes()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nServers stopped.")

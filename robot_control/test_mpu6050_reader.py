"""
test_mpu6050_reader.py

Test script for MPU6050 IMU reading.

This script tests MPU6050 accelerometer and gyroscope:
1. Initialize MPU6050 reader
2. Read accelerometer and gyroscope data repeatedly
3. Calculate and display tilt angle
4. Check for dangerous tilt
5. Handle sensor errors gracefully

Usage:
======
python3 robot_control/test_mpu6050_reader.py
"""

import os
import sys
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from robot_control.mpu6050_reader import MPU6050Reader, USE_DUMMY_MPU, DANGEROUS_TILT_DEGREES


def test_mpu6050_reader(num_readings=20, interval=0.5):
    """Test MPU6050 IMU reading."""
    print("=" * 60)
    print("MPU6050 IMU Reader Test")
    print("=" * 60)
    print(f"Readings: {num_readings}")
    print(f"Interval: {interval}s")
    print(f"Dangerous tilt threshold: {DANGEROUS_TILT_DEGREES}°")
    print()
    
    # Enable dummy mode for testing without hardware
    import robot_control.mpu6050_reader as mpu_module
    original_dummy = mpu_module.USE_DUMMY_MPU
    mpu_module.USE_DUMMY_MPU = True
    
    try:
        # Create reader
        print("Initializing MPU6050...")
        mpu = MPU6050Reader()
        mpu.start()
        print("MPU6050 started")
        print()
        
        # Read repeatedly
        print("Reading MPU6050 data...")
        print("-" * 60)
        
        valid_count = 0
        dangerous_tilt_count = 0
        
        for i in range(num_readings):
            data = mpu.get_data()
            
            valid = data.get("valid", False)
            if valid:
                valid_count += 1
                
            accel_x = data.get("accel_x")
            accel_y = data.get("accel_y")
            accel_z = data.get("accel_z")
            gyro_x = data.get("gyro_x")
            gyro_y = data.get("gyro_y")
            gyro_z = data.get("gyro_z")
            temp = data.get("temperature_c")
            tilt = data.get("tilt_angle")
            
            # Check dangerous tilt
            is_dangerous = mpu.is_tilted_dangerously()
            if is_dangerous:
                dangerous_tilt_count += 1
                
            tilt_status = "DANGEROUS!" if is_dangerous else "OK"
            
            # Format values
            def fmt(v):
                return f"{v:>8.4f}" if v is not None else "    None"
                
            def fmt_temp(v):
                return f"{v:>6.1f}°C" if v is not None else "  None"
                
            def fmt_tilt(v):
                return f"{v:>6.1f}°" if v is not None else " None°"
            
            print(f"  [{i+1:3d}] Accel: ({fmt(accel_x)}, {fmt(accel_y)}, {fmt(accel_z)}) | "
                  f"Gyro: ({fmt(gyro_x)}, {fmt(gyro_y)}, {fmt(gyro_z)}) | "
                  f"Temp: {fmt_temp(temp)} | "
                  f"Tilt: {fmt_tilt(tilt)} | "
                  f"Status: {tilt_status}")
            
            time.sleep(interval)
            
        print("-" * 60)
        print(f"Valid readings: {valid_count}/{num_readings}")
        print(f"Dangerous tilt events: {dangerous_tilt_count}")
        print()
        
        # Stop
        print("Stopping MPU6050...")
        mpu.stop()
        print("MPU6050 stopped")
        
        print()
        print("=" * 60)
        print("Test complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        # Restore original dummy setting
        mpu_module.USE_DUMMY_MPU = original_dummy
        
    return True


def main():
    success = test_mpu6050_reader()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
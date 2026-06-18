"""
test_lidar_reader.py

Test script for LiDAR distance reading.

This script tests LiDAR sensor communication:
1. Initialize LiDAR reader
2. Read distance repeatedly
3. Print distance and validity
4. Handle sensor errors gracefully

Usage:
======
python3 robot_control/test_lidar_reader.py
"""

import os
import sys
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from robot_control.lidar_reader import LidarReader, USE_DUMMY_LIDAR


def test_lidar_reader(num_readings=20, interval=0.5):
    """Test LiDAR distance reading."""
    print("=" * 60)
    print("LiDAR Reader Test")
    print("=" * 60)
    print(f"Readings: {num_readings}")
    print(f"Interval: {interval}s")
    print()
    
    # Enable dummy mode for testing without hardware
    import robot_control.lidar_reader as lidar_module
    original_dummy = lidar_module.USE_DUMMY_LIDAR
    lidar_module.USE_DUMMY_LIDAR = True
    
    try:
        # Create reader
        print("Initializing LiDAR...")
        lidar = LidarReader()
        lidar.start()
        print("LiDAR started")
        print()
        
        # Read repeatedly
        print("Reading LiDAR data...")
        print("-" * 60)
        
        valid_count = 0
        for i in range(num_readings):
            distance, valid = lidar.get_distance()
            data = lidar.get_data()
            
            status = "VALID" if valid else "INVALID"
            dist_str = f"{distance:.3f}m" if distance else "None"
            strength = data.get('strength', 'N/A')
            temp = data.get('temperature_c', 'N/A')
            temp_str = f"{temp}°C" if temp else "N/A"
            
            print(f"  [{i+1:3d}] Distance: {dist_str:>8s} | "
                  f"Status: {status:>7s} | "
                  f"Strength: {str(strength):>6s} | "
                  f"Temp: {temp_str}")
            
            if valid:
                valid_count += 1
                
            time.sleep(interval)
            
        print("-" * 60)
        print(f"Valid readings: {valid_count}/{num_readings}")
        print()
        
        # Stop
        print("Stopping LiDAR...")
        lidar.stop()
        print("LiDAR stopped")
        
        print()
        print("=" * 60)
        print("Test complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        # Restore original dummy setting
        lidar_module.USE_DUMMY_LIDAR = original_dummy
        
    return True


def main():
    success = test_lidar_reader()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
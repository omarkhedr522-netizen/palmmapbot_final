"""
test_relay_gpio_controller.py

Test script for Raspberry Pi GPIO relay controller.

This script tests basic relay control:
1. Initialize GPIO relay controller
2. Send STOP
3. Send FWD for 1 second
4. Send STOP
5. Send BACK for 1 second
6. Send STOP
7. Send LEFT for 1 second
8. Send STOP
9. Send RIGHT for 1 second
10. Send STOP
11. Cleanup

Safety: Test with wheels lifted off ground first!
"""

import os
import sys
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from robot_control.relay_gpio_controller import RelayGPIOController, GPIO_AVAILABLE


def test_relay_gpio_controller():
    """Test relay GPIO controller."""
    print("=" * 60)
    print("Relay GPIO Controller Test")
    print("=" * 60)
    
    if not GPIO_AVAILABLE:
        print("WARNING: GPIO not available - running in dummy mode")
        print("On Raspberry Pi, install gpiozero or RPi.GPIO")
        
    try:
        controller = RelayGPIOController()
        
        print("\nTest 1: STOP (initialize)")
        controller.stop()
        time.sleep(0.5)
        print("  STOP sent")
        
        print("\nTest 2: FORWARD for 1 second")
        controller.forward()
        time.sleep(1)
        controller.stop()
        print("  FORWARD complete")
        
        print("\nTest 3: BACKWARD for 1 second")
        controller.backward()
        time.sleep(1)
        controller.stop()
        print("  BACKWARD complete")
        
        print("\nTest 4: LEFT for 1 second")
        controller.left()
        time.sleep(1)
        controller.stop()
        print("  LEFT complete")
        
        print("\nTest 5: RIGHT for 1 second")
        controller.right()
        time.sleep(1)
        controller.stop()
        print("  RIGHT complete")
        
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
        
        controller.cleanup()
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    success = test_relay_gpio_controller()
    sys.exit(0 if success else 1)
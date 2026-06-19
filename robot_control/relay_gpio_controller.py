"""
relay_gpio_controller.py

Raspberry Pi GPIO relay controller for PalmMapBot.

This module directly controls a 4-channel relay module through Raspberry Pi GPIO pins.
It replaces the old Arduino serial controller.

Relay GPIO Mapping (BCM numbering) - UPDATED:
- Forward relay = GPIO 22
- Backward relay = GPIO 23
- Left relay = GPIO 27
- Right relay = GPIO 17

Safety:
- All relays initialize OFF by default
- Never allows Forward+Backward simultaneously
- Never allows Left+Right simultaneously
- Always stops before changing direction
- All relays turn OFF on cleanup/exit
- Single safe stop function that cuts all relay inputs

Wiring:
- Pi GPIO22 → Relay IN1 (Forward)
- Pi GPIO23 → Relay IN2 (Backward)
- Pi GPIO27 → Relay IN3 (Left)
- Pi GPIO17 → Relay IN4 (Right)
- Pi GND → Relay GND
- Relay VCC → appropriate power source

Important: Use a relay board compatible with 3.3V GPIO input.
"""

import os
import sys
import logging
import time
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default GPIO pins (BCM numbering) - CORRECTED
GPIO_FORWARD = 22
GPIO_BACKWARD = 23
GPIO_LEFT = 27
GPIO_RIGHT = 17

# Try to import GPIO libraries
RELAY_CONTROLLER = None
GPIO_AVAILABLE = False

try:
    # Try gpiozero first (safer, higher-level API)
    from gpiozero import LED
    RELAY_CONTROLLER = "gpiozero"
    GPIO_AVAILABLE = True
    logger.info("Using gpiozero for relay control")
except ImportError:
    try:
        # Fallback to RPi.GPIO
        import RPi.GPIO as GPIO
        RELAY_CONTROLLER = "RPi.GPIO"
        GPIO_AVAILABLE = True
        logger.info("Using RPi.GPIO for relay control")
    except ImportError:
        logger.warning("No GPIO library available - using dummy mode")
        RELAY_CONTROLLER = None
        GPIO_AVAILABLE = False


class RelayGPIOController:
    """
    Raspberry Pi GPIO relay controller with safe state management.
    
    This controller ensures:
    - All relays are OFF by default at startup
    - Only one direction is active at a time
    - Safe stop function cuts all relay inputs
    - Thread-safe operations with locks
    """
    
    def __init__(self, forward_pin=GPIO_FORWARD, backward_pin=GPIO_BACKWARD,
                 left_pin=GPIO_LEFT, right_pin=GPIO_RIGHT, active_low=True):
        """
        Initialize relay GPIO controller.
        
        Args:
            forward_pin: GPIO pin for forward relay (BCM) - default 22
            backward_pin: GPIO pin for backward relay (BCM) - default 23
            left_pin: GPIO pin for left relay (BCM) - default 27
            right_pin: GPIO pin for right relay (BCM) - default 17
            active_low: If True, GPIO LOW = relay ON (common for relay modules)
        """
        self.forward_pin = forward_pin
        self.backward_pin = backward_pin
        self.left_pin = left_pin
        self.right_pin = right_pin
        self.active_low = active_low
        
        self.forward_relay = None
        self.backward_relay = None
        self.left_relay = None
        self.right_relay = None
        
        self.initialized = False
        self._lock = threading.Lock()
        self._current_state = "STOP"  # Track current relay state
        
        self._setup_gpio()
        
    def _setup_gpio(self):
        """Set up GPIO pins for relay control. All relays start OFF."""
        if not GPIO_AVAILABLE:
            logger.warning("GPIO not available - using dummy mode")
            return
            
        try:
            if RELAY_CONTROLLER == "gpiozero":
                # gpiozero uses LED class for simple on/off control
                # initial_value=False ensures relays start OFF
                self.forward_relay = LED(self.forward_pin, initial_value=False)
                self.backward_relay = LED(self.backward_pin, initial_value=False)
                self.left_relay = LED(self.left_pin, initial_value=False)
                self.right_relay = LED(self.right_pin, initial_value=False)
                
            elif RELAY_CONTROLLER == "RPi.GPIO":
                # RPi.GPIO setup
                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)
                # initial=GPIO.HIGH means OFF for active-low relays
                GPIO.setup(self.forward_pin, GPIO.OUT, initial=GPIO.HIGH)
                GPIO.setup(self.backward_pin, GPIO.OUT, initial=GPIO.HIGH)
                GPIO.setup(self.left_pin, GPIO.OUT, initial=GPIO.HIGH)
                GPIO.setup(self.right_pin, GPIO.OUT, initial=GPIO.HIGH)
                
            self.initialized = True
            logger.info(f"Relay GPIO initialized: FWD={self.forward_pin}, BACK={self.backward_pin}, "
                       f"LEFT={self.left_pin}, RIGHT={self.right_pin}")
            logger.info("All relays set to OFF (safe default state)")
            
        except Exception as e:
            logger.error(f"GPIO setup failed: {e}")
            self.initialized = False
            
    def stop_all_relays(self):
        """
        SAFE STOP FUNCTION - Cut off all relay inputs completely.
        
        This is the primary safe stop function that must be used everywhere
        the car needs to stop. It ensures:
        - GPIO 22 OFF (forward)
        - GPIO 23 OFF (backward)
        - GPIO 27 OFF (left)
        - GPIO 17 OFF (right)
        """
        with self._lock:
            if not self.initialized:
                return
                
            try:
                if RELAY_CONTROLLER == "gpiozero":
                    if self.forward_relay:
                        self.forward_relay.off()
                    if self.backward_relay:
                        self.backward_relay.off()
                    if self.left_relay:
                        self.left_relay.off()
                    if self.right_relay:
                        self.right_relay.off()
                elif RELAY_CONTROLLER == "RPi.GPIO":
                    # HIGH = OFF for active-low relays
                    GPIO.output(self.forward_pin, GPIO.HIGH)
                    GPIO.output(self.backward_pin, GPIO.HIGH)
                    GPIO.output(self.left_pin, GPIO.HIGH)
                    GPIO.output(self.right_pin, GPIO.HIGH)
                    
                self._current_state = "STOP"
                logger.info("STOP ALL RELAYS - All inputs OFF (FWD=OFF, BACK=OFF, LEFT=OFF, RIGHT=OFF)")
                
            except Exception as e:
                logger.error(f"Stop all relays failed: {e}")
    
    def stop(self):
        """Alias for stop_all_relays() - Stop all movement."""
        self.stop_all_relays()
            
    def forward(self):
        """
        Move forward - activate ONLY GPIO 22.
        
        Safety: Stops all relays first, then activates only forward.
        Ensures GPIO 23, GPIO 27, and GPIO 17 remain OFF.
        """
        with self._lock:
            if not self.initialized:
                return
                
            try:
                # First, ensure all relays are OFF
                self._deactivate_all()
                time.sleep(0.05)  # Brief delay for safety
                
                # Activate ONLY forward relay
                self._activate_relay_only("forward")
                self._current_state = "FORWARD"
                logger.info(f"FORWARD - GPIO {self.forward_pin} ON, all others OFF")
                
            except Exception as e:
                logger.error(f"Forward failed: {e}")
                self.stop_all_relays()
            
    def backward(self):
        """
        Move backward - activate ONLY GPIO 23.
        
        Safety: Stops all relays first, then activates only backward.
        Ensures GPIO 22, GPIO 27, and GPIO 17 remain OFF.
        """
        with self._lock:
            if not self.initialized:
                return
                
            try:
                # First, ensure all relays are OFF
                self._deactivate_all()
                time.sleep(0.05)  # Brief delay for safety
                
                # Activate ONLY backward relay
                self._activate_relay_only("backward")
                self._current_state = "BACKWARD"
                logger.info(f"BACKWARD - GPIO {self.backward_pin} ON, all others OFF")
                
            except Exception as e:
                logger.error(f"Backward failed: {e}")
                self.stop_all_relays()
            
    def left(self):
        """
        Turn left - activate ONLY GPIO 27.
        
        Safety: Stops all relays first, then activates only left.
        Ensures GPIO 22, GPIO 23, and GPIO 17 remain OFF.
        """
        with self._lock:
            if not self.initialized:
                return
                
            try:
                # First, ensure all relays are OFF
                self._deactivate_all()
                time.sleep(0.05)  # Brief delay for safety
                
                # Activate ONLY left relay
                self._activate_relay_only("left")
                self._current_state = "LEFT"
                logger.info(f"LEFT - GPIO {self.left_pin} ON, all others OFF")
                
            except Exception as e:
                logger.error(f"Left failed: {e}")
                self.stop_all_relays()
            
    def right(self):
        """
        Turn right - activate ONLY GPIO 17.
        
        Safety: Stops all relays first, then activates only right.
        Ensures GPIO 22, GPIO 23, and GPIO 27 remain OFF.
        """
        with self._lock:
            if not self.initialized:
                return
                
            try:
                # First, ensure all relays are OFF
                self._deactivate_all()
                time.sleep(0.05)  # Brief delay for safety
                
                # Activate ONLY right relay
                self._activate_relay_only("right")
                self._current_state = "RIGHT"
                logger.info(f"RIGHT - GPIO {self.right_pin} ON, all others OFF")
                
            except Exception as e:
                logger.error(f"Right failed: {e}")
                self.stop_all_relays()
    
    def _deactivate_all(self):
        """Internal: Deactivate all relays (must be called with lock held)."""
        if RELAY_CONTROLLER == "gpiozero":
            if self.forward_relay:
                self.forward_relay.off()
            if self.backward_relay:
                self.backward_relay.off()
            if self.left_relay:
                self.left_relay.off()
            if self.right_relay:
                self.right_relay.off()
        elif RELAY_CONTROLLER == "RPi.GPIO":
            GPIO.output(self.forward_pin, GPIO.HIGH)
            GPIO.output(self.backward_pin, GPIO.HIGH)
            GPIO.output(self.left_pin, GPIO.HIGH)
            GPIO.output(self.right_pin, GPIO.HIGH)
    
    def _activate_relay_only(self, direction):
        """Internal: Activate only the specified relay (must be called with lock held)."""
        if RELAY_CONTROLLER == "gpiozero":
            if direction == "forward" and self.forward_relay:
                self.forward_relay.on()
            elif direction == "backward" and self.backward_relay:
                self.backward_relay.on()
            elif direction == "left" and self.left_relay:
                self.left_relay.on()
            elif direction == "right" and self.right_relay:
                self.right_relay.on()
        elif RELAY_CONTROLLER == "RPi.GPIO":
            # First set all to HIGH (OFF)
            GPIO.output(self.forward_pin, GPIO.HIGH)
            GPIO.output(self.backward_pin, GPIO.HIGH)
            GPIO.output(self.left_pin, GPIO.HIGH)
            GPIO.output(self.right_pin, GPIO.HIGH)
            
            # Then set only the desired pin to LOW (ON)
            if direction == "forward":
                GPIO.output(self.forward_pin, GPIO.LOW)
            elif direction == "backward":
                GPIO.output(self.backward_pin, GPIO.LOW)
            elif direction == "left":
                GPIO.output(self.left_pin, GPIO.LOW)
            elif direction == "right":
                GPIO.output(self.right_pin, GPIO.LOW)
            
    def forward_ms(self, milliseconds):
        """Move forward for specified milliseconds."""
        self.forward()
        time.sleep(milliseconds / 1000.0)
        self.stop_all_relays()
        
    def backward_ms(self, milliseconds):
        """Move backward for specified milliseconds."""
        self.backward()
        time.sleep(milliseconds / 1000.0)
        self.stop_all_relays()
        
    def left_ms(self, milliseconds):
        """Turn left for specified milliseconds."""
        self.left()
        time.sleep(milliseconds / 1000.0)
        self.stop_all_relays()
        
    def right_ms(self, milliseconds):
        """Turn right for specified milliseconds."""
        self.right()
        time.sleep(milliseconds / 1000.0)
        self.stop_all_relays()
        
    def cleanup(self):
        """Clean up GPIO resources. Ensures all relays are OFF."""
        logger.info("Cleaning up relay GPIO controller...")
        self.stop_all_relays()
        
        if RELAY_CONTROLLER == "RPi.GPIO":
            try:
                GPIO.cleanup()
            except Exception as e:
                logger.error(f"GPIO cleanup error: {e}")
                
        self.initialized = False
        logger.info("Relay GPIO cleaned up - all relays OFF")
        
    def get_status(self):
        """Get current relay status."""
        return {
            "initialized": self.initialized,
            "gpio_available": GPIO_AVAILABLE,
            "controller": RELAY_CONTROLLER,
            "active_low": self.active_low,
            "current_state": self._current_state,
            "pins": {
                "forward": self.forward_pin,
                "backward": self.backward_pin,
                "left": self.left_pin,
                "right": self.right_pin
            }
        }
        
    def is_moving(self):
        """Check if any relay is currently active."""
        return self._current_state != "STOP"
    
    def get_current_state(self):
        """Get the current relay state string."""
        return self._current_state


# Global controller instance
_controller = None
_controller_lock = threading.Lock()


def get_relay_controller():
    """Get or create global relay controller instance (thread-safe)."""
    global _controller
    with _controller_lock:
        if _controller is None:
            _controller = RelayGPIOController()
        return _controller


def stop_all():
    """Stop all relays using global controller (safe stop)."""
    global _controller
    with _controller_lock:
        if _controller is not None:
            _controller.stop_all_relays()


def cleanup_all():
    """Clean up global controller."""
    global _controller
    with _controller_lock:
        if _controller is not None:
            _controller.cleanup()
            _controller = None


def reset_controller():
    """Reset the global controller (useful for reinitialization)."""
    global _controller
    with _controller_lock:
        if _controller is not None:
            _controller.cleanup()
            _controller = None


# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("Relay GPIO Controller Test")
    print("=" * 60)
    print(f"GPIO Pins: FWD={GPIO_FORWARD}, BACK={GPIO_BACKWARD}, LEFT={GPIO_LEFT}, RIGHT={GPIO_RIGHT}")
    print()
    
    if not GPIO_AVAILABLE:
        print("GPIO not available - running in dummy mode")
        print("On Raspberry Pi, install gpiozero or RPi.GPIO")
        print()
        
    controller = RelayGPIOController()
    
    try:
        print("\nTest 1: Initial State (should be STOP)")
        print(f"  Current state: {controller.get_current_state()}")
        
        print("\nTest 2: STOP")
        controller.stop_all_relays()
        time.sleep(0.5)
        print("  STOP sent - all relays OFF")
        
        print("\nTest 3: FORWARD for 1 second")
        controller.forward()
        print(f"  State: {controller.get_current_state()}")
        time.sleep(1)
        controller.stop_all_relays()
        print("  STOP - FORWARD complete")
        
        print("\nTest 4: BACKWARD for 1 second")
        controller.backward()
        print(f"  State: {controller.get_current_state()}")
        time.sleep(1)
        controller.stop_all_relays()
        print("  STOP - BACKWARD complete")
        
        print("\nTest 5: LEFT for 1 second")
        controller.left()
        print(f"  State: {controller.get_current_state()}")
        time.sleep(1)
        controller.stop_all_relays()
        print("  STOP - LEFT complete")
        
        print("\nTest 6: RIGHT for 1 second")
        controller.right()
        print(f"  State: {controller.get_current_state()}")
        time.sleep(1)
        controller.stop_all_relays()
        print("  STOP - RIGHT complete")
        
        print("\n" + "=" * 60)
        print("All tests complete!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        controller.cleanup()
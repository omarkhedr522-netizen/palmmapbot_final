"""
relay_gpio_controller.py

Raspberry Pi GPIO relay controller for PalmMapBot.

This module directly controls a 4-channel relay module through Raspberry Pi GPIO pins.
It replaces the old Arduino serial controller.

Relay GPIO Mapping (BCM numbering):
- Forward relay = GPIO17
- Backward relay = GPIO27
- Left relay = GPIO22
- Right relay = GPIO23

Safety:
- All relays initialize OFF
- Never allows Forward+Backward simultaneously
- Never allows Left+Right simultaneously
- Always stops before changing direction
- All relays turn OFF on cleanup/exit

Wiring:
- Pi GPIO17 → Relay IN1 (Forward)
- Pi GPIO27 → Relay IN2 (Backward)
- Pi GPIO22 → Relay IN3 (Left)
- Pi GPIO23 → Relay IN4 (Right)
- Pi GND → Relay GND
- Relay VCC → appropriate power source

Important: Use a relay board compatible with 3.3V GPIO input.
"""

import os
import sys
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default GPIO pins (BCM numbering)
GPIO_FORWARD = 17
GPIO_BACKWARD = 27
GPIO_LEFT = 22
GPIO_RIGHT = 23

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
    """Raspberry Pi GPIO relay controller."""
    
    def __init__(self, forward_pin=GPIO_FORWARD, backward_pin=GPIO_BACKWARD,
                 left_pin=GPIO_LEFT, right_pin=GPIO_RIGHT, active_low=True):
        """
        Initialize relay GPIO controller.
        
        Args:
            forward_pin: GPIO pin for forward relay (BCM)
            backward_pin: GPIO pin for backward relay (BCM)
            left_pin: GPIO pin for left relay (BCM)
            right_pin: GPIO pin for right relay (BCM)
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
        self._setup_gpio()
        
    def _setup_gpio(self):
        """Set up GPIO pins for relay control."""
        if not GPIO_AVAILABLE:
            logger.warning("GPIO not available - using dummy mode")
            return
            
        try:
            if RELAY_CONTROLLER == "gpiozero":
                # gpiozero uses LED class for simple on/off control
                self.forward_relay = LED(self.forward_pin, initial_value=False)
                self.backward_relay = LED(self.backward_pin, initial_value=False)
                self.left_relay = LED(self.left_pin, initial_value=False)
                self.right_relay = LED(self.right_pin, initial_value=False)
                
            elif RELAY_CONTROLLER == "RPi.GPIO":
                # RPi.GPIO setup
                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)
                GPIO.setup(self.forward_pin, GPIO.OUT, initial=GPIO.HIGH)
                GPIO.setup(self.backward_pin, GPIO.OUT, initial=GPIO.HIGH)
                GPIO.setup(self.left_pin, GPIO.OUT, initial=GPIO.HIGH)
                GPIO.setup(self.right_pin, GPIO.OUT, initial=GPIO.HIGH)
                
            self.initialized = True
            logger.info(f"Relay GPIO initialized: FWD={self.forward_pin}, BACK={self.backward_pin}, "
                       f"LEFT={self.left_pin}, RIGHT={self.right_pin}")
            
        except Exception as e:
            logger.error(f"GPIO setup failed: {e}")
            self.initialized = False
            
    def _activate_relay(self, relay):
        """Activate a relay (turn ON)."""
        if relay is None:
            return
            
        if RELAY_CONTROLLER == "gpiozero":
            relay.on()
        elif RELAY_CONTROLLER == "RPi.GPIO":
            GPIO.output(relay.pin if hasattr(relay, 'pin') else relay, GPIO.LOW if self.active_low else GPIO.HIGH)
            
    def _deactivate_relay(self, relay):
        """Deactivate a relay (turn OFF)."""
        if relay is None:
            return
            
        if RELAY_CONTROLLER == "gpiozero":
            relay.off()
        elif RELAY_CONTROLLER == "RPi.GPIO":
            GPIO.output(relay.pin if hasattr(relay, 'pin') else relay, GPIO.HIGH if self.active_low else GPIO.LOW)
            
    def stop(self):
        """Stop all movement - turn off all relays."""
        if not self.initialized:
            return
            
        try:
            self._deactivate_relay(self.forward_relay)
            self._deactivate_relay(self.backward_relay)
            self._deactivate_relay(self.left_relay)
            self._deactivate_relay(self.right_relay)
            logger.debug("All relays OFF (STOP)")
        except Exception as e:
            logger.error(f"Stop failed: {e}")
            
    def forward(self):
        """Move forward - activate forward relay only."""
        if not self.initialized:
            return
            
        try:
            self.stop()  # Safety: stop first
            time.sleep(0.05)  # Brief delay
            self._activate_relay(self.forward_relay)
            logger.debug("FORWARD")
        except Exception as e:
            logger.error(f"Forward failed: {e}")
            self.stop()
            
    def backward(self):
        """Move backward - activate backward relay only."""
        if not self.initialized:
            return
            
        try:
            self.stop()  # Safety: stop first
            time.sleep(0.05)  # Brief delay
            self._activate_relay(self.backward_relay)
            logger.debug("BACKWARD")
        except Exception as e:
            logger.error(f"Backward failed: {e}")
            self.stop()
            
    def left(self):
        """Turn left - activate left relay only."""
        if not self.initialized:
            return
            
        try:
            self.stop()  # Safety: stop first
            time.sleep(0.05)  # Brief delay
            self._activate_relay(self.left_relay)
            logger.debug("LEFT")
        except Exception as e:
            logger.error(f"Left failed: {e}")
            self.stop()
            
    def right(self):
        """Turn right - activate right relay only."""
        if not self.initialized:
            return
            
        try:
            self.stop()  # Safety: stop first
            time.sleep(0.05)  # Brief delay
            self._activate_relay(self.right_relay)
            logger.debug("RIGHT")
        except Exception as e:
            logger.error(f"Right failed: {e}")
            self.stop()
            
    def forward_ms(self, milliseconds):
        """Move forward for specified milliseconds."""
        self.forward()
        time.sleep(milliseconds / 1000.0)
        self.stop()
        
    def backward_ms(self, milliseconds):
        """Move backward for specified milliseconds."""
        self.backward()
        time.sleep(milliseconds / 1000.0)
        self.stop()
        
    def left_ms(self, milliseconds):
        """Turn left for specified milliseconds."""
        self.left()
        time.sleep(milliseconds / 1000.0)
        self.stop()
        
    def right_ms(self, milliseconds):
        """Turn right for specified milliseconds."""
        self.right()
        time.sleep(milliseconds / 1000.0)
        self.stop()
        
    def cleanup(self):
        """Clean up GPIO resources."""
        self.stop()
        
        if RELAY_CONTROLLER == "RPi.GPIO":
            try:
                GPIO.cleanup()
            except Exception:
                pass
                
        self.initialized = False
        logger.info("Relay GPIO cleaned up")
        
    def get_status(self):
        """Get current relay status."""
        return {
            "initialized": self.initialized,
            "gpio_available": GPIO_AVAILABLE,
            "controller": RELAY_CONTROLLER,
            "active_low": self.active_low,
            "pins": {
                "forward": self.forward_pin,
                "backward": self.backward_pin,
                "left": self.left_pin,
                "right": self.right_pin
            }
        }


# Global controller instance
_controller = None


def get_relay_controller():
    """Get or create global relay controller instance."""
    global _controller
    if _controller is None:
        _controller = RelayGPIOController()
    return _controller


def stop_all():
    """Stop all relays using global controller."""
    global _controller
    if _controller is not None:
        _controller.stop()


def cleanup_all():
    """Clean up global controller."""
    global _controller
    if _controller is not None:
        _controller.cleanup()
        _controller = None


# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("Relay GPIO Controller Test")
    print("=" * 60)
    
    if not GPIO_AVAILABLE:
        print("GPIO not available - running in dummy mode")
        print("On Raspberry Pi, install gpiozero or RPi.GPIO")
        
    controller = RelayGPIOController()
    
    try:
        print("\nTest 1: STOP")
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
        print("All tests complete!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        controller.cleanup()
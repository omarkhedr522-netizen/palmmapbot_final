"""
robot_state_manager.py

Centralized, thread-safe robot state manager for PalmMapBot.

This module provides:
- Thread-safe state management with locks
- Centralized relay control coordination
- Mission state tracking
- Abort handling with immediate relay cutoff
- Timed manual control pulses
- Single source of truth for robot state

This should be the primary interface for controlling the robot
from the dashboard, mission controller, or any other component.
"""

import threading
import time
import logging

from robot_control.relay_gpio_controller import (
    RelayGPIOController, get_relay_controller, stop_all, cleanup_all
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# Manual Control Timing Constants (in seconds)
# =============================================================================
# These define how long each relay stays active when a manual button is pressed
MANUAL_FORWARD_DURATION = 0.5    # Forward relay GPIO 22: 0.5 seconds
MANUAL_BACKWARD_DURATION = 0.5   # Backward relay GPIO 23: 0.5 seconds
MANUAL_LEFT_DURATION = 0.25      # Left relay GPIO 27: 0.25 seconds
MANUAL_RIGHT_DURATION = 0.25     # Right relay GPIO 17: 0.25 seconds


class RobotStateManager:
    """
    Centralized, thread-safe robot state manager.
    
    This class manages:
    - Robot mode (IDLE, MANUAL, AUTO, STOPPED, EMERGENCY_STOP)
    - Mission state (active/inactive)
    - Emergency stop state
    - Relay control with safe stop guarantees
    - Thread-safe state access
    - Timed manual control pulses
    
    All state changes are protected by locks to prevent race conditions.
    """
    
    # Robot modes
    MODE_IDLE = "IDLE"
    MODE_MANUAL = "MANUAL"
    MODE_AUTO = "AUTO"
    MODE_STOPPED = "STOPPED"
    MODE_EMERGENCY_STOP = "EMERGENCY_STOP"
    
    def __init__(self):
        """Initialize the robot state manager."""
        self._lock = threading.RLock()  # Reentrant lock for nested calls
        self._state_lock = threading.Lock()  # Separate lock for state dict
        self._manual_lock = threading.Lock()  # Lock for manual control timing
        
        # State variables
        self._mode = self.MODE_IDLE
        self._mission_active = False
        self._emergency_stop = False
        self._running = False
        self._abort_requested = False
        
        # Relay controller (singleton)
        self._relay_controller = None
        
        # Manual control timing thread
        self._manual_pulse_thread = None
        self._manual_pulse_active = False
        
        # State change callbacks
        self._state_callbacks = []
        
        logger.info("RobotStateManager initialized")
    
    def initialize_relay_controller(self):
        """
        Initialize the relay controller. Must be called before any relay operations.
        
        This ensures all relays start in the OFF state.
        """
        with self._lock:
            try:
                self._relay_controller = get_relay_controller()
                # Ensure all relays are OFF after initialization
                self._relay_controller.stop_all_relays()
                logger.info("Relay controller initialized - all relays OFF")
                self._running = True
                return True
            except Exception as e:
                logger.error(f"Failed to initialize relay controller: {e}")
                return False
    
    def get_relay_controller(self):
        """Get the relay controller instance."""
        return self._relay_controller
    
    # =========================================================================
    # State Query Methods (thread-safe)
    # =========================================================================
    
    def get_state(self):
        """Get current robot state as a dictionary."""
        with self._state_lock:
            relay_state = "OFF"
            if self._relay_controller:
                relay_state = self._relay_controller.get_current_state()
            
            return {
                "mode": self._mode,
                "mission_active": self._mission_active,
                "emergency_stop": self._emergency_stop,
                "abort_requested": self._abort_requested,
                "running": self._running,
                "relay_status": relay_state,
                "last_command": relay_state if relay_state != "OFF" else "STOP"
            }
    
    def get_mode(self):
        """Get current mode."""
        with self._state_lock:
            return self._mode
    
    def is_mission_active(self):
        """Check if a mission is currently active."""
        with self._state_lock:
            return self._mission_active
    
    def is_emergency_stop(self):
        """Check if emergency stop is active."""
        with self._state_lock:
            return self._emergency_stop
    
    def is_abort_requested(self):
        """Check if abort has been requested."""
        with self._state_lock:
            return self._abort_requested
    
    def is_running(self):
        """Check if the state manager is running."""
        with self._state_lock:
            return self._running
    
    # =========================================================================
    # Mission Control Methods
    # =========================================================================
    
    def start_mission(self):
        """
        Start a mission.
        
        - Clears emergency stop and abort flags
        - Sets mode to AUTO
        - Does NOT activate relays (caller must explicitly command movement)
        """
        with self._lock:
            with self._state_lock:
                if self._emergency_stop:
                    logger.warning("Cannot start mission - emergency stop is active")
                    return False
                
                self._mission_active = True
                self._abort_requested = False
                self._mode = self.MODE_AUTO
                
            logger.info("MISSION STARTED - Mode: AUTO, Relays: OFF (waiting for command)")
            self._notify_state_change()
            return True
    
    def stop_mission(self):
        """
        Stop a mission gracefully.
        
        - Deactivates mission
        - Stops all relays
        - Sets mode to STOPPED
        """
        with self._lock:
            with self._state_lock:
                self._mission_active = False
                
            # Stop all relays
            self._safe_stop_all()
            
            with self._state_lock:
                self._mode = self.MODE_STOPPED
                
            logger.info("MISSION STOPPED - All relays OFF")
            self._notify_state_change()
    
    def abort_mission(self):
        """
        EMERGENCY ABORT - Immediately stop everything.
        
        This is the highest priority stop command:
        - Cuts all relay inputs immediately
        - Sets emergency stop flag
        - Deactivates mission
        - Overrides all other commands
        
        This method is designed to be called from any thread and will
        immediately stop the robot.
        """
        with self._lock:
            # First, cut all relays immediately (outside state lock for speed)
            self._safe_stop_all()
            
            with self._state_lock:
                self._emergency_stop = True
                self._abort_requested = True
                self._mission_active = False
                self._mode = self.MODE_EMERGENCY_STOP
                
            logger.warning("ABORT MISSION - EMERGENCY STOP - All relays OFF")
            self._notify_state_change()
    
    def complete_mission(self):
        """
        Complete a mission normally.
        
        - Deactivates mission
        - Stops all relays
        - Sets mode to IDLE
        """
        with self._lock:
            self._safe_stop_all()
            
            with self._state_lock:
                self._mission_active = False
                self._mode = self.MODE_IDLE
                
            logger.info("MISSION COMPLETED - All relays OFF")
            self._notify_state_change()
    
    # =========================================================================
    # Mode Switching Methods
    # =========================================================================
    
    def switch_to_manual(self):
        """Switch to manual control mode."""
        with self._lock:
            self._safe_stop_all()
            
            with self._state_lock:
                self._mission_active = False
                self._mode = self.MODE_MANUAL
                
            logger.info("MODE SWITCHED TO MANUAL - All relays OFF")
            self._notify_state_change()
    
    def switch_to_auto(self):
        """Switch to auto mode (ready, not moving)."""
        with self._lock:
            self._safe_stop_all()
            
            with self._state_lock:
                self._mission_active = False
                self._mode = self.MODE_IDLE
                
            logger.info("MODE SWITCHED TO AUTO (IDLE) - All relays OFF")
            self._notify_state_change()
    
    # =========================================================================
    # Manual Control Methods - TIMED PULSES
    # =========================================================================
    # Manual control uses timed pulses:
    # - Forward/Backward: 0.5 seconds
    # - Left/Right: 0.25 seconds
    # The relay activates for the specified duration, then automatically turns off.
    # =========================================================================
    
    def _manual_pulse_worker(self, direction, duration):
        """
        Internal worker thread for timed manual control pulses.
        
        Activates the specified relay for the given duration, then stops.
        Runs in a separate thread to avoid blocking the dashboard.
        """
        with self._manual_lock:
            self._manual_pulse_active = True
        
        try:
            # Activate the relay
            if direction == "forward":
                self._relay_controller.forward()
            elif direction == "backward":
                self._relay_controller.backward()
            elif direction == "left":
                self._relay_controller.left()
            elif direction == "right":
                self._relay_controller.right()
            
            # Wait for the specified duration
            time.sleep(duration)
            
            # Stop all relays
            self._safe_stop_all()
            logger.info(f"Manual {direction.upper()} pulse complete ({duration}s)")
            
        except Exception as e:
            logger.error(f"Manual pulse worker error: {e}")
            self._safe_stop_all()
        finally:
            with self._manual_lock:
                self._manual_pulse_active = False
                self._manual_pulse_thread = None
    
    def manual_forward(self):
        """
        Manual forward control with timed pulse.
        
        Only works in MANUAL mode and when emergency stop is not active.
        Activates ONLY GPIO 22 (forward relay) for 0.5 seconds, then stops.
        """
        with self._lock:
            with self._state_lock:
                if self._mode != self.MODE_MANUAL:
                    logger.warning("Manual forward rejected - not in MANUAL mode")
                    return False
                if self._emergency_stop:
                    logger.warning("Manual forward rejected - emergency stop active")
                    return False
            
            if self._relay_controller:
                # Cancel any existing manual pulse
                self._cancel_manual_pulse()
                
                # Start new timed pulse in background thread
                self._manual_pulse_thread = threading.Thread(
                    target=self._manual_pulse_worker,
                    args=("forward", MANUAL_FORWARD_DURATION),
                    daemon=True
                )
                self._manual_pulse_thread.start()
                logger.info(f"MANUAL FORWARD - GPIO 22 ON for {MANUAL_FORWARD_DURATION}s")
                return True
        return False
    
    def manual_backward(self):
        """
        Manual backward control with timed pulse.
        
        Only works in MANUAL mode and when emergency stop is not active.
        Activates ONLY GPIO 23 (backward relay) for 0.5 seconds, then stops.
        """
        with self._lock:
            with self._state_lock:
                if self._mode != self.MODE_MANUAL:
                    logger.warning("Manual backward rejected - not in MANUAL mode")
                    return False
                if self._emergency_stop:
                    logger.warning("Manual backward rejected - emergency stop active")
                    return False
            
            if self._relay_controller:
                # Cancel any existing manual pulse
                self._cancel_manual_pulse()
                
                # Start new timed pulse in background thread
                self._manual_pulse_thread = threading.Thread(
                    target=self._manual_pulse_worker,
                    args=("backward", MANUAL_BACKWARD_DURATION),
                    daemon=True
                )
                self._manual_pulse_thread.start()
                logger.info(f"MANUAL BACKWARD - GPIO 23 ON for {MANUAL_BACKWARD_DURATION}s")
                return True
        return False
    
    def manual_left(self):
        """
        Manual left turn control with timed pulse.
        
        Only works in MANUAL mode and when emergency stop is not active.
        Activates ONLY GPIO 27 (left relay) for 0.25 seconds, then stops.
        """
        with self._lock:
            with self._state_lock:
                if self._mode != self.MODE_MANUAL:
                    logger.warning("Manual left rejected - not in MANUAL mode")
                    return False
                if self._emergency_stop:
                    logger.warning("Manual left rejected - emergency stop active")
                    return False
            
            if self._relay_controller:
                # Cancel any existing manual pulse
                self._cancel_manual_pulse()
                
                # Start new timed pulse in background thread
                self._manual_pulse_thread = threading.Thread(
                    target=self._manual_pulse_worker,
                    args=("left", MANUAL_LEFT_DURATION),
                    daemon=True
                )
                self._manual_pulse_thread.start()
                logger.info(f"MANUAL LEFT - GPIO 27 ON for {MANUAL_LEFT_DURATION}s")
                return True
        return False
    
    def manual_right(self):
        """
        Manual right turn control with timed pulse.
        
        Only works in MANUAL mode and when emergency stop is not active.
        Activates ONLY GPIO 17 (right relay) for 0.25 seconds, then stops.
        """
        with self._lock:
            with self._state_lock:
                if self._mode != self.MODE_MANUAL:
                    logger.warning("Manual right rejected - not in MANUAL mode")
                    return False
                if self._emergency_stop:
                    logger.warning("Manual right rejected - emergency stop active")
                    return False
            
            if self._relay_controller:
                # Cancel any existing manual pulse
                self._cancel_manual_pulse()
                
                # Start new timed pulse in background thread
                self._manual_pulse_thread = threading.Thread(
                    target=self._manual_pulse_worker,
                    args=("right", MANUAL_RIGHT_DURATION),
                    daemon=True
                )
                self._manual_pulse_thread.start()
                logger.info(f"MANUAL RIGHT - GPIO 17 ON for {MANUAL_RIGHT_DURATION}s")
                return True
        return False
    
    def _cancel_manual_pulse(self):
        """Cancel any active manual pulse."""
        with self._manual_lock:
            if self._manual_pulse_active:
                self._safe_stop_all()
                self._manual_pulse_active = False
                logger.info("Manual pulse cancelled")
    
    def manual_stop(self):
        """
        Manual stop - works in any mode.
        
        Stops all relays immediately and cancels any active manual pulse.
        """
        with self._lock:
            self._cancel_manual_pulse()
            self._safe_stop_all()
            logger.info("MANUAL STOP - All relays OFF")
            self._notify_state_change()
    
    # =========================================================================
    # Automation Control Methods
    # =========================================================================
    
    def auto_forward(self):
        """
        Automatic forward control (for mission automation).
        
        Only works when mission is active and no abort/emergency.
        Activates ONLY GPIO 22 (forward relay).
        Unlike manual control, this stays active until explicitly stopped.
        """
        with self._lock:
            with self._state_lock:
                if not self._mission_active:
                    logger.warning("Auto forward rejected - mission not active")
                    return False
                if self._emergency_stop or self._abort_requested:
                    logger.warning("Auto forward rejected - emergency/abort active")
                    return False
                if self._mode != self.MODE_AUTO:
                    logger.warning("Auto forward rejected - not in AUTO mode")
                    return False
            
            if self._relay_controller:
                self._relay_controller.forward()
                logger.info("AUTO FORWARD - GPIO 22 ON")
                return True
        return False
    
    def auto_stop(self):
        """
        Automatic stop (for tree detection, etc.).
        
        Stops all relays during automation.
        """
        with self._lock:
            self._safe_stop_all()
            logger.info("AUTO STOP - All relays OFF")
    
    def should_continue_mission(self):
        """
        Check if the mission should continue.
        
        Returns True if:
        - Mission is active
        - No emergency stop
        - No abort requested
        - Manager is running
        
        This should be checked frequently in the mission loop.
        """
        with self._state_lock:
            return (self._mission_active and 
                    not self._emergency_stop and 
                    not self._abort_requested and 
                    self._running)
    
    # =========================================================================
    # Internal Methods
    # =========================================================================
    
    def _safe_stop_all(self):
        """
        Internal safe stop - cut all relay inputs.
        
        This is the primary safe stop function that ensures all relays are OFF.
        """
        if self._relay_controller:
            self._relay_controller.stop_all_relays()
    
    def add_state_callback(self, callback):
        """Add a callback to be notified of state changes."""
        self._state_callbacks.append(callback)
    
    def _notify_state_change(self):
        """Notify all callbacks of a state change."""
        state = self.get_state()
        for callback in self._state_callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"State callback error: {e}")
    
    def shutdown(self):
        """
        Shutdown the state manager.
        
        Stops all relays and marks the manager as not running.
        """
        with self._lock:
            self._running = False
            self._cancel_manual_pulse()
            self._safe_stop_all()
            
            with self._state_lock:
                self._mode = self.MODE_IDLE
                self._mission_active = False
                
            logger.info("RobotStateManager shutdown - all relays OFF")
    
    def cleanup(self):
        """Clean up resources."""
        self.shutdown()
        cleanup_all()


# Global state manager instance
_state_manager = None
_state_manager_lock = threading.Lock()


def get_robot_state_manager():
    """Get or create the global robot state manager instance."""
    global _state_manager
    with _state_manager_lock:
        if _state_manager is None:
            _state_manager = RobotStateManager()
        return _state_manager


def reset_robot_state_manager():
    """Reset the global state manager (for testing or reinitialization)."""
    global _state_manager
    with _state_manager_lock:
        if _state_manager is not None:
            _state_manager.cleanup()
            _state_manager = None


def stop_all_relays():
    """Convenience function to stop all relays."""
    manager = get_robot_state_manager()
    if manager:
        manager.manual_stop()
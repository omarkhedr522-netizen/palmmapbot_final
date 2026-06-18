/**
 * relay_serial_controller.ino
 * 
 * Arduino Uno sketch for PalmMapBot robot relay control.
 * 
 * This sketch receives serial commands from Raspberry Pi and controls
 * a 4-channel relay module to simulate button presses on an RC car remote.
 * 
 * Hardware:
 * - Arduino Uno
 * - 4-channel relay module (active LOW)
 * - Relay IN1 (pin 7) = Forward
 * - Relay IN2 (pin 8) = Backward
 * - Relay IN3/IN4 = Unused
 * 
 * Serial Commands:
 * - "FWD"       : Hold forward button (relay ON continuously)
 * - "BACK"      : Hold backward button (relay ON continuously)
 * - "STOP"      : Release all buttons (all relays OFF)
 * - "FWD_MS:1000" : Press forward for 1000ms, then stop
 * - "BACK_MS:1000": Press backward for 1000ms, then stop
 * 
 * Safety:
 * - Never allows forward and backward relays ON simultaneously
 * - Starts with all relays OFF
 * - Sends serial acknowledgment for commands
 * 
 * Wiring:
 * - Arduino 5V -> Relay VCC
 * - Arduino GND -> Relay GND
 * - Arduino pin 7 -> Relay IN1 (Forward)
 * - Arduino pin 8 -> Relay IN2 (Backward)
 * 
 * Relay to RC Remote:
 * - Forward button pad 1 -> Relay 1 COM
 * - Forward button pad 2 -> Relay 1 NO
 * - Backward button pad 1 -> Relay 2 COM
 * - Backward button pad 2 -> Relay 2 NO
 * 
 * IMPORTANT: Do not connect Arduino 5V/GND to remote button pads.
 * The remote keeps its own battery. Relays act as isolated switches.
 */

// Pin definitions
const int FORWARD_RELAY = 7;  // Relay IN1 for forward movement
const int BACKWARD_RELAY = 8; // Relay IN2 for backward movement

// Serial settings
const long BAUD_RATE = 9600;

// State tracking
bool forwardActive = false;
bool backwardActive = false;

// Non-blocking timer for timed movements
unsigned long timedMovementStartTime = 0;
unsigned long timedMovementDuration = 0;
bool timedMovementActive = false;
char timedMovementType = 0; // 'F' for forward, 'B' for backward

/**
 * Stop all movement - safety function
 * Turns off all relays and resets state
 */
void stopCar() {
  digitalWrite(FORWARD_RELAY, HIGH); // HIGH = relay OFF (active LOW)
  digitalWrite(BACKWARD_RELAY, HIGH);
  forwardActive = false;
  backwardActive = false;
  timedMovementActive = false;
  Serial.println("STOPPED");
}

/**
 * Setup function - runs once at startup
 */
void setup() {
  // Initialize serial communication
  Serial.begin(BAUD_RATE);
  
  // Wait for serial port to stabilize
  delay(100);
  
  // Initialize relay pins as outputs
  pinMode(FORWARD_RELAY, OUTPUT);
  pinMode(BACKWARD_RELAY, OUTPUT);
  
  // Start with all relays OFF (HIGH because active LOW)
  digitalWrite(FORWARD_RELAY, HIGH);
  digitalWrite(BACKWARD_RELAY, HIGH);
  
  Serial.println("Arduino Relay Controller Ready");
  Serial.println("Waiting for commands...");
}

/**
 * Parse and execute timed movement command
 * Format: "FWD_MS:1000" or "BACK_MS:1000"
 */
void executeTimedMovement(char type, unsigned long duration) {
  // Stop any existing movement first
  stopCar();
  
  timedMovementStartTime = millis();
  timedMovementDuration = duration;
  timedMovementActive = true;
  timedMovementType = type;
  
  if (type == 'F') {
    digitalWrite(FORWARD_RELAY, LOW); // LOW = relay ON
    forwardActive = true;
    Serial.print("FWD_MS:");
  } else if (type == 'B') {
    digitalWrite(BACKWARD_RELAY, LOW);
    backwardActive = true;
    Serial.print("BACK_MS:");
  }
  
  Serial.println(duration);
}

/**
 * Main loop - runs continuously
 */
void loop() {
  // Check for serial commands
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim(); // Remove whitespace
    
    if (command.length() > 0) {
      Serial.print("Received: ");
      Serial.println(command);
      
      if (command == "FWD") {
        // Hold forward continuously
        if (backwardActive) {
          digitalWrite(BACKWARD_RELAY, HIGH);
          backwardActive = false;
        }
        timedMovementActive = false;
        digitalWrite(FORWARD_RELAY, LOW);
        forwardActive = true;
        Serial.println("FORWARD");
        
      } else if (command == "BACK") {
        // Hold backward continuously
        if (forwardActive) {
          digitalWrite(FORWARD_RELAY, HIGH);
          forwardActive = false;
        }
        timedMovementActive = false;
        digitalWrite(BACKWARD_RELAY, LOW);
        backwardActive = true;
        Serial.println("BACKWARD");
        
      } else if (command == "STOP") {
        // Emergency stop - release all
        stopCar();
        
      } else if (command.startsWith("FWD_MS:")) {
        // Timed forward movement
        String durationStr = command.substring(7);
        long duration = durationStr.toInt();
        if (duration > 0 && duration <= 30000) {
          executeTimedMovement('F', (unsigned long)duration);
        } else {
          Serial.println("ERROR: Invalid duration");
        }
        
      } else if (command.startsWith("BACK_MS:")) {
        // Timed backward movement
        String durationStr = command.substring(8);
        long duration = durationStr.toInt();
        if (duration > 0 && duration <= 30000) {
          executeTimedMovement('B', (unsigned long)duration);
        } else {
          Serial.println("ERROR: Invalid duration");
        }
        
      } else {
        Serial.print("ERROR: Unknown command: ");
        Serial.println(command);
      }
    }
  }
  
  // Check for timed movement completion (non-blocking)
  if (timedMovementActive) {
    unsigned long elapsed = millis() - timedMovementStartTime;
    if (elapsed >= timedMovementDuration) {
      stopCar();
      Serial.print("TIMED_");
      if (timedMovementType == 'F') {
        Serial.println("FWD_COMPLETE");
      } else {
        Serial.println("BACK_COMPLETE");
      }
    }
  }
  
  // Small delay to prevent serial flooding
  delay(10);
}
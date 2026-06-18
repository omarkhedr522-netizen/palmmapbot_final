# PalmMapBot Robot Control System

## System Overview

**Raspberry Pi = main brain.** Arduino has been **removed** from the control system.

The Raspberry Pi directly controls:
- Relay module through GPIO pins
- MPU6050 through I2C
- LiDAR through UART/I2C
- Camera
- GPS
- Dashboard
- Mission logic
- Database logging

## Mission Logic

```
WAIT FOR DASHBOARD START
→ START MISSION CLICKED
→ MOVE FORWARD
→ SIMPLE TREE YES/NO DETECTION
→ IF TREE YES: STOP IMMEDIATELY
→ READ LiDAR DISTANCE
→ RUN YOLO PALM DETECTION
→ READ GPS + MPU6050
→ STORE ALL DATA
→ CHECK MISSION STILL ACTIVE
→ MOVE FORWARD AGAIN
→ REPEAT
```

**Current system:** GPS-assisted visual mapping. SLAM-ready architecture. Not full SLAM.

## Safety Requirements

### The car must NEVER move automatically when the program starts.

The car only moves when:
1. The dashboard/backend is running
2. The user explicitly clicks **Start Mission** from the dashboard (AUTO mode)
   OR
3. The user switches to **Manual Control** mode and clicks a manual movement button

On program startup:
- Relay outputs must initialize OFF
- Car must be stopped
- Mission status must be IDLE
- No forward movement should happen automatically

## File Structure

```
robot_control/
    relay_gpio_controller.py      # Raspberry Pi GPIO relay control (replaces Arduino)
    tree_binary_inference.py      # Binary tree/no-tree classifier
    database.py                   # SQLite database
    gps_reader.py                 # GPS location
    lidar_reader.py               # LiDAR distance
    mpu6050_reader.py             # MPU6050 IMU
    map_builder.py                # Tree position estimation
    robot_tree_stop_mission.py    # Main mission runtime
    test_*.py                     # Test scripts

training/
    train_tree_binary_classifier.py  # Train binary classifier
    prepare_binary_dataset.py        # Prepare dataset from YOLO format
    generate_no_tree_images.py       # Generate synthetic no-tree images

docs/
    ROBOT_CONTROL_README.md       # This file
```

## Wiring Summary

### Raspberry Pi to Relay Module

| Pi GPIO (BCM) | Relay Input | Function |
|---------------|-------------|----------|
| GPIO17        | IN1         | Forward  |
| GPIO27        | IN2         | Backward |
| GPIO22        | IN3         | Left     |
| GPIO23        | IN4         | Right    |
| GND           | GND         | Ground   |

**Important:**
- Use a relay board compatible with 3.3V GPIO input
- Do NOT power relay coils directly from GPIO pins
- Relay VCC should be connected to appropriate power source

### Relay to RC Remote Control

For each relay channel:
- One remote button pad → Relay COM
- Other remote button pad → Relay NO
- COM/NO polarity does not matter

**Do NOT connect Raspberry Pi GPIO to remote button pads!**

### MPU6050 to Raspberry Pi

| MPU6050 | Raspberry Pi |
|---------|--------------|
| VCC     | 3.3V         |
| GND     | GND          |
| SDA     | GPIO2 (pin 3)|
| SCL     | GPIO3 (pin 5)|

### LiDAR to Raspberry Pi

- UART or I2C configurable
- Check voltage level requirements
- Use level shifter if needed (5V → 3.3V)

## Python Setup

### Install dependencies:
```bash
pip install -r requirements-robot-control.txt
```

### Enable I2C on Raspberry Pi:
```bash
sudo raspi-config
# Interface Options → I2C → Enable
i2cdetect -y 1  # Check I2C devices
```

## Training Binary Classifier

### Prepare dataset:
```
data/tree_binary/
    train/tree/      # Images with trees
    train/no_tree/   # Images without trees
    val/tree/
    val/no_tree/
```

### Train:
```bash
python3 training/train_tree_binary_classifier.py
```

## Testing

```bash
# Test relay GPIO controller
python3 robot_control/test_relay_gpio_controller.py

# Test LiDAR
python3 robot_control/test_lidar_reader.py

# Test MPU6050
python3 robot_control/test_mpu6050_reader.py

# Test binary tree detector
python3 robot_control/test_tree_binary_inference.py --camera
```

## Running Mission

The mission is controlled from the dashboard. Start the dashboard:

```bash
python3 dashboard/app.py
```

Then:
1. Open browser to `http://raspberry-pi-ip:5000`
2. Confirm status is IDLE and car is stopped
3. Click **Start Mission** to begin AUTO mission
4. Click **Stop Mission** to stop
5. Click **Emergency Stop** for immediate stop

### Manual Control

1. Click **Switch to Manual Control** to enable manual section
2. Use Manual Forward/Backward/Left/Right buttons
3. Manual Stop always works
4. Switch to Auto does not start movement until Start Mission is clicked

## Mission States

| State          | Description |
|----------------|-------------|
| IDLE           | Ready, not moving |
| AUTO           | Automatic mission active |
| MANUAL         | Manual control enabled |
| STOPPED        | Mission stopped |
| ERROR          | Error occurred |
| EMERGENCY_STOP | Emergency stop activated |

## Control Modes

### AUTO Mode
- Uses only Forward, Backward, Stop
- Never uses Left or Right
- Started by clicking "Start Mission" on dashboard

### MANUAL Mode
- Uses Forward, Backward, Left, Right, Stop
- Enabled by clicking "Switch to Manual Control"
- Manual movement buttons only work in MANUAL mode

## Dashboard Test Checklist

- [ ] Backend starts with mode IDLE
- [ ] Relay outputs OFF on startup
- [ ] Start Mission makes car move forward
- [ ] Stop Mission stops car
- [ ] Emergency Stop stops car
- [ ] Switch to Manual Control stops car
- [ ] Manual Forward does nothing in AUTO/IDLE
- [ ] Manual Backward does nothing in AUTO/IDLE
- [ ] Manual Left does nothing in AUTO/IDLE
- [ ] Manual Right does nothing in AUTO/IDLE
- [ ] Manual Forward works only in MANUAL
- [ ] Manual Backward works only in MANUAL
- [ ] Manual Left works only in MANUAL
- [ ] Manual Right works only in MANUAL
- [ ] Manual Stop works in all modes
- [ ] Switch to Auto does not start movement until Start Mission is clicked
- [ ] AUTO mission only uses Forward/Backward/Stop
- [ ] AUTO mission never uses Left/Right

## Safety Testing

1. **First test with wheels lifted off ground**
2. Test dashboard buttons with wheels lifted
3. Confirm startup does not move car
4. Confirm manual buttons do nothing unless manual mode is enabled
5. Test camera detection with wheels lifted
6. Test full mission slowly in open space
7. **Keep manual power cutoff ready**
8. **Never test near people, stairs, roads, pets, water, or fragile objects**

## Troubleshooting

### GPIO not available
```bash
pip install gpiozero
# or
pip install RPi.GPIO
```

### I2C device not found
```bash
sudo raspi-config  # Enable I2C
i2cdetect -y 1     # Check devices
```

### Relay not activating
- Check wiring
- Verify relay board is 3.3V compatible
- Check relay VCC power source

### Model not found
```bash
python3 training/train_tree_binary_classifier.py
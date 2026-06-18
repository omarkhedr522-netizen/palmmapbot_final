# Raspberry Pi Deployment Guide

Complete guide for deploying PalmMapBot on Raspberry Pi.

## Hardware Requirements

### Raspberry Pi
- **Recommended**: Raspberry Pi 4 Model B (4GB or 8GB)
- **Minimum**: Raspberry Pi 3 Model B+
- **OS**: Raspberry Pi OS (64-bit) Bullseye or later

### Sensors & Hardware
1. **Camera**: USB webcam or Raspberry Pi Camera Module
2. **GPS**: USB GPS receiver (e.g., VK-172)
3. **LiDAR**: RPLiDAR A1/A2/A3 or similar
4. **IMU**: MPU6050 (I2C)
5. **Relay Module**: 4-channel 5V relay module
6. **RC Remote**: Modified RC car remote (forward/backward/left/right)

### Wiring
See `docs/ROBOT_CONTROL_README.md` for detailed wiring diagrams.

## Software Installation

### Option A: Using Git (Recommended)

#### 1. Update System
```bash
sudo apt update
sudo apt upgrade -y
sudo reboot
```

#### 2. Install System Dependencies
```bash
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    libatlas-base-dev \
    libopenblas-dev \
    libhdf5-serial-dev \
    libhdf5-dev \
    libffi-dev \
    libssl-dev \
    i2c-tools \
    python3-smbus \
    git \
    cmake \
    libjpeg-dev \
    libtiff5-dev \
    libjasper-dev \
    libpng-dev
```

#### 3. Enable I2C for MPU6050
```bash
sudo raspi-config
# Navigate to: Interface Options → I2C → Enable
sudo reboot
```

#### 4. Clone Repository
```bash
cd ~
git clone <your-repo-url> PalmMapBot
cd PalmMapBot/Palmmapbot
```

#### 5. Setup Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

#### 6. Install Python Dependencies
```bash
pip install -r requirements-robot-control.txt
```

### Option B: Using Setup Script

If you copied the project files manually, run the setup script:

```bash
cd ~/Palmmapbot
chmod +x setup_raspberry_pi.sh
./setup_raspberry_pi.sh
```

This automates all the steps above.

### 6. Create Required Directories
```bash
mkdir -p data models captured_frames validation_assets
```

### 7. Copy Model Files
Place your trained models in the `models/` directory:
- `tree_binary_classifier.pt` - Binary tree classifier
- `palm_tree_detector.pt` - YOLO palm tree detector (optional)

## Configuration

### Environment Variables
Create a `.env` file in the project root:

```bash
# Camera
CAMERA_INDEX=0
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
CAMERA_FPS=30

# Binary Classifier
BINARY_THRESHOLD=0.60
BINARY_CHECK_INTERVAL=0.1

# GPS (set to false if using real GPS)
USE_DUMMY_GPS=false
GPS_PORT=/dev/ttyUSB0

# LiDAR (set to false if using real LiDAR)
USE_DUMMY_LIDAR=false
LIDAR_PORT=/dev/ttyUSB1

# MPU6050 (set to false if using real MPU)
USE_DUMMY_MPU=false

# Dashboard
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=5000
```

### GPIO Pin Configuration
Edit `robot_control/config.py` if you need to change GPIO pins:

```python
GPIO_FORWARD = 17
GPIO_BACKWARD = 27
GPIO_LEFT = 22
GPIO_RIGHT = 23
GPIO_ACTIVE_LOW = True
```

## Testing Hardware

### 1. Test GPIO (Relay Controller)
```bash
source venv/bin/activate
python -c "
from robot_control.relay_gpio_controller import RelayGPIOController
controller = RelayGPIOController()
controller.forward()
print('Forward relay activated')
import time; time.sleep(1)
controller.stop()
print('All relays stopped')
controller.cleanup()
"
```

### 2. Test Camera
```bash
python -c "
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    cv2.imwrite('test_camera.jpg', frame)
    print('Camera test successful - saved test_camera.jpg')
else:
    print('Camera test failed')
cap.release()
"
```

### 3. Test MPU6050
```bash
# Check I2C detection
sudo i2cdetect -y 1

# Should show 68 if MPU6050 is connected properly
python -c "
from robot_control.mpu6050_reader import MPU6050Reader
mpu = MPU6050Reader()
mpu.start()
import time; time.sleep(1)
data = mpu.get_data()
print(f'MPU Data: {data}')
mpu.stop()
"
```

### 4. Test GPS
```bash
python -c "
from robot_control.gps_reader import GPSReader
gps = GPSReader('/dev/ttyUSB0', use_dummy=False)
gps.start()
import time; time.sleep(2)
loc = gps.get_location()
print(f'GPS Location: {loc}')
gps.stop()
"
```

### 5. Test LiDAR
```bash
python -c "
from robot_control.lidar_reader import LidarReader
lidar = LidarReader()
lidar.start()
import time; time.sleep(2)
dist = lidar.get_distance()
print(f'LiDAR Distance: {dist}')
lidar.stop()
"
```

## Running the Dashboard

### Start the Dashboard
```bash
cd ~/PalmMapBot/Palmmapbot
source venv/bin/activate
python3 dashboard/app.py
```

### Access the Dashboard
Open a web browser and navigate to:
- **Local**: http://localhost:5000
- **Network**: http://<raspberry-pi-ip>:5000

Find your Raspberry Pi IP with:
```bash
hostname -I
```

## Running as a Service (Auto-start on Boot)

### 1. Create Systemd Service
```bash
sudo nano /etc/systemd/system/palmmapbot.service
```

### 2. Add Service Configuration
```ini
[Unit]
Description=PalmMapBot Dashboard
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/PalmMapBot/Palmmapbot
ExecStart=/home/pi/PalmMapBot/Palmmapbot/venv/bin/python3 dashboard/app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 3. Enable and Start Service
```bash
sudo systemctl daemon-reload
sudo systemctl enable palmmapbot.service
sudo systemctl start palmmapbot.service

# Check status
sudo systemctl status palmmapbot.service

# View logs
sudo journalctl -u palmmapbot.service -f
```

## Troubleshooting

### GPIO Permission Issues
```bash
# Add user to gpio group
sudo usermod -a -G gpio $USER
sudo reboot
```

### I2C Not Working
```bash
# Check if I2C is enabled
lsmod | grep i2c

# Enable if not loaded
sudo modprobe i2c-dev
```

### Camera Not Found
```bash
# List video devices
ls -l /dev/video*

# Test with different index
CAMERA_INDEX=1 python3 dashboard/app.py
```

### USB Device Permissions
```bash
# Create udev rules for USB devices
sudo nano /etc/udev/rules.d/99-palmmapbot.rules

# Add lines (replace with your device IDs):
# SUBSYSTEM=="usb", ATTRS{idVendor}=="<vendor>", ATTRS{idProduct}=="<product>", MODE="0666"

sudo udevadm control --reload-rules
sudo udevadm trigger
```

### Performance Optimization
```bash
# Overclock Raspberry Pi 4 (optional, use with caution)
sudo nano /boot/config.txt

# Add at the end:
# over_voltage=2
# arm_freq=1750
# gpu_freq=500

sudo reboot
```

## Monitoring and Logs

### View Dashboard Logs
```bash
# If running as service
sudo journalctl -u palmmapbot.service -f

# If running manually
# Check the terminal output
```

### Monitor System Resources
```bash
# CPU and Memory
htop

# Temperature
vcgencmd measure_temp

# GPU Memory
vcgencmd get_mem gpu
```

## Safety Checklist Before First Run

1. ☐ All GPIO connections verified
2. ☐ Relay module tested independently
3. ☐ Camera feed working
4. ☐ All sensors responding
5. ☐ Dashboard accessible
6. ☐ Emergency stop tested
7. ☐ Car on stand (wheels off ground)
8. ☐ Remote control direction verified
9. ☐ Battery fully charged
10. ☐ Network connection stable

## Next Steps

1. **Calibrate Sensors**: Adjust thresholds in `config.py`
2. **Test Mission**: Run a short test mission in a controlled area
3. **Fine-tune AI**: Collect more data if needed
4. **Deploy**: Set up in your palm grove

## Support

For issues, check:
- `docs/ROBOT_CONTROL_README.md` - Detailed technical documentation
- `robot_control/config.py` - Configuration options
- GitHub Issues - Report bugs and request features
#!/bin/bash

# PalmMapBot Raspberry Pi Setup Script
# This script automates the installation and configuration of PalmMapBot on Raspberry Pi

set -e  # Exit on error

echo "============================================="
echo "  PalmMapBot Raspberry Pi Setup Script"
echo "============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "Please do not run this script as root"
    exit 1
fi

# Check if running on Raspberry Pi
if [ ! -f /proc/device-tree/model ]; then
    print_error "This script is designed for Raspberry Pi"
    print_info "Model file not found at /proc/device-tree/model"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Get project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME=$(basename "$PROJECT_DIR")

print_info "Project directory: $PROJECT_DIR"

# Step 1: Update system
echo ""
print_info "Step 1/8: Updating system packages..."
sudo apt update
sudo apt upgrade -y
print_success "System updated"

# Step 2: Install system dependencies
echo ""
print_info "Step 2/8: Installing system dependencies..."
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
    libpng-dev \
    libgtk-3-dev
print_success "System dependencies installed"

# Step 3: Enable I2C
echo ""
print_info "Step 3/8: Enabling I2C interface..."
if grep -q "dtparam=i2c_arm=on" /boot/config.txt 2>/dev/null || grep -q "dtparam=i2c_arm=on" /boot/firmware/config.txt 2>/dev/null; then
    print_success "I2C already enabled"
else
    if [ -f /boot/config.txt ]; then
        echo "dtparam=i2c_arm=on" | sudo tee -a /boot/config.txt > /dev/null
    elif [ -f /boot/firmware/config.txt ]; then
        echo "dtparam=i2c_arm=on" | sudo tee -a /boot/firmware/config.txt > /dev/null
    fi
    print_success "I2C enabled (reboot required)"
fi

# Step 4: Create virtual environment
echo ""
print_info "Step 4/8: Creating Python virtual environment..."
if [ -d "venv" ]; then
    print_info "Virtual environment already exists"
else
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
pip install --upgrade pip

# Step 5: Install Python dependencies
echo ""
print_info "Step 5/8: Installing Python dependencies..."

if [ -f "requirements.txt" ]; then
    print_info "Installing from requirements.txt..."
    pip install -r requirements.txt
fi

if [ -f "requirements-robot-control.txt" ]; then
    print_info "Installing from requirements-robot-control.txt..."
    pip install -r requirements-robot-control.txt
fi

print_success "Python dependencies installed"

# Step 6: Create required directories
echo ""
print_info "Step 6/8: Creating required directories..."
mkdir -p data models captured_frames validation_assets
print_success "Directories created"

# Step 7: Add user to gpio group
echo ""
print_info "Step 7/8: Configuring GPIO permissions..."
sudo usermod -a -G gpio $USER
print_success "GPIO permissions configured (reboot required)"

# Step 8: Create .env file template
echo ""
print_info "Step 8/8: Creating environment configuration..."
if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# PalmMapBot Environment Configuration
# Edit these values according to your hardware setup

# Camera Settings
CAMERA_INDEX=0
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
CAMERA_FPS=30

# Binary Tree Classifier
BINARY_THRESHOLD=0.60
BINARY_CHECK_INTERVAL=0.1

# GPS Settings (set USE_DUMMY_GPS=false for real GPS)
USE_DUMMY_GPS=true
GPS_PORT=/dev/ttyUSB0
GPS_BAUD=9600

# LiDAR Settings (set USE_DUMMY_LIDAR=false for real LiDAR)
USE_DUMMY_LIDAR=true
LIDAR_PORT=/dev/ttyUSB1
LIDAR_BAUD=115200

# MPU6050 Settings (set USE_DUMMY_MPU=false for real MPU)
USE_DUMMY_MPU=true
I2C_BUS=1

# Dashboard Settings
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=5000
DASHBOARD_REFRESH_INTERVAL=5

# Logging
LOG_LEVEL=INFO
EOF
    print_success "Environment file created (.env)"
else
    print_info "Environment file already exists (.env)"
fi

# Summary
echo ""
echo "============================================="
echo "  Setup Complete!"
echo "============================================="
echo ""
print_success "All dependencies installed"
print_success "Virtual environment configured"
print_success "GPIO permissions set"
print_success "Environment file created"
echo ""
print_info "Next steps:"
echo "  1. Reboot your Raspberry Pi:"
echo "     sudo reboot"
echo ""
echo "  2. After reboot, activate the virtual environment:"
echo "     cd $PROJECT_DIR"
echo "     source venv/bin/activate"
echo ""
echo "  3. Test the hardware:"
echo "     python -c \"from robot_control.relay_gpio_controller import RelayGPIOController; c = RelayGPIOController(); c.forward(); import time; time.sleep(1); c.stop(); c.cleanup()\""
echo ""
echo "  4. Start the dashboard:"
echo "     python dashboard/app.py"
echo ""
echo "  5. Open browser to: http://localhost:5000"
echo ""
print_info "For detailed instructions, see: docs/RASPBERRY_PI_DEPLOYMENT.md"
echo ""

# Ask if user wants to install as system service
echo ""
read -p "Install PalmMapBot as a system service (auto-start on boot)? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Creating systemd service..."
    
    SERVICE_FILE="/etc/systemd/system/palmmapbot.service"
    sudo bash -c "cat > $SERVICE_FILE" << EOF
[Unit]
Description=PalmMapBot Dashboard
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$PROJECT_DIR/venv/bin/python3 dashboard/app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable palmmapbot.service
    
    print_success "System service installed"
    print_info "Start the service with: sudo systemctl start palmmapbot.service"
    print_info "Check status with: sudo systemctl status palmmapbot.service"
fi

echo ""
print_success "Setup complete! Please reboot your Raspberry Pi."
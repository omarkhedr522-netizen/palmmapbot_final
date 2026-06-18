"""
map_builder.py

Map builder module for PalmMapBot robot.

This module provides basic map-ready tree position estimation.
It uses GPS-assisted visual mapping - the current approach uses
the robot's GPS position at the stopping point as the approximate
tree position.

This is NOT full SLAM. It's a placeholder architecture that will
be improved with:
- Heading/yaw integration
- Visual odometry
- Wheel encoders
- ORB-SLAM3
- RTAB-Map
- Depth camera fusion
- Better LiDAR integration

Current Mapping Mode: gps_assisted_visual_mapping
"""

import os
import sys
import logging
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default configuration
EARTH_RADIUS_M = 6371000  # Earth radius in meters

# Mapping modes
MAPPING_MODE_GPS_ASSISTED = "gps_assisted_visual_mapping"
MAPPING_MODE_SLAM_READY = "slam_ready_placeholder"


def estimate_tree_position(robot_latitude, robot_longitude, 
                          lidar_distance_m, lidar_valid,
                          robot_heading_deg=None):
    """
    Estimate tree position based on robot position and LiDAR distance.
    
    Current implementation (GPS-assisted visual mapping):
    - Uses robot GPS position as approximate tree position
    - LiDAR distance is logged but not used for position offset
    - This is a simple placeholder until full SLAM is implemented
    
    Future implementation (SLAM-ready):
    - Use robot heading to offset tree position
    - Use LiDAR distance for precise offset
    - Fuse with visual odometry
    - Use wheel encoders for dead reckoning
    
    Args:
        robot_latitude: Robot's GPS latitude at stopping point
        robot_longitude: Robot's GPS longitude at stopping point
        lidar_distance_m: Distance to tree from LiDAR (meters)
        lidar_valid: Whether LiDAR reading is valid
        robot_heading_deg: Robot's heading/yaw in degrees (optional)
        
    Returns:
        tuple: (estimated_latitude, estimated_longitude, mapping_mode)
    """
    # If no GPS data, return None
    if robot_latitude is None or robot_longitude is None:
        logger.warning("No GPS data available for tree position estimation")
        return None, None, "no_gps_data"
        
    # If LiDAR is not valid, just use robot position
    if not lidar_valid or lidar_distance_m is None:
        logger.info("LiDAR invalid, using robot GPS as tree position")
        return robot_latitude, robot_longitude, MAPPING_MODE_GPS_ASSISTED
        
    # If no heading available, use robot position
    if robot_heading_deg is None:
        logger.info("No heading available, using robot GPS as tree position")
        return robot_latitude, robot_longitude, MAPPING_MODE_GPS_ASSISTED
        
    # TODO: Full SLAM implementation would go here
    # For now, we use a simple offset calculation based on heading and distance
    
    try:
        # Convert heading to radians
        heading_rad = math.radians(robot_heading_deg)
        
        # Calculate offset in meters (assuming tree is in front of robot)
        # Note: This is a simplified model. Real implementation would need
        # to account for camera offset from robot center, tree detection
        # angle in camera frame, etc.
        
        # Offset distance (use LiDAR distance as approximation)
        offset_distance = lidar_distance_m
        
        # Calculate latitude/longitude offset
        # Latitude: 1 degree ≈ 111,320 meters
        # Longitude: 1 degree ≈ 111,320 * cos(latitude) meters
        
        lat_offset = (offset_distance * math.cos(heading_rad)) / 111320.0
        lon_offset = (offset_distance * math.sin(heading_rad)) / (111320.0 * math.cos(math.radians(robot_latitude)))
        
        estimated_lat = robot_latitude + lat_offset
        estimated_lon = robot_longitude + lon_offset
        
        logger.info(f"Tree position estimated: offset {offset_distance:.2f}m at heading {robot_heading_deg:.1f}°")
        
        return estimated_lat, estimated_lon, MAPPING_MODE_GPS_ASSISTED
        
    except Exception as e:
        logger.error(f"Error estimating tree position: {e}")
        # Fall back to robot position
        return robot_latitude, robot_longitude, MAPPING_MODE_GPS_ASSISTED


def estimate_tree_position_simple(robot_latitude, robot_longitude, 
                                  lidar_distance_m=None, lidar_valid=False,
                                  robot_heading_deg=None):
    """
    Simple tree position estimation - just use robot GPS.
    
    This is the safest default: record where the robot was when
    it detected the tree, without trying to estimate exact tree position.
    
    Args:
        robot_latitude: Robot's GPS latitude
        robot_longitude: Robot's GPS longitude
        lidar_distance_m: Distance to tree (logged but not used for position)
        lidar_valid: Whether LiDAR reading is valid
        robot_heading_deg: Robot heading (not used in simple mode)
        
    Returns:
        tuple: (estimated_latitude, estimated_longitude, mapping_mode)
    """
    if robot_latitude is None or robot_longitude is None:
        return None, None, "no_gps_data"
        
    return robot_latitude, robot_longitude, MAPPING_MODE_GPS_ASSISTED


def calculate_distance_between_points(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two GPS coordinates using Haversine formula.
    
    Args:
        lat1, lon1: First point coordinates (degrees)
        lat2, lon2: Second point coordinates (degrees)
        
    Returns:
        float: Distance in meters
    """
    if any(v is None for v in [lat1, lon1, lat2, lon2]):
        return None
        
    try:
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        # Haversine formula
        a = (math.sin(delta_lat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        distance = EARTH_RADIUS_M * c
        return distance
        
    except Exception as e:
        logger.error(f"Error calculating distance: {e}")
        return None


def calculate_bearing_between_points(lat1, lon1, lat2, lon2):
    """
    Calculate bearing from point 1 to point 2.
    
    Args:
        lat1, lon1: First point coordinates (degrees)
        lat2, lon2: Second point coordinates (degrees)
        
    Returns:
        float: Bearing in degrees (0-360)
    """
    if any(v is None for v in [lat1, lon1, lat2, lon2]):
        return None
        
    try:
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lon = math.radians(lon2 - lon1)
        
        # Calculate bearing
        x = math.sin(delta_lon) * math.cos(lat2_rad)
        y = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon))
        
        bearing_rad = math.atan2(x, y)
        bearing_deg = math.degrees(bearing_rad)
        
        # Normalize to 0-360
        bearing_deg = (bearing_deg + 360) % 360
        
        return bearing_deg
        
    except Exception as e:
        logger.error(f"Error calculating bearing: {e}")
        return None


class MapBuilder:
    """Map builder for tree position estimation and mapping."""
    
    def __init__(self, use_simple_mode=True):
        """
        Initialize map builder.
        
        Args:
            use_simple_mode: If True, use simple GPS position as tree position.
                           If False, attempt offset calculation with LiDAR.
        """
        self.use_simple_mode = use_simple_mode
        
    def estimate_tree_position(self, robot_latitude, robot_longitude,
                              lidar_distance_m=None, lidar_valid=False,
                              robot_heading_deg=None):
        """
        Estimate tree position.
        
        Args:
            robot_latitude: Robot GPS latitude
            robot_longitude: Robot GPS longitude
            lidar_distance_m: LiDAR distance to tree
            lidar_valid: Whether LiDAR reading is valid
            robot_heading_deg: Robot heading
            
        Returns:
            tuple: (lat, lon, mapping_mode)
        """
        if self.use_simple_mode:
            return estimate_tree_position_simple(
                robot_latitude, robot_longitude,
                lidar_distance_m, lidar_valid,
                robot_heading_deg
            )
        else:
            return estimate_tree_position(
                robot_latitude, robot_longitude,
                lidar_distance_m, lidar_valid,
                robot_heading_deg
            )


# Global map builder instance
_map_builder = None


def get_map_builder(use_simple_mode=True):
    """
    Get or create global map builder instance.
    
    Args:
        use_simple_mode: Use simple GPS position mode
        
    Returns:
        MapBuilder instance
    """
    global _map_builder
    if _map_builder is None:
        _map_builder = MapBuilder(use_simple_mode=use_simple_mode)
    return _map_builder


# Test function
if __name__ == "__main__":
    print("Map Builder Test")
    print("=" * 40)
    
    # Test position estimation
    robot_lat = 29.203451
    robot_lon = 25.519833
    lidar_dist = 2.5
    heading = 45.0
    
    print("\nTest 1: Simple mode (use robot position)")
    lat, lon, mode = estimate_tree_position_simple(
        robot_lat, robot_lon, lidar_dist, True, heading
    )
    print(f"  Robot: ({robot_lat}, {robot_lon})")
    print(f"  Tree: ({lat}, {lon})")
    print(f"  Mode: {mode}")
    
    print("\nTest 2: Offset mode (calculate offset)")
    lat, lon, mode = estimate_tree_position(
        robot_lat, robot_lon, lidar_dist, True, heading
    )
    print(f"  Robot: ({robot_lat}, {robot_lon})")
    print(f"  Tree: ({lat}, {lon})")
    print(f"  Mode: {mode}")
    
    print("\nTest 3: Distance calculation")
    dist = calculate_distance_between_points(robot_lat, robot_lon, lat, lon)
    print(f"  Distance: {dist:.2f}m")
    
    print("\nTest 4: Bearing calculation")
    bearing = calculate_bearing_between_points(robot_lat, robot_lon, lat, lon)
    print(f"  Bearing: {bearing:.1f}°")
    
    print("\nTest complete!")
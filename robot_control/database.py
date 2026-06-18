"""
database.py

SQLite database module for PalmMapBot tree mapping.

This module handles all database operations for storing tree detection
records, including GPS coordinates, sensor readings, YOLO detections,
and images.

Database: data/palmmapbot.db
Table: trees

The database stores comprehensive records for each detected tree:
- Timestamp
- GPS location (raw and estimated)
- LiDAR distance
- MPU6050 IMU data
- YOLO detection results
- Captured image path
- Robot status
"""

import os
import sys
import sqlite3
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                "data", "palmmapbot.db")


def get_db_path():
    """Get the database path, creating the data directory if needed."""
    db_dir = os.path.dirname(DEFAULT_DB_PATH)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
        logger.info(f"Created data directory: {db_dir}")
    return DEFAULT_DB_PATH


def initialize_database(db_path=None):
    """
    Initialize the SQLite database and create tables if they don't exist.
    
    Args:
        db_path: Path to database file (uses default if None)
        
    Returns:
        Connection object
    """
    if db_path is None:
        db_path = get_db_path()
        
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create trees table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                
                -- GPS data
                latitude REAL,
                longitude REAL,
                gps_valid INTEGER DEFAULT 0,
                
                -- Estimated tree position (may differ from robot position)
                estimated_latitude REAL,
                estimated_longitude REAL,
                
                -- Binary tree detector confidence
                tree_present_confidence REAL,
                
                -- LiDAR data
                lidar_distance_m REAL,
                lidar_valid INTEGER DEFAULT 0,
                
                -- MPU6050 IMU data
                accel_x REAL,
                accel_y REAL,
                accel_z REAL,
                gyro_x REAL,
                gyro_y REAL,
                gyro_z REAL,
                mpu_valid INTEGER DEFAULT 0,
                
                -- YOLO detection results
                yolo_confidence REAL,
                class_name TEXT,
                bbox_x1 INTEGER,
                bbox_y1 INTEGER,
                bbox_x2 INTEGER,
                bbox_y2 INTEGER,
                
                -- Captured image
                image_path TEXT,
                
                -- Robot and mapping info
                robot_status TEXT,
                mapping_mode TEXT,
                
                -- Tree status (active, inactive, removed)
                status TEXT DEFAULT 'active'
            )
        """)
        
        # Create index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trees_timestamp 
            ON trees(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trees_status 
            ON trees(status)
        """)
        
        conn.commit()
        logger.info(f"Database initialized: {db_path}")
        
        return conn
        
    except sqlite3.Error as e:
        logger.error(f"Database initialization error: {e}")
        raise


def insert_tree_record(
    db_path=None,
    timestamp=None,
    latitude=None,
    longitude=None,
    gps_valid=False,
    estimated_latitude=None,
    estimated_longitude=None,
    tree_present_confidence=None,
    lidar_distance_m=None,
    lidar_valid=False,
    accel_x=None,
    accel_y=None,
    accel_z=None,
    gyro_x=None,
    gyro_y=None,
    gyro_z=None,
    mpu_valid=False,
    yolo_confidence=None,
    class_name=None,
    bbox_x1=None,
    bbox_y1=None,
    bbox_x2=None,
    bbox_y2=None,
    image_path=None,
    robot_status=None,
    mapping_mode=None,
    status='active'
):
    """
    Insert a tree detection record into the database.
    
    All parameters except db_path are optional. Missing values will be NULL.
    
    Args:
        db_path: Path to database file
        timestamp: ISO format timestamp (auto-generated if None)
        latitude: GPS latitude
        longitude: GPS longitude
        gps_valid: Whether GPS has valid fix
        estimated_latitude: Estimated tree latitude
        estimated_longitude: Estimated tree longitude
        tree_present_confidence: Binary classifier confidence
        lidar_distance_m: LiDAR distance in meters
        lidar_valid: Whether LiDAR reading is valid
        accel_x, accel_y, accel_z: Accelerometer readings
        gyro_x, gyro_y, gyro_z: Gyroscope readings
        mpu_valid: Whether MPU6050 data is valid
        yolo_confidence: YOLO detection confidence
        class_name: YOLO class name
        bbox_x1, bbox_y1, bbox_x2, bbox_y2: Bounding box coordinates
        image_path: Path to captured frame image
        robot_status: Robot status string
        mapping_mode: Mapping mode description
        status: Tree status (active/inactive/removed)
        
    Returns:
        int: ID of inserted record, or -1 on error
    """
    if db_path is None:
        db_path = get_db_path()
        
    if timestamp is None:
        timestamp = datetime.now().isoformat()
        
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO trees (
                timestamp,
                latitude, longitude, gps_valid,
                estimated_latitude, estimated_longitude,
                tree_present_confidence,
                lidar_distance_m, lidar_valid,
                accel_x, accel_y, accel_z,
                gyro_x, gyro_y, gyro_z, mpu_valid,
                yolo_confidence, class_name,
                bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                image_path,
                robot_status, mapping_mode,
                status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            latitude, longitude, 1 if gps_valid else 0,
            estimated_latitude, estimated_longitude,
            tree_present_confidence,
            lidar_distance_m, 1 if lidar_valid else 0,
            accel_x, accel_y, accel_z,
            gyro_x, gyro_y, gyro_z, 1 if mpu_valid else 0,
            yolo_confidence, class_name,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            image_path,
            robot_status, mapping_mode,
            status
        ))
        
        record_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Tree record inserted: ID={record_id}")
        return record_id
        
    except sqlite3.Error as e:
        logger.error(f"Database insert error: {e}")
        return -1


def get_all_trees(db_path=None, limit=100):
    """
    Retrieve all tree records from the database.
    
    Args:
        db_path: Path to database file
        limit: Maximum number of records to return
        
    Returns:
        list: List of tree records as dictionaries
    """
    if db_path is None:
        db_path = get_db_path()
        
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM trees 
            ORDER BY id DESC 
            LIMIT ?
        """, (limit,))
        
        trees = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return trees
        
    except sqlite3.Error as e:
        logger.error(f"Database query error: {e}")
        return []


def get_tree_count(db_path=None):
    """
    Get total number of tree records.
    
    Args:
        db_path: Path to database file
        
    Returns:
        int: Total tree count
    """
    if db_path is None:
        db_path = get_db_path()
        
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM trees")
        count = cursor.fetchone()[0]
        conn.close()
        
        return count
        
    except sqlite3.Error as e:
        logger.error(f"Database count error: {e}")
        return 0


def update_tree_status(tree_id, status, db_path=None):
    """
    Update the status of a tree record.
    
    Args:
        tree_id: ID of tree record
        status: New status (active/inactive/removed)
        db_path: Path to database file
        
    Returns:
        bool: True if updated successfully
    """
    if db_path is None:
        db_path = get_db_path()
        
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE trees SET status = ? WHERE id = ?
        """, (status, tree_id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Tree {tree_id} status updated to: {status}")
        return True
        
    except sqlite3.Error as e:
        logger.error(f"Database update error: {e}")
        return False


# Convenience class for database operations
class TreeDatabase:
    """Convenience class for tree database operations."""
    
    def __init__(self, db_path=None):
        """
        Initialize tree database.
        
        Args:
            db_path: Path to database file
        """
        self.db_path = db_path if db_path else get_db_path()
        self.conn = None
        
    def connect(self):
        """Connect to database and ensure tables exist."""
        self.conn = initialize_database(self.db_path)
        return self.conn
        
    def insert(self, **kwargs):
        """Insert a tree record."""
        return insert_tree_record(self.db_path, **kwargs)
        
    def get_all(self, limit=100):
        """Get all tree records."""
        return get_all_trees(self.db_path, limit)
        
    def count(self):
        """Get total tree count."""
        return get_tree_count(self.db_path)
        
    def update_status(self, tree_id, status):
        """Update tree status."""
        return update_tree_status(tree_id, status, self.db_path)
        
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


# Test function
if __name__ == "__main__":
    print("Tree Database Test")
    print("=" * 40)
    
    # Initialize database
    db = TreeDatabase()
    db.connect()
    
    # Insert a test record
    test_id = db.insert(
        latitude=29.123456,
        longitude=25.654321,
        gps_valid=True,
        estimated_latitude=29.123460,
        estimated_longitude=25.654325,
        tree_present_confidence=0.85,
        lidar_distance_m=2.5,
        lidar_valid=True,
        yolo_confidence=0.92,
        class_name="palm_tree",
        bbox_x1=100,
        bbox_y1=50,
        bbox_x2=300,
        bbox_y2=400,
        robot_status="stopped",
        mapping_mode="gps_assisted_visual_mapping"
    )
    
    print(f"Inserted test record: ID={test_id}")
    
    # Query records
    trees = db.get_all(limit=10)
    print(f"\nTotal trees in database: {db.count()}")
    print(f"Recent records:")
    for tree in trees:
        print(f"  ID={tree['id']}, Lat={tree['latitude']}, Lon={tree['longitude']}, "
              f"Confidence={tree['yolo_confidence']}")
    
    db.close()
    print("\nTest complete!")
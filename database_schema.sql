-- =========================================================
-- Real-Time Object Detection & Logging Platform
-- Assignment 5
-- Database Schema
-- =========================================================


-- ---------------------------------------------------------
-- Create Database
-- ---------------------------------------------------------

CREATE DATABASE IF NOT EXISTS vision_platform;

-- Select Database
USE vision_platform;


-- ---------------------------------------------------------
-- Create Detection Logs Table
-- ---------------------------------------------------------

CREATE TABLE IF NOT EXISTS detection_logs (

    -- Primary Key
    log_id INT AUTO_INCREMENT PRIMARY KEY,

    -- Date and time of detection
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,

    -- Detected object name
    object_class VARCHAR(100) NOT NULL,

    -- YOLO confidence score
    confidence FLOAT NOT NULL,

    -- Bounding box coordinates
    bbox_x INT NOT NULL,
    bbox_y INT NOT NULL,
    bbox_w INT NOT NULL,
    bbox_h INT NOT NULL,

    -- Helpful for recent-log queries and object filtering
    INDEX idx_timestamp (timestamp),
    INDEX idx_object_class (object_class)

);

-- ---------------------------------------------------------
-- Verify Table
-- ---------------------------------------------------------

DESCRIBE detection_logs;


-- ---------------------------------------------------------
-- Test Query
-- ---------------------------------------------------------

SELECT *
FROM detection_logs
ORDER BY timestamp DESC
LIMIT 20;
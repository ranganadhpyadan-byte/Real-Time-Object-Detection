# database.py
# MySQL Database Module
# Real-Time Object Detection & Logging Platform

import logging
import queue
import threading

import mysql.connector
from mysql.connector import Error
from config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER

logger = logging.getLogger(__name__)


class DatabaseManager:

    def __init__(self):
        """
        Initialize database configuration using environment variables.
        """

        self.host = DB_HOST
        self.user = DB_USER
        self.password = DB_PASSWORD
        self.database = DB_NAME
        self.port = DB_PORT

        self.connection = None
        self.last_error = None

        try:
            self.connect()
        except Error as e:
            self.last_error = str(e)

    # -----------------------------------------------------
    # CONNECT TO MYSQL
    # -----------------------------------------------------
    def connect(self):
        """
        Connect to MySQL database.
        """

        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                port=self.port
            )

            if self.connection.is_connected():
                cursor = self.connection.cursor()
                try:
                    cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS detection_logs (
                            log_id INT AUTO_INCREMENT PRIMARY KEY,
                            timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                            object_class VARCHAR(100) NOT NULL,
                            confidence FLOAT NOT NULL,
                            bbox_x INT NOT NULL,
                            bbox_y INT NOT NULL,
                            bbox_w INT NOT NULL,
                            bbox_h INT NOT NULL,
                            INDEX idx_timestamp (timestamp),
                            INDEX idx_object_class (object_class)
                        )
                        """
                    )
                    self.connection.commit()
                finally:
                    cursor.close()

                self.last_error = None
                print("MySQL database connected successfully.")
                return True

        except Error as e:
            self.connection = None
            self.last_error = str(e)
            print(f"MySQL connection error: {e}")
            raise

        return False

    # -----------------------------------------------------
    # CHECK DATABASE CONNECTION
    # -----------------------------------------------------
    def ensure_connection(self):
        """
        Reconnect if the MySQL connection has been lost.
        """

        try:
            if self.connection is None or not self.connection.is_connected():
                self.connect()

        except Error as e:
            print(f"Database reconnection error: {e}")
            raise

    # -----------------------------------------------------
    # INSERT DETECTION
    # -----------------------------------------------------
    def insert_detection(
        self,
        object_class,
        confidence,
        bbox_x,
        bbox_y,
        bbox_w,
        bbox_h
    ):
        """
        Insert a detection event into MySQL.
        """

        cursor = None

        try:
            self.ensure_connection()

            cursor = self.connection.cursor()

            query = """
                INSERT INTO detection_logs
                (
                    timestamp,
                    object_class,
                    confidence,
                    bbox_x,
                    bbox_y,
                    bbox_w,
                    bbox_h
                )
                VALUES
                (
                    NOW(),
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s
                )
            """

            values = (
                object_class,
                float(confidence),
                int(bbox_x),
                int(bbox_y),
                int(bbox_w),
                int(bbox_h)
            )

            cursor.execute(query, values)

            self.connection.commit()
            self.last_error = None

        except Error as e:

            if self.connection:
                self.connection.rollback()

            self.last_error = str(e)
            print(f"Error inserting detection: {e}")
            raise

        finally:

            if cursor:
                cursor.close()

    # -----------------------------------------------------
    # GET RECENT LOGS
    # -----------------------------------------------------
    def get_recent_logs(self, limit=20):
        """
        Get the latest detection events from MySQL.
        """

        cursor = None

        try:
            self.ensure_connection()

            cursor = self.connection.cursor(dictionary=True)

            # Limit cannot safely be passed like normal SQL values
            limit = max(1, min(int(limit), 1000))

            query = f"""
                SELECT
                    log_id,
                    timestamp,
                    object_class,
                    confidence,
                    bbox_x,
                    bbox_y,
                    bbox_w,
                    bbox_h
                FROM detection_logs
                ORDER BY timestamp DESC
                LIMIT {limit}
            """

            cursor.execute(query)

            rows = cursor.fetchall()
            self.last_error = None
            return rows

        except Error as e:
            self.last_error = str(e)
            print(f"Error retrieving logs: {e}")
            raise

        finally:

            if cursor:
                cursor.close()

    # -----------------------------------------------------
    # GET TOTAL DETECTION COUNT
    # -----------------------------------------------------
    def get_total_detections(self):
        """
        Return total number of stored detection events.
        """

        cursor = None

        try:
            self.ensure_connection()

            cursor = self.connection.cursor()

            cursor.execute(
                "SELECT COUNT(*) FROM detection_logs"
            )

            result = cursor.fetchone()

            self.last_error = None
            return result[0] if result else 0

        except Error as e:
            self.last_error = str(e)
            print(f"Error counting detections: {e}")
            raise

        finally:

            if cursor:
                cursor.close()

    # -----------------------------------------------------
    # CLOSE DATABASE CONNECTION
    # -----------------------------------------------------
    def close(self):
        """
        Close MySQL connection.
        """

        try:

            if self.connection and self.connection.is_connected():

                self.connection.close()

                print("MySQL connection closed.")

        except Error as e:

            print(f"Error closing MySQL connection: {e}")


class DetectionLogWorker:
    """Write detection events off the video-processing thread."""

    def __init__(self, database_factory=DatabaseManager, max_pending=100):
        self._database_factory = database_factory
        self._events = queue.Queue(maxsize=max_pending)
        self._thread = threading.Thread(
            target=self._run,
            name="detection-log-writer",
            daemon=True,
        )
        self._thread.start()

    def submit(self, detection):
        event = {
            "object_class": detection["class"],
            "confidence": detection["confidence"],
            "bbox_x": detection["x"],
            "bbox_y": detection["y"],
            "bbox_w": detection["w"],
            "bbox_h": detection["h"],
        }
        try:
            self._events.put_nowait(event)
        except queue.Full:
            logger.warning("Detection log queue is full; dropping one event.")

    def _run(self):
        database = None
        while True:
            event = self._events.get()
            try:
                if database is None:
                    database = self._database_factory()
                database.insert_detection(**event)
            except Exception:
                logger.exception("Unable to write a detection event to MySQL.")
                if database is not None:
                    database.close()
                    database = None
            finally:
                self._events.task_done()
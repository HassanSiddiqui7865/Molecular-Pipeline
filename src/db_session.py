"""
Database session management for persistent session storage.
Uses a separate database from the ICD codes database.
"""
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Try to import PostgreSQL library
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, Json
    from psycopg2.pool import ThreadedConnectionPool
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False
    logger.warning("psycopg2 not available. Install with: pip install psycopg2-binary")

# Connection pool for the application database
_db_pool = None


def get_app_db_config() -> Dict[str, Any]:
    """
    Get application database configuration from environment variables.
    This is separate from the ICD codes database.
    
    Returns:
        Dictionary with database configuration
    """
    import os
    from config import _ensure_env_loaded
    _ensure_env_loaded()
    
    return {
        'db_host': os.getenv('APP_DB_HOST', 'localhost'),
        'db_port': int(os.getenv('APP_DB_PORT', '5432')),
        'db_name': os.getenv('APP_DB_NAME', 'molecular_pipeline'),
        'db_username': os.getenv('APP_DB_USERNAME', ''),
        'db_password': os.getenv('APP_DB_PASSWORD', ''),
    }


def init_db_pool():
    """Initialize database connection pool."""
    global _db_pool
    if _db_pool is not None:
        return
    
    if not POSTGRESQL_AVAILABLE:
        logger.warning("PostgreSQL not available, session persistence disabled")
        return
    
    try:
        db_config = get_app_db_config()
        
        if not db_config.get('db_name') or not db_config.get('db_username'):
            logger.warning("Application database not configured, session persistence disabled")
            return
        
        _db_pool = ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            host=db_config['db_host'],
            port=db_config['db_port'],
            database=db_config['db_name'],
            user=db_config['db_username'],
            password=db_config['db_password']
        )
        logger.info("Database connection pool initialized")
        
        # Create tables if they don't exist
        _create_tables()
        
    except Exception as e:
        logger.error(f"Failed to initialize database pool: {e}")
        _db_pool = None


def _create_tables():
    """Create database tables if they don't exist."""
    if not _db_pool:
        return
    
    try:
        conn = _db_pool.getconn()
        try:
            cur = conn.cursor()
            
            # Create sessions table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_sessions (
                    session_id VARCHAR(255) PRIMARY KEY,
                    input_parameters JSONB NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    progress INTEGER DEFAULT 0,
                    current_stage VARCHAR(100),
                    error_message TEXT,
                    result JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                )
            """)
            
            # Create index on created_at for faster queries
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_created_at 
                ON pipeline_sessions(created_at DESC)
            """)
            
            # Create index on status
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_status 
                ON pipeline_sessions(status)
            """)
            
            conn.commit()
            logger.info("Database tables created/verified")
            
        finally:
            _db_pool.putconn(conn)
            
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise


@contextmanager
def get_db_connection():
    """Get database connection from pool."""
    if not _db_pool:
        raise RuntimeError("Database pool not initialized")
    
    conn = _db_pool.getconn()
    try:
        yield conn
    finally:
        _db_pool.putconn(conn)


def save_session(
    session_id: str,
    input_parameters: Dict[str, Any],
    status: str = 'running',
    progress: int = 0,
    current_stage: Optional[str] = None,
    error_message: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Save or update session in database.
    
    Args:
        session_id: Unique session identifier
        input_parameters: Input parameters for the pipeline
        status: Session status (running, completed, error)
        progress: Progress percentage (0-100)
        current_stage: Current pipeline stage
        error_message: Error message if any
        result: Final result if completed
        
    Returns:
        True if successful, False otherwise
    """
    if not _db_pool:
        return False
    
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            
            # Check if session exists
            cur.execute(
                "SELECT session_id FROM pipeline_sessions WHERE session_id = %s",
                (session_id,)
            )
            exists = cur.fetchone()
            
            if exists:
                # Update existing session
                cur.execute("""
                    UPDATE pipeline_sessions
                    SET status = %s,
                        progress = %s,
                        current_stage = %s,
                        error_message = %s,
                        result = %s,
                        updated_at = CURRENT_TIMESTAMP,
                        completed_at = CASE WHEN %s IN ('completed', 'error') THEN CURRENT_TIMESTAMP ELSE completed_at END
                    WHERE session_id = %s
                """, (
                    status,
                    progress,
                    current_stage,
                    error_message,
                    Json(result) if result else None,
                    status,
                    session_id
                ))
            else:
                # Insert new session
                cur.execute("""
                    INSERT INTO pipeline_sessions 
                    (session_id, input_parameters, status, progress, current_stage, error_message, result)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    session_id,
                    Json(input_parameters),
                    status,
                    progress,
                    current_stage,
                    error_message,
                    Json(result) if result else None
                ))
            
            conn.commit()
            return True
            
    except Exception as e:
        logger.error(f"Error saving session {session_id}: {e}")
        return False


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve session from database.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Session data dictionary or None if not found
    """
    if not _db_pool:
        return None
    
    try:
        with get_db_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                "SELECT * FROM pipeline_sessions WHERE session_id = %s",
                (session_id,)
            )
            row = cur.fetchone()
            
            if row:
                # Convert to regular dict and handle JSONB fields
                session = dict(row)
                return session
            return None
            
    except Exception as e:
        logger.error(f"Error retrieving session {session_id}: {e}")
        return None


def list_sessions(limit: int = 50, status: Optional[str] = None) -> list:
    """
    List recent sessions.
    
    Args:
        limit: Maximum number of sessions to return
        status: Filter by status (optional)
        
    Returns:
        List of session dictionaries
    """
    if not _db_pool:
        return []
    
    try:
        with get_db_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            if status:
                cur.execute("""
                    SELECT session_id, input_parameters, status, progress, 
                           current_stage, created_at, updated_at, completed_at
                    FROM pipeline_sessions
                    WHERE status = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (status, limit))
            else:
                cur.execute("""
                    SELECT session_id, input_parameters, status, progress, 
                           current_stage, created_at, updated_at, completed_at
                    FROM pipeline_sessions
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (limit,))
            
            rows = cur.fetchall()
            return [dict(row) for row in rows]
            
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        return []


def delete_session(session_id: str) -> bool:
    """
    Delete session from database.
    
    Args:
        session_id: Session identifier
        
    Returns:
        True if successful, False otherwise
    """
    if not _db_pool:
        return False
    
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM pipeline_sessions WHERE session_id = %s",
                (session_id,)
            )
            conn.commit()
            return cur.rowcount > 0
            
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}")
        return False


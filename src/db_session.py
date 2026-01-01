"""
Database session management for persistent session storage.
Uses the same database as the ICD codes database with SSH tunnel support.
"""
import json
import logging
import socket
import threading
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import SSH library
try:
    import paramiko
    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False
    logger.warning("paramiko not available. Install with: pip install paramiko")

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
_ssh_tunnel = None
_ssh_local_port = None


def get_app_db_config() -> Dict[str, Any]:
    """
    Get application database configuration from environment variables.
    Uses the same database configuration as ICD codes database.
    
    Returns:
        Dictionary with database configuration including SSH tunnel settings
    """
    from config import get_database_config
    # Use the same database config as ICD codes
    return get_database_config()


def _handler(src, dst):
    """Forward data from src to dst."""
    try:
        while True:
            data = src.recv(1024)
            if not data:
                break
            dst.send(data)
    except Exception:
        pass
    finally:
        try:
            src.close()
            dst.close()
        except Exception:
            pass


def _forward_tunnel(local_port, remote_host, remote_port, transport):
    """Forward connections from local_port to remote_host:remote_port via transport."""
    class ForwardServer:
        def __init__(self, remote_host, remote_port, transport):
            self.server = None
            self.remote_host = remote_host
            self.remote_port = remote_port
            self.transport = transport
            self.running = False
            
        def handle(self, client, addr):
            try:
                chan = self.transport.open_channel('direct-tcpip', (self.remote_host, self.remote_port), addr)
            except Exception as e:
                logger.warning(f"Error opening channel: {e}")
                try:
                    client.close()
                except Exception:
                    pass
                return
            
            if chan is None:
                logger.warning(f"Channel not opened for {self.remote_host}:{self.remote_port}")
                try:
                    client.close()
                except Exception:
                    pass
                return
            
            # Start forwarding in both directions
            threading.Thread(target=_handler, args=(client, chan), daemon=True).start()
            threading.Thread(target=_handler, args=(chan, client), daemon=True).start()
        
        def start(self, local_port):
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server.bind(('127.0.0.1', local_port))
            self.server.listen(100)
            self.running = True
            
            while self.running:
                try:
                    client, addr = self.server.accept()
                    threading.Thread(target=self.handle, args=(client, addr), daemon=True).start()
                except Exception:
                    break
        
        def stop(self):
            self.running = False
            if self.server:
                try:
                    self.server.close()
                except Exception:
                    pass
    
    return ForwardServer(remote_host, remote_port, transport)


@contextmanager
def get_ssh_tunnel(db_config: Dict[str, Any]):
    """
    Create and manage SSH tunnel to database server using paramiko.
    Context manager version for temporary tunnels (used by ICD transform node).
    
    Args:
        db_config: Database configuration dictionary
        
    Yields:
        Local port number for the tunnel
    """
    if not PARAMIKO_AVAILABLE:
        raise ImportError("paramiko not available")
    
    ssh_client = None
    forward_server = None
    local_port = None
    
    try:
        # Create SSH client
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Load SSH key from project root (check for both "key" and "key.txt")
        pkey = None
        project_root = Path(__file__).parent.parent
        # Try both "key.txt" and "key" (key.txt takes precedence)
        ssh_key_path = None
        for key_name in ['key.txt', 'key']:
            potential_path = project_root / key_name
            if potential_path.exists() and potential_path.is_file():
                ssh_key_path = potential_path
                break
        
        ssh_password = db_config.get('ssh_password', '')
        
        if ssh_key_path:
            logger.info(f"Looking for SSH key at: {ssh_key_path}")
            logger.info(f"Key file exists: {ssh_key_path.exists()}")
        else:
            logger.info(f"SSH key file not found in project root ({project_root}), tried: key, key.txt")
        
        # Try to load key from project root if it exists
        if ssh_key_path and ssh_key_path.exists() and ssh_key_path.is_file():
            logger.info(f"Found SSH key file at {ssh_key_path}, attempting to load...")
            key_errors = []
            for key_class in [paramiko.RSAKey, paramiko.Ed25519Key, paramiko.ECDSAKey]:
                try:
                    try:
                        pkey = key_class.from_private_key_file(str(ssh_key_path))
                        logger.info(f"Successfully loaded SSH key as {key_class.__name__}")
                        break
                    except paramiko.ssh_exception.PasswordRequiredException:
                        if ssh_password:
                            pkey = key_class.from_private_key_file(str(ssh_key_path), password=ssh_password)
                            logger.info(f"Successfully loaded SSH key as {key_class.__name__} with passphrase")
                            break
                        else:
                            raise
                except paramiko.ssh_exception.PasswordRequiredException:
                    key_errors.append(f"{key_class.__name__}: requires passphrase")
                except Exception as e:
                    key_errors.append(f"{key_class.__name__}: {e}")
            
            if not pkey:
                logger.warning(f"Could not load SSH key from {ssh_key_path}. Tried: {', '.join(key_errors)}")
                logger.info("Will attempt SSH connection with password only")
        else:
            logger.info(f"SSH key file not found at {ssh_key_path}, will use password authentication only")
        
        # Connect to SSH server
        logger.info(f"Connecting to SSH server {db_config['ssh_host']}:{db_config['ssh_port']}...")
        auth_method = 'key' if pkey else 'password'
        logger.info(f"Using authentication method: {auth_method}")
        
        try:
            if pkey:
                # Use key-based authentication
                ssh_client.connect(
                    hostname=db_config['ssh_host'],
                    port=db_config['ssh_port'],
                    username=db_config['ssh_username'],
                    pkey=pkey,
                    allow_agent=False,
                    look_for_keys=False
                )
            else:
                # Use password authentication
                if not ssh_password:
                    raise ValueError("SSH password is required when key file is not available")
                ssh_client.connect(
                    hostname=db_config['ssh_host'],
                    port=db_config['ssh_port'],
                    username=db_config['ssh_username'],
                    password=ssh_password,
                    allow_agent=False,
                    look_for_keys=False
                )
        except paramiko.AuthenticationException as e:
            logger.error(f"SSH authentication failed: {e}")
            raise
        except Exception as e:
            logger.error(f"SSH connection error: {e}")
            raise
        
        # Get transport for port forwarding
        transport = ssh_client.get_transport()
        
        # Find an available local port
        temp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        temp_socket.bind(('127.0.0.1', 0))
        local_port = temp_socket.getsockname()[1]
        temp_socket.close()
        
        # Start port forwarding
        forward_server = _forward_tunnel(
            local_port,
            db_config['db_host'],
            db_config['db_port'],
            transport
        )
        
        # Start forwarding server in a thread
        forward_thread = threading.Thread(target=forward_server.start, args=(local_port,))
        forward_thread.daemon = True
        forward_thread.start()
        
        logger.info(f"SSH tunnel established. Local port: {local_port}")
        
        yield local_port
        
    finally:
        # Clean up
        if forward_server:
            try:
                forward_server.stop()
            except Exception as e:
                logger.warning(f"Error stopping forward server: {e}")
        
        if ssh_client:
            try:
                ssh_client.close()
                logger.info("SSH tunnel closed")
            except Exception as e:
                logger.warning(f"Error closing SSH connection: {e}")


def _start_ssh_tunnel(db_config: Dict[str, Any]) -> int:
    """
    Start persistent SSH tunnel to database server using paramiko.
    
    Args:
        db_config: Database configuration dictionary
        
    Returns:
        Local port number for the tunnel
    """
    global _ssh_tunnel, _ssh_local_port
    
    if not PARAMIKO_AVAILABLE:
        raise ImportError("paramiko not available")
    
    if not db_config.get('ssh_host'):
        # No SSH tunnel needed, return None to indicate direct connection
        return None
    
    ssh_client = None
    forward_server = None
    local_port = None
    
    try:
        # Create SSH client
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Load SSH key from project root (check for both "key" and "key.txt")
        pkey = None
        project_root = Path(__file__).parent.parent
        # Try both "key.txt" and "key" (key.txt takes precedence)
        ssh_key_path = None
        for key_name in ['key.txt', 'key']:
            potential_path = project_root / key_name
            if potential_path.exists() and potential_path.is_file():
                ssh_key_path = potential_path
                break
        
        ssh_password = db_config.get('ssh_password', '')
        
        if ssh_key_path:
            logger.info(f"Looking for SSH key at: {ssh_key_path}")
            logger.info(f"Key file exists: {ssh_key_path.exists()}")
        else:
            logger.info(f"SSH key file not found in project root ({project_root}), tried: key, key.txt")
        
        # Try to load key from project root if it exists
        if ssh_key_path and ssh_key_path.exists() and ssh_key_path.is_file():
            logger.info(f"Found SSH key file at {ssh_key_path}, attempting to load...")
            key_errors = []
            for key_class in [paramiko.RSAKey, paramiko.Ed25519Key, paramiko.ECDSAKey]:
                try:
                    try:
                        pkey = key_class.from_private_key_file(str(ssh_key_path))
                        logger.info(f"Successfully loaded SSH key as {key_class.__name__}")
                        break
                    except paramiko.ssh_exception.PasswordRequiredException:
                        if ssh_password:
                            pkey = key_class.from_private_key_file(str(ssh_key_path), password=ssh_password)
                            logger.info(f"Successfully loaded SSH key as {key_class.__name__} with passphrase")
                            break
                        else:
                            raise
                except paramiko.ssh_exception.PasswordRequiredException:
                    key_errors.append(f"{key_class.__name__}: requires passphrase")
                except Exception as e:
                    key_errors.append(f"{key_class.__name__}: {e}")
            
            if not pkey:
                logger.warning(f"Could not load SSH key from {ssh_key_path}. Tried: {', '.join(key_errors)}")
                logger.info("Will attempt SSH connection with password only")
        else:
            logger.info(f"SSH key file not found at {ssh_key_path}, will use password authentication only")
        
        # Connect to SSH server
        logger.info(f"Connecting to SSH server {db_config['ssh_host']}:{db_config['ssh_port']}...")
        auth_method = 'key' if pkey else 'password'
        logger.info(f"Using authentication method: {auth_method}")
        
        try:
            if pkey:
                # Use key-based authentication
                ssh_client.connect(
                    hostname=db_config['ssh_host'],
                    port=db_config['ssh_port'],
                    username=db_config['ssh_username'],
                    pkey=pkey,
                    allow_agent=False,
                    look_for_keys=False
                )
            else:
                # Use password authentication
                if not ssh_password:
                    raise ValueError("SSH password is required when key file is not available")
                ssh_client.connect(
                    hostname=db_config['ssh_host'],
                    port=db_config['ssh_port'],
                    username=db_config['ssh_username'],
                    password=ssh_password,
                    allow_agent=False,
                    look_for_keys=False
                )
        except paramiko.AuthenticationException as e:
            logger.error(f"SSH authentication failed: {e}")
            raise
        except Exception as e:
            logger.error(f"SSH connection error: {e}")
            raise
        
        # Get transport for port forwarding
        transport = ssh_client.get_transport()
        
        # Find an available local port
        temp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        temp_socket.bind(('127.0.0.1', 0))
        local_port = temp_socket.getsockname()[1]
        temp_socket.close()
        
        # Start port forwarding
        forward_server = _forward_tunnel(
            local_port,
            db_config['db_host'],
            db_config['db_port'],
            transport
        )
        
        # Start forwarding server in a thread
        forward_thread = threading.Thread(target=forward_server.start, args=(local_port,))
        forward_thread.daemon = True
        forward_thread.start()
        
        logger.info(f"SSH tunnel established. Local port: {local_port}")
        
        # Store references to keep tunnel alive
        _ssh_tunnel = {
            'ssh_client': ssh_client,
            'forward_server': forward_server,
            'forward_thread': forward_thread
        }
        _ssh_local_port = local_port
        
        return local_port
        
    except Exception as e:
        # Clean up on error
        if forward_server:
            try:
                forward_server.stop()
            except Exception:
                pass
        if ssh_client:
            try:
                ssh_client.close()
            except Exception:
                pass
        raise


def _stop_ssh_tunnel():
    """Stop the persistent SSH tunnel."""
    global _ssh_tunnel, _ssh_local_port
    
    if _ssh_tunnel:
        try:
            if _ssh_tunnel.get('forward_server'):
                _ssh_tunnel['forward_server'].stop()
        except Exception as e:
            logger.warning(f"Error stopping forward server: {e}")
        
        try:
            if _ssh_tunnel.get('ssh_client'):
                _ssh_tunnel['ssh_client'].close()
                logger.info("SSH tunnel closed")
        except Exception as e:
            logger.warning(f"Error closing SSH connection: {e}")
        
        _ssh_tunnel = None
        _ssh_local_port = None


def init_db_pool():
    """Initialize database connection pool with SSH tunnel support if needed."""
    global _db_pool, _ssh_local_port
    
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
        
        # Start SSH tunnel if SSH host is configured
        connect_host = '127.0.0.1'
        connect_port = db_config['db_port']
        
        if db_config.get('ssh_host'):
            try:
                local_port = _start_ssh_tunnel(db_config)
                if local_port:
                    connect_host = '127.0.0.1'
                    connect_port = local_port
                    logger.info(f"Using SSH tunnel for database connection (local port: {local_port})")
                else:
                    logger.warning("SSH tunnel configuration provided but tunnel failed to start, trying direct connection")
            except Exception as e:
                logger.error(f"Failed to start SSH tunnel: {e}. Trying direct connection...")
                # Fall back to direct connection if SSH tunnel fails
                connect_host = db_config['db_host']
                connect_port = db_config['db_port']
        else:
            # Direct connection (no SSH tunnel)
            connect_host = db_config['db_host']
            connect_port = db_config['db_port']
        
        _db_pool = ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            host=connect_host,
            port=connect_port,
            database=db_config['db_name'],
            user=db_config['db_username'],
            password=db_config['db_password']
        )
        logger.info(f"Database connection pool initialized (host: {connect_host}, port: {connect_port})")
        
        # Create tables if they don't exist
        _create_tables()
        
    except Exception as e:
        logger.error(f"Failed to initialize database pool: {e}")
        _db_pool = None
        # Clean up SSH tunnel if it was started
        _stop_ssh_tunnel()


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
                CREATE TABLE IF NOT EXISTS dev.pipeline_sessions (
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
                ON dev.pipeline_sessions(created_at DESC)
            """)
            
            # Create index on status
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_status 
                ON dev.pipeline_sessions(status)
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
                "SELECT session_id FROM dev.pipeline_sessions WHERE session_id = %s",
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
                    INSERT INTO dev.pipeline_sessions 
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
                    FROM dev.pipeline_sessions
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
                "DELETE FROM dev.pipeline_sessions WHERE session_id = %s",
                (session_id,)
            )
            conn.commit()
            return cur.rowcount > 0
            
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}")
        return False


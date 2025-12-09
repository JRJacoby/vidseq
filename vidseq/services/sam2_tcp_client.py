"""TCP client for communicating with SAM2 worker server."""

import json
import socket
import struct
from typing import Optional


class SAM2TCPClient:
    """TCP client for SAM2 worker communication."""
    
    def __init__(self):
        self.socket: Optional[socket.socket] = None
        self.host: Optional[str] = None
        self.port: Optional[int] = None
    
    def connect(self, host: str, port: int, timeout: float = 10.0) -> None:
        """
        Connect to SAM2 worker server.
        
        Args:
            host: Server hostname (typically 'localhost')
            port: Server port number
            timeout: Connection timeout in seconds
            
        Raises:
            ConnectionError: If connection fails
        """
        if self.socket is not None:
            self.disconnect()
        
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(timeout)
            self.socket.connect((host, port))
            self.host = host
            self.port = port
        except Exception as e:
            self.socket = None
            raise ConnectionError(f"Failed to connect to SAM2 worker at {host}:{port}: {e}")
    
    def disconnect(self) -> None:
        """Close connection to server."""
        if self.socket is not None:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None
            self.host = None
            self.port = None
    
    def send_command(self, cmd: dict, timeout: float = 120.0) -> dict:
        """
        Send command to server and wait for response.
        
        Args:
            cmd: Command dictionary (will be JSON serialized)
            timeout: Response timeout in seconds
            
        Returns:
            Response dictionary from server
            
        Raises:
            ConnectionError: If not connected or connection lost
            TimeoutError: If response timeout exceeded
        """
        if self.socket is None:
            raise ConnectionError("Not connected to SAM2 worker")
        
        # Set socket timeout
        old_timeout = self.socket.gettimeout()
        self.socket.settimeout(timeout)
        
        try:
            # Serialize command to JSON
            cmd_json = json.dumps(cmd)
            cmd_bytes = cmd_json.encode('utf-8')
            
            # Send length prefix (4 bytes, big-endian)
            length_prefix = struct.pack('>I', len(cmd_bytes))
            self.socket.sendall(length_prefix)
            
            # Send command data
            self.socket.sendall(cmd_bytes)
            
            # Receive response length
            length_bytes = self._recv_exact(4)
            if len(length_bytes) != 4:
                raise ConnectionError("Connection closed by server")
            
            response_length = struct.unpack('>I', length_bytes)[0]
            
            # Receive response data
            response_bytes = self._recv_exact(response_length)
            response_json = response_bytes.decode('utf-8')
            response = json.loads(response_json)
            
            return response
        except socket.timeout:
            raise TimeoutError(f"Timeout waiting for response to {cmd.get('type', 'unknown')}")
        except Exception as e:
            if isinstance(e, (ConnectionError, TimeoutError)):
                raise
            raise ConnectionError(f"Error communicating with SAM2 worker: {e}")
        finally:
            self.socket.settimeout(old_timeout)
    
    def _recv_exact(self, n: int) -> bytes:
        """Receive exactly n bytes from socket."""
        data = b''
        while len(data) < n:
            chunk = self.socket.recv(n - len(data))
            if not chunk:
                raise ConnectionError("Connection closed by server")
            data += chunk
        return data
    
    def is_connected(self) -> bool:
        """Check if client is connected."""
        # Just check if socket exists - if it's actually closed, we'll get an error
        # when we try to use it, and can reconnect then
        return self.socket is not None


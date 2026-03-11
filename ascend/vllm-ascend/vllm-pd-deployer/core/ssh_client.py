"""
SSH client module for remote command execution and file transfer.
"""

import paramiko
import os
import socket
from typing import Tuple, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SSHError(Exception):
    """SSH operation error."""
    pass


class SSHClient:
    """SSH client for remote operations."""
    
    def __init__(
        self,
        host: str,
        user: str,
        port: int = 22,
        key_file: Optional[str] = None,
        password: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize SSH client.
        
        Args:
            host: Remote host address
            user: Username for authentication
            port: SSH port (default: 22)
            key_file: Path to private key file
            password: Password for authentication
            timeout: Connection timeout in seconds
        """
        self.host = host
        self.user = user
        self.port = port
        self.key_file = key_file
        self.password = password
        self.timeout = timeout
        
        self.client: Optional[paramiko.SSHClient] = None
        self.sftp: Optional[paramiko.SFTPClient] = None
    
    def connect(self) -> None:
        """Establish SSH connection."""
        try:
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            connect_kwargs = {
                'hostname': self.host,
                'port': self.port,
                'username': self.user,
                'timeout': self.timeout,
                'allow_agent': True,
                'look_for_keys': True
            }
            
            # Use key file if provided
            if self.key_file:
                key_file_path = os.path.expanduser(self.key_file)
                if os.path.exists(key_file_path):
                    connect_kwargs['key_filename'] = key_file_path
                else:
                    logger.warning(f"Key file not found: {key_file_path}")
            
            # Use password if provided and no key file
            if self.password and not self.key_file:
                connect_kwargs['password'] = self.password
            
            self.client.connect(**connect_kwargs)
            logger.info(f"Connected to {self.host}:{self.port} as {self.user}")
            
        except paramiko.AuthenticationException as e:
            raise SSHError(f"Authentication failed for {self.host}: {e}")
        except paramiko.SSHException as e:
            raise SSHError(f"SSH connection failed to {self.host}: {e}")
        except socket.timeout:
            raise SSHError(f"Connection timeout to {self.host}:{self.port}")
        except Exception as e:
            raise SSHError(f"Failed to connect to {self.host}: {e}")
    
    def disconnect(self) -> None:
        """Close SSH connection."""
        if self.sftp:
            try:
                self.sftp.close()
            except Exception:
                pass
            self.sftp = None
        
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None
            logger.info(f"Disconnected from {self.host}")
    
    def ensure_connected(self) -> None:
        """Ensure SSH connection is established."""
        if not self.client or self.client.get_transport() is None:
            self.connect()
    
    def execute(
        self,
        command: str,
        timeout: Optional[int] = None,
        get_pty: bool = False
    ) -> Tuple[int, str, str]:
        """
        Execute command on remote host.
        
        Args:
            command: Command to execute
            timeout: Command timeout in seconds
            get_pty: Whether to allocate a pseudo-terminal
            
        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        self.ensure_connected()
        
        try:
            logger.debug(f"Executing on {self.host}: {command[:100]}...")
            
            stdin, stdout, stderr = self.client.exec_command(
                command,
                timeout=timeout,
                get_pty=get_pty
            )
            
            exit_code = stdout.channel.recv_exit_status()
            stdout_data = stdout.read().decode('utf-8', errors='replace')
            stderr_data = stderr.read().decode('utf-8', errors='replace')
            
            return exit_code, stdout_data, stderr_data
            
        except socket.timeout:
            raise SSHError(f"Command timeout on {self.host}")
        except Exception as e:
            raise SSHError(f"Command execution failed on {self.host}: {e}")
    
    def execute_with_retry(
        self,
        command: str,
        timeout: Optional[int] = None,
        max_retries: int = 3,
        retry_delay: int = 5
    ) -> Tuple[int, str, str]:
        """
        Execute command with retry logic.
        
        Args:
            command: Command to execute
            timeout: Command timeout
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
            
        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        import time
        
        last_error = None
        for attempt in range(max_retries):
            try:
                return self.execute(command, timeout)
            except SSHError as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    # Reconnect on failure
                    self.disconnect()
                    self.connect()
        
        raise SSHError(f"All {max_retries} attempts failed: {last_error}")
    
    def get_sftp(self) -> paramiko.SFTPClient:
        """Get or create SFTP client."""
        self.ensure_connected()
        
        if not self.sftp or self.sftp.sock.closed:
            self.sftp = self.client.open_sftp()
        
        return self.sftp
    
    def upload_file(self, local_path: str, remote_path: str) -> None:
        """
        Upload file to remote host.
        
        Args:
            local_path: Path to local file
            remote_path: Destination path on remote host
        """
        sftp = self.get_sftp()
        
        try:
            # Create remote directory if needed
            remote_dir = os.path.dirname(remote_path)
            if remote_dir:
                self._mkdir_p(remote_dir)
            
            sftp.put(local_path, remote_path)
            logger.debug(f"Uploaded {local_path} -> {self.host}:{remote_path}")
            
        except Exception as e:
            raise SSHError(f"Failed to upload {local_path} to {self.host}: {e}")
    
    def upload_directory(
        self,
        local_dir: str,
        remote_dir: str,
        exclude: Optional[List[str]] = None
    ) -> None:
        """
        Upload entire directory to remote host.
        
        Args:
            local_dir: Path to local directory
            remote_dir: Destination path on remote host
            exclude: List of patterns to exclude
        """
        sftp = self.get_sftp()
        exclude = exclude or ['.git', '__pycache__', '*.pyc', '.DS_Store']
        
        local_path = Path(local_dir)
        
        for item in local_path.rglob('*'):
            # Check exclusions
            if any(ex in str(item) for ex in exclude):
                continue
            
            if item.is_file():
                relative_path = item.relative_to(local_path)
                remote_file = os.path.join(remote_dir, str(relative_path))
                
                # Create remote directory
                remote_file_dir = os.path.dirname(remote_file)
                self._mkdir_p(remote_file_dir)
                
                # Upload file
                sftp.put(str(item), remote_file)
        
        logger.info(f"Uploaded directory {local_dir} -> {self.host}:{remote_dir}")
    
    def download_file(self, remote_path: str, local_path: str) -> None:
        """
        Download file from remote host.
        
        Args:
            remote_path: Path to remote file
            local_path: Destination path on local machine
        """
        sftp = self.get_sftp()
        
        try:
            # Create local directory if needed
            local_dir = os.path.dirname(local_path)
            if local_dir:
                os.makedirs(local_dir, exist_ok=True)
            
            sftp.get(remote_path, local_path)
            logger.debug(f"Downloaded {self.host}:{remote_path} -> {local_path}")
            
        except Exception as e:
            raise SSHError(f"Failed to download from {self.host}: {e}")
    
    def _mkdir_p(self, remote_directory: str) -> None:
        """Create remote directory recursively (mkdir -p)."""
        sftp = self.get_sftp()
        
        if remote_directory in ['/', '']:
            return
        
        try:
            sftp.stat(remote_directory)
        except FileNotFoundError:
            # Directory doesn't exist, create parent first
            parent = os.path.dirname(remote_directory)
            if parent and parent != remote_directory:
                self._mkdir_p(parent)
            
            try:
                sftp.mkdir(remote_directory)
            except IOError:
                # Directory might have been created by another process
                pass
    
    def check_file_exists(self, remote_path: str) -> bool:
        """Check if file exists on remote host."""
        try:
            sftp = self.get_sftp()
            sftp.stat(remote_path)
            return True
        except FileNotFoundError:
            return False
    
    def list_directory(self, remote_path: str) -> List[str]:
        """List directory contents on remote host."""
        try:
            sftp = self.get_sftp()
            return sftp.listdir(remote_path)
        except Exception as e:
            raise SSHError(f"Failed to list directory {remote_path} on {self.host}: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False

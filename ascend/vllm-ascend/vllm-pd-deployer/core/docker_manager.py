"""
Docker container management module for executing commands inside containers.
"""

import logging
from typing import Tuple, Optional
from .ssh_client import SSHClient, SSHError

logger = logging.getLogger(__name__)


class DockerManager:
    """Manager for Docker container operations."""
    
    def __init__(self, ssh_client: SSHClient, container_name: str):
        """
        Initialize Docker manager.
        
        Args:
            ssh_client: Connected SSH client
            container_name: Name of the Docker container
        """
        self.ssh = ssh_client
        self.container_name = container_name
    
    def is_container_running(self) -> bool:
        """
        Check if container is running.
        
        Returns:
            True if container is running, False otherwise
        """
        try:
            format_str = "{{{{.Names}}}}"
            cmd = f"docker ps --filter 'name={self.container_name}' --filter 'status=running' --format '{format_str}'"
            exit_code, stdout, stderr = self.ssh.execute(cmd)
            
            if exit_code == 0:
                containers = stdout.strip().split('\n')
                return self.container_name in containers
            return False
        except SSHError:
            return False
    
    def is_container_exists(self) -> bool:
        """
        Check if container exists (running or stopped).
        
        Returns:
            True if container exists, False otherwise
        """
        try:
            format_str = "{{{{.Names}}}}"
            cmd = f"docker ps -a --filter 'name={self.container_name}' --format '{format_str}'"
            exit_code, stdout, stderr = self.ssh.execute(cmd)
            
            if exit_code == 0:
                containers = stdout.strip().split('\n')
                return self.container_name in containers
            return False
        except SSHError:
            return False
    
    def start_container(
        self,
        image: str,
        shm_size: str = "80g",
        network: str = "host",
        privileged: bool = True,
        volumes: Optional[list] = None,
        environment: Optional[dict] = None
    ) -> None:
        """
        Start a Docker container.
        
        Args:
            image: Docker image name
            shm_size: Shared memory size
            network: Network mode
            privileged: Run in privileged mode
            volumes: List of volume mounts (host:container)
            environment: Environment variables dict
        """
        if self.is_container_running():
            logger.info(f"Container {self.container_name} is already running")
            return
        
        # Remove existing container if stopped
        if self.is_container_exists():
            logger.info(f"Removing existing container {self.container_name}")
            cmd = f"docker rm -f {self.container_name}"
            self.ssh.execute(cmd)
        
        # Build docker run command
        cmd_parts = ["docker run"]
        cmd_parts.append(f"--name {self.container_name}")
        cmd_parts.append(f"--shm-size={shm_size}")
        cmd_parts.append(f"--net={network}")
        
        if privileged:
            cmd_parts.append("--privileged")
        
        # Add volumes
        default_volumes = [
            "/usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool",
            "/usr/local/bin/npu-smi:/usr/local/bin/npu-smi",
            "/usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/",
            "/usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info",
            "/etc/ascend_install.info:/etc/ascend_install.info",
            "/root:/root",
            "/mnt/sfs_turbo:/mnt/sfs_turbo"
        ]
        
        volumes = volumes or default_volumes
        for vol in volumes:
            cmd_parts.append(f"-v {vol}")
        
        # Add environment variables
        if environment:
            for key, value in environment.items():
                cmd_parts.append(f"-e {key}={value}")
        
        # Add image and command
        cmd_parts.append(f"-itd {image}")
        cmd_parts.append("bash")
        
        cmd = " ".join(cmd_parts)
        logger.info(f"Starting container {self.container_name}")
        
        exit_code, stdout, stderr = self.ssh.execute(cmd)
        if exit_code != 0:
            raise SSHError(f"Failed to start container: {stderr}")
        
        logger.info(f"Container {self.container_name} started successfully")
    
    def stop_container(self) -> None:
        """Stop the Docker container."""
        if not self.is_container_running():
            logger.info(f"Container {self.container_name} is not running")
            return
        
        cmd = f"docker stop {self.container_name}"
        logger.info(f"Stopping container {self.container_name}")
        
        exit_code, stdout, stderr = self.ssh.execute(cmd)
        if exit_code != 0:
            raise SSHError(f"Failed to stop container: {stderr}")
        
        logger.info(f"Container {self.container_name} stopped")
    
    def exec_in_container(
        self,
        command: str,
        workdir: Optional[str] = None,
        timeout: Optional[int] = None,
        detach: bool = False
    ) -> Tuple[int, str, str]:
        """
        Execute command inside the container.
        
        Args:
            command: Command to execute
            workdir: Working directory inside container
            timeout: Command timeout
            detach: Run in detached mode
            
        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        if not self.is_container_running():
            raise SSHError(f"Container {self.container_name} is not running")
        
        # Build docker exec command
        cmd_parts = ["docker exec"]
        
        if detach:
            cmd_parts.append("-d")
        else:
            cmd_parts.append("-i")
        
        if workdir:
            cmd_parts.append(f"-w {workdir}")
        
        cmd_parts.append(self.container_name)
        
        # Use bash -c to execute complex commands
        cmd_parts.append(f"bash -c '{command}'")
        
        cmd = " ".join(cmd_parts)
        
        return self.ssh.execute(cmd, timeout=timeout)
    
    def copy_to_container(self, local_path: str, container_path: str) -> None:
        """
        Copy file to container.
        
        Args:
            local_path: Path on remote host
            container_path: Destination path in container
        """
        if not self.is_container_running():
            raise SSHError(f"Container {self.container_name} is not running")
        
        cmd = f"docker cp {local_path} {self.container_name}:{container_path}"
        exit_code, stdout, stderr = self.ssh.execute(cmd)
        
        if exit_code != 0:
            raise SSHError(f"Failed to copy to container: {stderr}")
    
    def get_logs(self, tail: int = 100) -> str:
        """
        Get container logs.
        
        Args:
            tail: Number of lines to retrieve
            
        Returns:
            Container logs
        """
        cmd = f"docker logs --tail {tail} {self.container_name}"
        exit_code, stdout, stderr = self.ssh.execute(cmd)
        
        if exit_code != 0:
            return f"Failed to get logs: {stderr}"
        
        return stdout
    
    def execute_script_in_container(
        self,
        script_path: str,
        workdir: Optional[str] = None,
        detach: bool = True
    ) -> Tuple[int, str, str]:
        """
        Execute a script file inside the container.
        
        Args:
            script_path: Path to script in container
            workdir: Working directory
            detach: Run in background
            
        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        cmd = f"bash {script_path}"
        return self.exec_in_container(cmd, workdir=workdir, detach=detach)

"""
Deployment orchestrator for vLLM-PD deployment.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from .config import Config, NodeConfig, ProxyConfig
from .generator import ScriptGenerator
from .ssh_client import SSHClient, SSHError
from .docker_manager import DockerManager

logger = logging.getLogger(__name__)


class DeploymentError(Exception):
    """Deployment error."""
    pass


class NodeStatus:
    """Status information for a node."""
    
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.status = "unknown"  # unknown, connecting, deploying, running, error, stopped
        self.health = "unknown"  # unknown, healthy, unhealthy
        self.message = ""
        self.last_update = datetime.now()
        self.uptime = ""
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'role': self.role,
            'status': self.status,
            'health': self.health,
            'message': self.message,
            'last_update': self.last_update.isoformat(),
            'uptime': self.uptime
        }


class Deployer:
    """Orchestrates vLLM-PD deployment across multiple nodes."""
    
    def __init__(self, config: Config, output_dir: str = "./generated"):
        """
        Initialize deployer.
        
        Args:
            config: Deployment configuration
            output_dir: Directory for generated scripts
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.generator = ScriptGenerator(
            os.path.join(os.path.dirname(__file__), '..', 'templates')
        )
        
        # Track node statuses
        self.nodes_status: Dict[str, NodeStatus] = {}
        self._init_status()
    
    def _init_status(self):
        """Initialize status tracking for all nodes."""
        for node in self.config.get_prefill_nodes():
            self.nodes_status[node.name] = NodeStatus(node.name, 'prefill')
        
        for node in self.config.get_decode_nodes():
            self.nodes_status[node.name] = NodeStatus(node.name, 'decode')
        
        if self.config.get_proxy_config().enabled:
            self.nodes_status['proxy'] = NodeStatus('proxy', 'proxy')
    
    def generate_scripts(self) -> Dict[str, dict]:
        """
        Generate all deployment scripts.
        
        Returns:
            Dictionary mapping node names to generated file paths
        """
        logger.info("Generating deployment scripts...")
        generated = self.generator.generate_all(self.config, str(self.output_dir))
        logger.info(f"Scripts generated in {self.output_dir}")
        return generated
    
    def deploy(self, target: str = "all", dry_run: bool = False) -> bool:
        """
        Execute deployment.
        
        Args:
            target: Deployment target ('all', 'prefill', 'decode', 'proxy')
            dry_run: If True, only generate scripts without deploying
            
        Returns:
            True if deployment successful, False otherwise
        """
        try:
            # Step 1: Generate scripts
            generated = self.generate_scripts()
            
            if dry_run:
                logger.info("Dry run mode - scripts generated but not deployed")
                return True
            
            # Step 2: Deploy according to target
            success = True
            
            if target in ['all', 'prefill']:
                if not self._deploy_prefill_nodes():
                    success = False
                    logger.error("Prefill deployment failed")
            
            if target in ['all', 'decode']:
                if not self._deploy_decode_nodes():
                    success = False
                    logger.error("Decode deployment failed")
            
            if target in ['all', 'proxy']:
                if not self._deploy_proxy():
                    success = False
                    logger.error("Proxy deployment failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False
    
    def _deploy_prefill_nodes(self) -> bool:
        """Deploy all prefill nodes in parallel."""
        logger.info("Deploying prefill nodes...")
        nodes = self.config.get_prefill_nodes()
        return self._deploy_nodes_parallel(nodes, 'prefill')
    
    def _deploy_decode_nodes(self) -> bool:
        """Deploy all decode nodes in parallel."""
        logger.info("Deploying decode nodes...")
        nodes = self.config.get_decode_nodes()
        return self._deploy_nodes_parallel(nodes, 'decode')
    
    def _deploy_nodes_parallel(self, nodes: List[NodeConfig], role: str) -> bool:
        """
        Deploy multiple nodes in parallel.
        
        Args:
            nodes: List of node configurations
            role: 'prefill' or 'decode'
            
        Returns:
            True if all nodes deployed successfully
        """
        success_count = 0
        
        with ThreadPoolExecutor(max_workers=min(len(nodes), 5)) as executor:
            futures = {
                executor.submit(self._deploy_single_node, node, role): node
                for node in nodes
            }
            
            for future in as_completed(futures):
                node = futures[future]
                try:
                    if future.result():
                        success_count += 1
                        logger.info(f"Successfully deployed {node.name}")
                    else:
                        logger.error(f"Failed to deploy {node.name}")
                except Exception as e:
                    logger.error(f"Exception deploying {node.name}: {e}")
        
        return success_count == len(nodes)
    
    def _deploy_single_node(self, node: NodeConfig, role: str) -> bool:
        """
        Deploy a single node.
        
        Args:
            node: Node configuration
            role: 'prefill' or 'decode'
            
        Returns:
            True if deployment successful
        """
        status = self.nodes_status[node.name]
        status.status = "connecting"
        
        ssh = None
        try:
            # 1. Connect via SSH
            ssh = SSHClient(
                host=node.host,
                user=node.ssh_config.user,
                port=node.ssh_config.port,
                key_file=node.ssh_config.key_file,
                password=node.ssh_config.password
            )
            ssh.connect()
            
            # 2. Upload scripts
            status.status = "deploying"
            local_node_dir = self.output_dir / role / node.name
            remote_base = "/mnt/sfs_turbo/glm5_PD_deploy"
            remote_node_dir = f"{remote_base}/{role}/{node.name}"
            
            ssh.execute(f"mkdir -p {remote_node_dir}")
            
            for script_file in ['server.sh', 'run_dp_template.sh', 'launch_online_dp.py']:
                local_file = local_node_dir / script_file
                remote_file = f"{remote_node_dir}/{script_file}"
                ssh.upload_file(str(local_file), remote_file)
            
            # Make scripts executable
            ssh.execute(f"chmod +x {remote_node_dir}/*.sh")
            
            # 3. Start Docker container if needed
            if node.docker_config.enabled:
                docker = DockerManager(ssh, node.docker_config.container_name)
                
                if not docker.is_container_running():
                    docker.start_container(
                        image=node.docker_config.image,
                        shm_size="80g",
                        network="host",
                        privileged=True
                    )
                
                # Copy scripts to container
                for script_file in ['server.sh', 'run_dp_template.sh', 'launch_online_dp.py']:
                    remote_file = f"{remote_node_dir}/{script_file}"
                    docker.copy_to_container(remote_file, node.docker_config.workdir)
                
                # Execute server.sh in container
                workdir = node.docker_config.workdir
                docker.execute_script_in_container(
                    f"{workdir}/server.sh",
                    workdir=workdir,
                    detach=True
                )
            else:
                # Execute directly on host
                ssh.execute(f"cd {remote_node_dir} && nohup bash server.sh > /dev/null 2>&1 &")
            
            status.status = "running"
            status.message = "Deployed successfully"
            return True
            
        except Exception as e:
            status.status = "error"
            status.message = str(e)
            logger.error(f"Failed to deploy {node.name}: {e}")
            return False
        finally:
            if ssh:
                ssh.disconnect()
            status.last_update = datetime.now()
    
    def _deploy_proxy(self) -> bool:
        """Deploy proxy/load balancer."""
        logger.info("Deploying proxy...")
        
        proxy_config = self.config.get_proxy_config()
        if not proxy_config.enabled:
            logger.info("Proxy is disabled")
            return True
        
        status = self.nodes_status['proxy']
        status.status = "connecting"
        
        ssh = None
        try:
            ssh = SSHClient(
                host=proxy_config.host,
                user=proxy_config.ssh_config.user,
                port=proxy_config.ssh_config.port,
                key_file=proxy_config.ssh_config.key_file,
                password=proxy_config.ssh_config.password
            )
            ssh.connect()
            
            status.status = "deploying"
            
            # Upload proxy script
            local_proxy_dir = self.output_dir / 'proxy'
            remote_base = "/mnt/sfs_turbo/glm5_PD_deploy"
            remote_proxy_dir = f"{remote_base}/proxy"
            
            ssh.execute(f"mkdir -p {remote_proxy_dir}")
            
            proxy_sh = local_proxy_dir / 'proxy.sh'
            ssh.upload_file(str(proxy_sh), f"{remote_proxy_dir}/proxy.sh")
            ssh.execute(f"chmod +x {remote_proxy_dir}/proxy.sh")
            
            # Execute proxy
            if proxy_config.docker_config and proxy_config.docker_config.enabled:
                docker = DockerManager(ssh, proxy_config.docker_config.container_name)
                
                if not docker.is_container_running():
                    docker.start_container(
                        image=proxy_config.docker_config.image,
                        shm_size="8g",
                        network="host",
                        privileged=False
                    )
                
                docker.copy_to_container(
                    f"{remote_proxy_dir}/proxy.sh",
                    proxy_config.docker_config.workdir
                )
                
                workdir = proxy_config.docker_config.workdir
                docker.execute_script_in_container(
                    f"{workdir}/proxy.sh",
                    workdir=workdir,
                    detach=True
                )
            else:
                ssh.execute(f"cd {remote_proxy_dir} && nohup bash proxy.sh > /dev/null 2>&1 &")
            
            status.status = "running"
            status.message = "Deployed successfully"
            return True
            
        except Exception as e:
            status.status = "error"
            status.message = str(e)
            logger.error(f"Failed to deploy proxy: {e}")
            return False
        finally:
            if ssh:
                ssh.disconnect()
            status.last_update = datetime.now()
    
    def stop(self, target: str = "all") -> bool:
        """
        Stop deployed services.
        
        Args:
            target: Target to stop ('all', 'prefill', 'decode', 'proxy')
            
        Returns:
            True if successful
        """
        success = True
        
        if target in ['all', 'proxy']:
            if not self._stop_proxy():
                success = False
        
        if target in ['all', 'decode']:
            if not self._stop_nodes(self.config.get_decode_nodes(), 'decode'):
                success = False
        
        if target in ['all', 'prefill']:
            if not self._stop_nodes(self.config.get_prefill_nodes(), 'prefill'):
                success = False
        
        return success
    
    def _stop_nodes(self, nodes: List[NodeConfig], role: str) -> bool:
        """Stop nodes of specified role."""
        success_count = 0
        
        for node in nodes:
            try:
                ssh = SSHClient(
                    host=node.host,
                    user=node.ssh_config.user,
                    port=node.ssh_config.port,
                    key_file=node.ssh_config.key_file,
                    password=node.ssh_config.password
                )
                ssh.connect()
                
                if node.docker_config.enabled:
                    docker = DockerManager(ssh, node.docker_config.container_name)
                    # Find and kill vLLM processes in container
                    docker.exec_in_container("pkill -f 'vllm serve' || true")
                else:
                    ssh.execute("pkill -f 'vllm serve' || true")
                
                self.nodes_status[node.name].status = "stopped"
                self.nodes_status[node.name].message = "Stopped"
                success_count += 1
                
                ssh.disconnect()
                
            except Exception as e:
                logger.error(f"Failed to stop {node.name}: {e}")
        
        return success_count == len(nodes)
    
    def _stop_proxy(self) -> bool:
        """Stop proxy service."""
        proxy_config = self.config.get_proxy_config()
        if not proxy_config.enabled:
            return True
        
        try:
            ssh = SSHClient(
                host=proxy_config.host,
                user=proxy_config.ssh_config.user,
                port=proxy_config.ssh_config.port,
                key_file=proxy_config.ssh_config.key_file,
                password=proxy_config.ssh_config.password
            )
            ssh.connect()
            
            if proxy_config.docker_config and proxy_config.docker_config.enabled:
                docker = DockerManager(ssh, proxy_config.docker_config.container_name)
                docker.exec_in_container("pkill -f load_balance_proxy || true")
            else:
                ssh.execute("pkill -f load_balance_proxy || true")
            
            self.nodes_status['proxy'].status = "stopped"
            self.nodes_status['proxy'].message = "Stopped"
            
            ssh.disconnect()
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop proxy: {e}")
            return False
    
    def get_status(self) -> Dict[str, dict]:
        """
        Get deployment status for all nodes.
        
        Returns:
            Dictionary of node statuses
        """
        return {name: status.to_dict() for name, status in self.nodes_status.items()}
    
    def print_status(self):
        """Print deployment status table."""
        from rich.console import Console
        from rich.table import Table
        
        console = Console()
        table = Table(title="vLLM-PD Deployment Status")
        
        table.add_column("Node", style="cyan")
        table.add_column("Role", style="magenta")
        table.add_column("Status", style="green")
        table.add_column("Health", style="yellow")
        table.add_column("Message")
        
        for name, status in self.nodes_status.items():
            table.add_row(
                status.name,
                status.role,
                status.status,
                status.health,
                status.message
            )
        
        console.print(table)
    
    def get_logs(self, node_name: str, tail: int = 100) -> str:
        """
        Get logs from a node.
        
        Args:
            node_name: Name of the node
            tail: Number of lines to retrieve
            
        Returns:
            Log content
        """
        # Find node
        node = None
        for n in self.config.get_prefill_nodes() + self.config.get_decode_nodes():
            if n.name == node_name:
                node = n
                break
        
        if not node:
            return f"Node {node_name} not found"
        
        try:
            ssh = SSHClient(
                host=node.host,
                user=node.ssh_config.user,
                port=node.ssh_config.port,
                key_file=node.ssh_config.key_file,
                password=node.ssh_config.password
            )
            ssh.connect()
            
            workdir = node.docker_config.workdir if node.docker_config.enabled else "/mnt/sfs_turbo/glm5_PD_deploy"
            
            if node.docker_config.enabled:
                docker = DockerManager(ssh, node.docker_config.container_name)
                logs = docker.exec_in_container(f"tail -n {tail} {workdir}/glm.log")
                return logs[1]  # stdout
            else:
                exit_code, stdout, stderr = ssh.execute(f"tail -n {tail} {workdir}/glm.log")
                return stdout if exit_code == 0 else stderr
                
        except Exception as e:
            return f"Failed to get logs: {e}"
        finally:
            if 'ssh' in locals():
                ssh.disconnect()

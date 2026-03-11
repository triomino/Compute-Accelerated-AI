"""
Configuration management module for vLLM-PD deployer.
"""

import yaml
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path


@dataclass
class SSHConfig:
    """SSH connection configuration."""
    user: str
    host: str
    port: int = 22
    key_file: Optional[str] = None
    password: Optional[str] = None


@dataclass
class DockerConfig:
    """Docker container configuration."""
    enabled: bool = False
    container_name: Optional[str] = None
    image: Optional[str] = None
    workdir: str = "/workspace"


@dataclass
class NodeConfig:
    """Node configuration data class."""
    name: str
    host: str
    ssh_config: SSHConfig
    docker_config: DockerConfig
    local_dp_size: int
    dp_rank_start: int
    dp_rpc_port: int
    vllm_start_port: int
    nic_name: str
    env_override: Dict[str, Any] = field(default_factory=dict)
    tp_size: int = 1  # Will be set from parent group
    dp_master_host: str = ""  # Will be set during validation


@dataclass
class ProxyConfig:
    """Proxy/load balancer configuration."""
    enabled: bool = True
    host: str = ""
    port: int = 8000
    ssh_config: Optional[SSHConfig] = None
    docker_config: Optional[DockerConfig] = None
    script_path: str = "load_balance_proxy_server_fixed.py"


class ConfigValidationError(Exception):
    """Configuration validation error."""
    pass


class Config:
    """Configuration manager for vLLM-PD deployment."""
    
    def __init__(self, config_path: str):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file.
        """
        self.config_path = Path(config_path)
        self.data = self._load_config()
        self._validate()
        self._parse()
    
    def _load_config(self) -> dict:
        """Load and parse YAML configuration file."""
        if not self.config_path.exists():
            raise ConfigValidationError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigValidationError(f"Invalid YAML format: {e}")
    
    def _validate(self):
        """Validate configuration structure and values."""
        required_sections = ['global', 'prefill', 'decode']
        for section in required_sections:
            if section not in self.data:
                raise ConfigValidationError(f"Missing required section: {section}")
        
        # Validate global section
        if 'model' not in self.data['global']:
            raise ConfigValidationError("Missing 'global.model' section")
        if 'path' not in self.data['global']['model']:
            raise ConfigValidationError("Missing 'global.model.path'")
        
        # Validate prefill section
        prefill = self.data['prefill']
        if 'nodes' not in prefill or not prefill['nodes']:
            raise ConfigValidationError("At least one prefill node is required")
        
        # Validate decode section
        decode = self.data['decode']
        if 'nodes' not in decode or not decode['nodes']:
            raise ConfigValidationError("At least one decode node is required")
        
        # Validate port conflicts
        self._validate_ports()
        
        # Validate DP rank continuity
        self._validate_dp_ranks()
    
    def _validate_ports(self):
        """Check for port conflicts across all nodes."""
        used_ports = set()
        
        # Check prefill nodes
        prefill = self.data['prefill']
        for i, node in enumerate(prefill.get('nodes', [])):
            rpc_port = node.get('dp_rpc_port')
            if rpc_port in used_ports:
                raise ConfigValidationError(
                    f"Port conflict: dp_rpc_port {rpc_port} is used multiple times"
                )
            used_ports.add(rpc_port)
            
            # Check vLLM ports range
            start_port = node.get('vllm_start_port', 9100)
            local_dp = node.get('local_dp_size', 1)
            for j in range(local_dp):
                port = start_port + j
                if port in used_ports:
                    raise ConfigValidationError(
                        f"Port conflict: vLLM port {port} in node {node.get('name', i)}"
                    )
                used_ports.add(port)
        
        # Check decode nodes
        decode = self.data['decode']
        for i, node in enumerate(decode.get('nodes', [])):
            rpc_port = node.get('dp_rpc_port')
            if rpc_port in used_ports:
                raise ConfigValidationError(
                    f"Port conflict: dp_rpc_port {rpc_port} is used multiple times"
                )
            used_ports.add(rpc_port)
            
            # Check vLLM ports range
            start_port = node.get('vllm_start_port', 9100)
            local_dp = node.get('local_dp_size', 1)
            for j in range(local_dp):
                port = start_port + j
                if port in used_ports:
                    raise ConfigValidationError(
                        f"Port conflict: vLLM port {port} in node {node.get('name', i)}"
                    )
                used_ports.add(port)
    
    def _validate_dp_ranks(self):
        """Validate DP rank assignments."""
        # Validate prefill nodes
        prefill = self.data['prefill']
        total_dp = prefill.get('dp_size', 0)
        ranks = set()
        
        for node in prefill.get('nodes', []):
            start = node.get('dp_rank_start', 0)
            local = node.get('local_dp_size', 1)
            for i in range(local):
                rank = start + i
                if rank in ranks:
                    raise ConfigValidationError(
                        f"Duplicate DP rank {rank} in prefill nodes"
                    )
                if rank >= total_dp:
                    raise ConfigValidationError(
                        f"DP rank {rank} exceeds prefill dp_size {total_dp}"
                    )
                ranks.add(rank)
        
        if len(ranks) != total_dp:
            raise ConfigValidationError(
                f"Prefill DP ranks incomplete: expected {total_dp}, got {len(ranks)}"
            )
        
        # Validate decode nodes
        decode = self.data['decode']
        total_dp = decode.get('dp_size', 0)
        ranks = set()
        
        for node in decode.get('nodes', []):
            start = node.get('dp_rank_start', 0)
            local = node.get('local_dp_size', 1)
            for i in range(local):
                rank = start + i
                if rank in ranks:
                    raise ConfigValidationError(
                        f"Duplicate DP rank {rank} in decode nodes"
                    )
                if rank >= total_dp:
                    raise ConfigValidationError(
                        f"DP rank {rank} exceeds decode dp_size {total_dp}"
                    )
                ranks.add(rank)
        
        if len(ranks) != total_dp:
            raise ConfigValidationError(
                f"Decode DP ranks incomplete: expected {total_dp}, got {len(ranks)}"
            )
    
    def _parse(self):
        """Parse configuration into structured objects."""
        # Parse global config
        self.global_config = self.data['global']
        
        # Parse prefill nodes
        prefill_data = self.data['prefill']
        self.prefill_tp_size = prefill_data.get('tp_size', 1)
        self.prefill_dp_size = prefill_data.get('dp_size', 1)
        
        # Get first prefill node as DP master
        first_prefill = prefill_data['nodes'][0]
        self.prefill_master_host = first_prefill.get('host', '')
        
        self.prefill_nodes = []
        for node_data in prefill_data['nodes']:
            self.prefill_nodes.append(self._parse_node(node_data, 'prefill'))
        
        # Parse decode nodes
        decode_data = self.data['decode']
        self.decode_tp_size = decode_data.get('tp_size', 1)
        self.decode_dp_size = decode_data.get('dp_size', 1)
        
        # Get first decode node as DP master
        first_decode = decode_data['nodes'][0]
        self.decode_master_host = first_decode.get('host', '')
        
        self.decode_nodes = []
        for node_data in decode_data['nodes']:
            self.decode_nodes.append(self._parse_node(node_data, 'decode'))
        
        # Parse proxy config
        proxy_data = self.data.get('proxy', {})
        self.proxy_config = self._parse_proxy(proxy_data)
    
    def _parse_node(self, node_data: dict, role: str) -> NodeConfig:
        """Parse a single node configuration."""
        ssh_data = node_data.get('ssh', {})
        docker_data = node_data.get('docker', {})
        
        ssh_config = SSHConfig(
            user=ssh_data.get('user', 'root'),
            host=node_data.get('host', ''),
            port=ssh_data.get('port', 22),
            key_file=ssh_data.get('key_file'),
            password=ssh_data.get('password')
        )
        
        docker_config = DockerConfig(
            enabled=docker_data.get('enabled', False),
            container_name=docker_data.get('container_name'),
            image=docker_data.get('image'),
            workdir=docker_data.get('workdir', '/workspace')
        )
        
        tp_size = self.prefill_tp_size if role == 'prefill' else self.decode_tp_size
        dp_master = self.prefill_master_host if role == 'prefill' else self.decode_master_host
        
        return NodeConfig(
            name=node_data.get('name', 'unnamed'),
            host=node_data.get('host', ''),
            ssh_config=ssh_config,
            docker_config=docker_config,
            local_dp_size=node_data.get('local_dp_size', 1),
            dp_rank_start=node_data.get('dp_rank_start', 0),
            dp_rpc_port=node_data.get('dp_rpc_port', 12345),
            vllm_start_port=node_data.get('vllm_start_port', 9100),
            nic_name=node_data.get('nic_name', 'eth0'),
            env_override=node_data.get('env_override', {}),
            tp_size=tp_size,
            dp_master_host=dp_master
        )
    
    def _parse_proxy(self, proxy_data: dict) -> ProxyConfig:
        """Parse proxy configuration."""
        ssh_data = proxy_data.get('ssh', {})
        docker_data = proxy_data.get('docker', {})
        
        ssh_config = None
        if ssh_data:
            ssh_config = SSHConfig(
                user=ssh_data.get('user', 'root'),
                host=proxy_data.get('host', ''),
                port=ssh_data.get('port', 22),
                key_file=ssh_data.get('key_file'),
                password=ssh_data.get('password')
            )
        
        docker_config = None
        if docker_data:
            docker_config = DockerConfig(
                enabled=docker_data.get('enabled', False),
                container_name=docker_data.get('container_name'),
                image=docker_data.get('image'),
                workdir=docker_data.get('workdir', '/workspace')
            )
        
        return ProxyConfig(
            enabled=proxy_data.get('enabled', True),
            host=proxy_data.get('host', ''),
            port=proxy_data.get('port', 8000),
            ssh_config=ssh_config,
            docker_config=docker_config,
            script_path=proxy_data.get('script_path', 'load_balance_proxy_server_fixed.py')
        )
    
    def get_prefill_nodes(self) -> List[NodeConfig]:
        """Get all prefill node configurations."""
        return self.prefill_nodes
    
    def get_decode_nodes(self) -> List[NodeConfig]:
        """Get all decode node configurations."""
        return self.decode_nodes
    
    def get_proxy_config(self) -> ProxyConfig:
        """Get proxy configuration."""
        return self.proxy_config
    
    def get_global_config(self) -> dict:
        """Get global configuration."""
        return self.global_config
    
    def get_prefill_summary(self) -> dict:
        """Get prefill layer summary."""
        return {
            'dp_size': self.prefill_dp_size,
            'tp_size': self.prefill_tp_size,
            'node_count': len(self.prefill_nodes),
            'master_host': self.prefill_master_host
        }
    
    def get_decode_summary(self) -> dict:
        """Get decode layer summary."""
        return {
            'dp_size': self.decode_dp_size,
            'tp_size': self.decode_tp_size,
            'node_count': len(self.decode_nodes),
            'master_host': self.decode_master_host
        }
    
    def get_all_hosts(self) -> List[str]:
        """Get all unique host addresses."""
        hosts = set()
        for node in self.prefill_nodes:
            hosts.add(node.host)
        for node in self.decode_nodes:
            hosts.add(node.host)
        if self.proxy_config.enabled and self.proxy_config.host:
            hosts.add(self.proxy_config.host)
        return list(hosts)
    
    def get_kv_transfer_config(self, role: str) -> dict:
        """Get KV transfer configuration for specified role."""
        kv_config = self.global_config.get('kv_transfer', {})
        
        return {
            'kv_connector': kv_config.get('connector', 'MooncakeConnectorV1'),
            'kv_role': 'kv_producer' if role == 'prefill' else 'kv_consumer',
            'kv_port': kv_config.get('prefill_port', '30000') if role == 'prefill' else kv_config.get('decode_port', '30199'),
            'engine_id': '0' if role == 'prefill' else '1',
            'kv_connector_extra_config': {
                'use_ascend_direct': kv_config.get('use_ascend_direct', True),
                'prefill': {
                    'dp_size': self.prefill_dp_size,
                    'tp_size': self.prefill_tp_size
                },
                'decode': {
                    'dp_size': self.decode_dp_size,
                    'tp_size': self.decode_tp_size
                }
            }
        }

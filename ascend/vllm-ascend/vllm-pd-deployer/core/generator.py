"""
Script generator module for creating deployment scripts from templates.
"""

import json
import os
from pathlib import Path
from typing import Optional
from jinja2 import Environment, FileSystemLoader, Template

from .config import Config, NodeConfig


class ScriptGenerator:
    """Generator for deployment scripts using Jinja2 templates."""
    
    def __init__(self, template_dir: str):
        """
        Initialize script generator.
        
        Args:
            template_dir: Directory containing Jinja2 templates
        """
        self.template_dir = Path(template_dir)
        if not self.template_dir.exists():
            raise ValueError(f"Template directory not found: {template_dir}")
        
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Add custom filters
        self.env.filters['tojson'] = json.dumps
    
    def generate_server_sh(self, node: NodeConfig, role: str, dp_size: int, tp_size: int, workdir: str) -> str:
        """
        Generate server.sh script.
        
        Args:
            node: Node configuration
            role: 'prefill' or 'decode'
            dp_size: Total DP size for this role
            tp_size: TP size
            workdir: Working directory in container
            
        Returns:
            Generated script content
        """
        template = self.env.get_template('server.sh.j2')
        return template.render(
            node=node,
            role=role,
            dp_size=dp_size,
            tp_size=tp_size,
            workdir=workdir
        )
    
    def generate_run_dp_template(
        self,
        node: NodeConfig,
        role: str,
        global_config: dict
    ) -> str:
        """
        Generate run_dp_template.sh script.
        
        Args:
            node: Node configuration
            role: 'prefill' or 'decode'
            global_config: Global configuration dict
            
        Returns:
            Generated script content
        """
        template = self.env.get_template('run_dp_template.sh.j2')
        
        # Extract configurations
        model_config = global_config.get('model', {})
        vllm_config = global_config.get('vllm', {})
        speculative_config = global_config.get('speculative', {})
        env_config = global_config.get('env', {})
        
        # Prepare additional config based on role
        if role == 'prefill':
            additional_config = {
                "layer_sharding": ["q_b_proj", "o_proj"],
                "rot_path": "/mnt/sfs_turbo/stl/rot.safetensors"
            }
        else:
            additional_config = {
                "multistream_overlap_shared_expert": True,
                "rot_path": "/mnt/sfs_turbo/stl/rot.safetensors",
                "recompute_scheduler_enable": False
            }
        
        # Get KV transfer config
        kv_config = global_config.get('kv_transfer', {})
        kv_transfer_config = {
            'kv_connector': kv_config.get('connector', 'MooncakeConnectorV1'),
            'kv_role': 'kv_producer' if role == 'prefill' else 'kv_consumer',
            'kv_port': kv_config.get('prefill_port', '30000') if role == 'prefill' else kv_config.get('decode_port', '30199'),
            'engine_id': '0' if role == 'prefill' else '1',
            'kv_connector_extra_config': {
                'use_ascend_direct': kv_config.get('use_ascend_direct', True),
                'prefill': {
                    'dp_size': global_config.get('prefill_dp_size', 2),
                    'tp_size': global_config.get('prefill_tp_size', 16)
                },
                'decode': {
                    'dp_size': global_config.get('decode_dp_size', 16),
                    'tp_size': global_config.get('decode_tp_size', 4)
                }
            }
        }
        
        return template.render(
            node=node,
            role=role,
            global_env=env_config,
            model=model_config,
            vllm_config=vllm_config,
            speculative=speculative_config,
            additional_config=additional_config,
            kv_transfer_config=kv_transfer_config
        )
    
    def generate_launch_online_dp(self, node: NodeConfig, role: str) -> str:
        """
        Generate launch_online_dp.py script.
        
        Args:
            node: Node configuration
            role: 'prefill' or 'decode'
            
        Returns:
            Generated script content
        """
        template = self.env.get_template('launch_online_dp.py.j2')
        return template.render(
            node=node,
            role=role
        )
    
    def generate_proxy_sh(self, config: Config) -> str:
        """
        Generate proxy.sh script.
        
        Args:
            config: Full configuration object
            
        Returns:
            Generated script content
        """
        template = self.env.get_template('proxy.sh.j2')
        
        proxy_config = config.get_proxy_config()
        
        # Generate prefill hosts and ports
        prefill_hosts = []
        prefill_ports = []
        for node in config.get_prefill_nodes():
            for i in range(node.local_dp_size):
                prefill_hosts.append(node.host)
                prefill_ports.append(node.vllm_start_port + i)
        
        # Generate decode hosts and ports
        decode_hosts = []
        decode_ports = []
        for node in config.get_decode_nodes():
            for i in range(node.local_dp_size):
                decode_hosts.append(node.host)
                decode_ports.append(node.vllm_start_port + i)
        
        return template.render(
            proxy=proxy_config,
            prefill_hosts=prefill_hosts,
            prefill_ports=prefill_ports,
            decode_hosts=decode_hosts,
            decode_ports=decode_ports
        )
    
    def generate_all(self, config: Config, output_dir: str) -> dict:
        """
        Generate all scripts for the deployment.
        
        Args:
            config: Full configuration object
            output_dir: Output directory for generated scripts
            
        Returns:
            Dictionary mapping node names to generated file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_files = {}
        global_config = config.get_global_config()
        
        # Add DP/TP sizes to global config for template rendering
        global_config['prefill_dp_size'] = config.prefill_dp_size
        global_config['prefill_tp_size'] = config.prefill_tp_size
        global_config['decode_dp_size'] = config.decode_dp_size
        global_config['decode_tp_size'] = config.decode_tp_size
        
        # Generate scripts for prefill nodes
        for node in config.get_prefill_nodes():
            node_dir = output_path / 'prefill' / node.name
            node_dir.mkdir(parents=True, exist_ok=True)
            
            # server.sh
            server_sh = self.generate_server_sh(
                node, 'prefill', config.prefill_dp_size,
                config.prefill_tp_size, node.docker_config.workdir
            )
            server_sh_path = node_dir / 'server.sh'
            with open(server_sh_path, 'w') as f:
                f.write(server_sh)
            
            # run_dp_template.sh
            run_dp_sh = self.generate_run_dp_template(node, 'prefill', global_config)
            run_dp_path = node_dir / 'run_dp_template.sh'
            with open(run_dp_path, 'w') as f:
                f.write(run_dp_sh)
            
            # launch_online_dp.py
            launch_py = self.generate_launch_online_dp(node, 'prefill')
            launch_path = node_dir / 'launch_online_dp.py'
            with open(launch_path, 'w') as f:
                f.write(launch_py)
            
            generated_files[f"prefill/{node.name}"] = {
                'server.sh': str(server_sh_path),
                'run_dp_template.sh': str(run_dp_path),
                'launch_online_dp.py': str(launch_path)
            }
        
        # Generate scripts for decode nodes
        for node in config.get_decode_nodes():
            node_dir = output_path / 'decode' / node.name
            node_dir.mkdir(parents=True, exist_ok=True)
            
            # server.sh
            server_sh = self.generate_server_sh(
                node, 'decode', config.decode_dp_size,
                config.decode_tp_size, node.docker_config.workdir
            )
            server_sh_path = node_dir / 'server.sh'
            with open(server_sh_path, 'w') as f:
                f.write(server_sh)
            
            # run_dp_template.sh
            run_dp_sh = self.generate_run_dp_template(node, 'decode', global_config)
            run_dp_path = node_dir / 'run_dp_template.sh'
            with open(run_dp_path, 'w') as f:
                f.write(run_dp_sh)
            
            # launch_online_dp.py
            launch_py = self.generate_launch_online_dp(node, 'decode')
            launch_path = node_dir / 'launch_online_dp.py'
            with open(launch_path, 'w') as f:
                f.write(launch_py)
            
            generated_files[f"decode/{node.name}"] = {
                'server.sh': str(server_sh_path),
                'run_dp_template.sh': str(run_dp_path),
                'launch_online_dp.py': str(launch_path)
            }
        
        # Generate proxy script if enabled
        proxy_config = config.get_proxy_config()
        if proxy_config.enabled:
            proxy_dir = output_path / 'proxy'
            proxy_dir.mkdir(parents=True, exist_ok=True)
            
            proxy_sh = self.generate_proxy_sh(config)
            proxy_path = proxy_dir / 'proxy.sh'
            with open(proxy_path, 'w') as f:
                f.write(proxy_sh)
            
            generated_files['proxy'] = {
                'proxy.sh': str(proxy_path)
            }
        
        return generated_files

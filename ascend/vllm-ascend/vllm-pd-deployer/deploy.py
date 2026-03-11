#!/usr/bin/env python3
"""
vLLM-Ascend PD分离一键部署工具

Usage:
    python deploy.py deploy --config config.yaml
    python deploy.py status --config config.yaml
    python deploy.py stop --config config.yaml
    python deploy.py logs --config config.yaml --node <node_name>
    python deploy.py validate --config config.yaml
    python deploy.py generate --config config.yaml --output ./generated
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Add core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

from core.config import Config, ConfigValidationError
from core.deployer import Deployer
from core.generator import ScriptGenerator

# Setup logging
def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def cmd_deploy(args):
    """Handle deploy command."""
    try:
        setup_logging(args.verbose)
        
        print(f"Loading configuration from {args.config}...")
        config = Config(args.config)
        
        print("Configuration loaded successfully!")
        print(f"  Prefill: {config.prefill_dp_size} DP x {config.prefill_tp_size} TP")
        print(f"  Decode:  {config.decode_dp_size} DP x {config.decode_tp_size} TP")
        print(f"  Nodes:   {len(config.get_prefill_nodes())} P + {len(config.get_decode_nodes())} D")
        
        deployer = Deployer(config, args.output or "./generated")
        
        if args.dry_run:
            print("\n[DRY RUN MODE] Generating scripts only...")
            deployer.generate_scripts()
            print(f"\nScripts generated in: {args.output or './generated'}")
            print("No actual deployment performed.")
            return 0
        
        print(f"\nStarting deployment (target: {args.target})...")
        success = deployer.deploy(target=args.target, dry_run=False)
        
        if success:
            print("\n Deployment completed successfully!")
            deployer.print_status()
            return 0
        else:
            print("\n Deployment completed with errors.")
            deployer.print_status()
            return 1
            
    except ConfigValidationError as e:
        print(f"Configuration error: {e}")
        return 1
    except Exception as e:
        print(f"Deployment failed: {e}")
        logging.exception("Unexpected error")
        return 1


def cmd_stop(args):
    """Handle stop command."""
    try:
        setup_logging(args.verbose)
        
        print(f"Loading configuration from {args.config}...")
        config = Config(args.config)
        
        deployer = Deployer(config, "./generated")
        
        print(f"\nStopping services (target: {args.target})...")
        success = deployer.stop(target=args.target)
        
        if success:
            print("\n Services stopped successfully!")
            return 0
        else:
            print("\n Some services may not have stopped properly.")
            return 1
            
    except ConfigValidationError as e:
        print(f"Configuration error: {e}")
        return 1
    except Exception as e:
        print(f"Stop failed: {e}")
        return 1


def cmd_status(args):
    """Handle status command."""
    try:
        setup_logging(args.verbose)
        
        print(f"Loading configuration from {args.config}...")
        config = Config(args.config)
        
        deployer = Deployer(config, "./generated")
        
        print("\nDeployment Status:")
        deployer.print_status()
        
        return 0
        
    except ConfigValidationError as e:
        print(f"Configuration error: {e}")
        return 1
    except Exception as e:
        print(f"Status check failed: {e}")
        return 1


def cmd_logs(args):
    """Handle logs command."""
    try:
        setup_logging(args.verbose)
        
        print(f"Loading configuration from {args.config}...")
        config = Config(args.config)
        
        deployer = Deployer(config, "./generated")
        
        print(f"\nRetrieving logs for {args.node} (last {args.tail} lines)...\n")
        logs = deployer.get_logs(args.node, tail=args.tail)
        print(logs)
        
        return 0
        
    except ConfigValidationError as e:
        print(f"Configuration error: {e}")
        return 1
    except Exception as e:
        print(f"Logs retrieval failed: {e}")
        return 1


def cmd_validate(args):
    """Handle validate command."""
    try:
        print(f"Validating configuration: {args.config}")
        config = Config(args.config)
        
        print("\n Configuration is valid!")
        print(f"\nDeployment Summary:")
        print(f"  Prefill Nodes: {len(config.get_prefill_nodes())}")
        prefill_summary = config.get_prefill_summary()
        print(f"    - DP Size: {prefill_summary['dp_size']}")
        print(f"    - TP Size: {prefill_summary['tp_size']}")
        print(f"    - Master:  {prefill_summary['master_host']}")
        
        print(f"\n  Decode Nodes:  {len(config.get_decode_nodes())}")
        decode_summary = config.get_decode_summary()
        print(f"    - DP Size: {decode_summary['dp_size']}")
        print(f"    - TP Size: {decode_summary['tp_size']}")
        print(f"    - Master:  {decode_summary['master_host']}")
        
        proxy_config = config.get_proxy_config()
        if proxy_config.enabled:
            print(f"\n  Proxy: Enabled")
            print(f"    - Host: {proxy_config.host}:{proxy_config.port}")
        else:
            print(f"\n  Proxy: Disabled")
        
        print(f"\n  Total Hosts: {len(config.get_all_hosts())}")
        
        return 0
        
    except ConfigValidationError as e:
        print(f"\n Configuration validation failed:")
        print(f"  {e}")
        return 1
    except Exception as e:
        print(f"\n Validation failed: {e}")
        return 1


def cmd_generate(args):
    """Handle generate command."""
    try:
        setup_logging(args.verbose)
        
        print(f"Loading configuration from {args.config}...")
        config = Config(args.config)
        
        print("Generating deployment scripts...")
        generator = ScriptGenerator(
            os.path.join(os.path.dirname(__file__), 'templates')
        )
        
        generated = generator.generate_all(config, args.output)
        
        print(f"\n Scripts generated successfully in: {args.output}")
        print("\nGenerated files:")
        for node_name, files in generated.items():
            print(f"  {node_name}:")
            for script_name, path in files.items():
                print(f"    - {script_name}: {path}")
        
        return 0
        
    except ConfigValidationError as e:
        print(f"Configuration error: {e}")
        return 1
    except Exception as e:
        print(f"Script generation failed: {e}")
        logging.exception("Unexpected error")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog='deploy.py',
        description='vLLM-Ascend PD分离一键部署工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 验证配置文件
  python deploy.py validate --config config.yaml

  # 仅生成脚本（不部署）
  python deploy.py generate --config config.yaml --output ./generated

  # 一键部署所有服务
  python deploy.py deploy --config config.yaml

  # 仅部署P层
  python deploy.py deploy --config config.yaml --target prefill

  # 查看状态
  python deploy.py status --config config.yaml

  # 查看日志
  python deploy.py logs --config config.yaml --node p-node-1 --tail 50

  # 停止所有服务
  python deploy.py stop --config config.yaml
        """
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Deploy command
    deploy_parser = subparsers.add_parser(
        'deploy',
        help='Deploy vLLM-PD services'
    )
    deploy_parser.add_argument(
        '-c', '--config',
        required=True,
        help='Path to configuration file (YAML)'
    )
    deploy_parser.add_argument(
        '-t', '--target',
        choices=['all', 'prefill', 'decode', 'proxy'],
        default='all',
        help='Deployment target (default: all)'
    )
    deploy_parser.add_argument(
        '-o', '--output',
        help='Output directory for generated scripts'
    )
    deploy_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Generate scripts only, do not deploy'
    )
    deploy_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    deploy_parser.set_defaults(func=cmd_deploy)
    
    # Stop command
    stop_parser = subparsers.add_parser(
        'stop',
        help='Stop vLLM-PD services'
    )
    stop_parser.add_argument(
        '-c', '--config',
        required=True,
        help='Path to configuration file (YAML)'
    )
    stop_parser.add_argument(
        '-t', '--target',
        choices=['all', 'prefill', 'decode', 'proxy'],
        default='all',
        help='Stop target (default: all)'
    )
    stop_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    stop_parser.set_defaults(func=cmd_stop)
    
    # Status command
    status_parser = subparsers.add_parser(
        'status',
        help='Show deployment status'
    )
    status_parser.add_argument(
        '-c', '--config',
        required=True,
        help='Path to configuration file (YAML)'
    )
    status_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    status_parser.set_defaults(func=cmd_status)
    
    # Logs command
    logs_parser = subparsers.add_parser(
        'logs',
        help='Show logs for a specific node'
    )
    logs_parser.add_argument(
        '-c', '--config',
        required=True,
        help='Path to configuration file (YAML)'
    )
    logs_parser.add_argument(
        '-n', '--node',
        required=True,
        help='Node name (e.g., p-node-1, d-node-1)'
    )
    logs_parser.add_argument(
        '--tail',
        type=int,
        default=100,
        help='Number of lines to show (default: 100)'
    )
    logs_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    logs_parser.set_defaults(func=cmd_logs)
    
    # Validate command
    validate_parser = subparsers.add_parser(
        'validate',
        help='Validate configuration file'
    )
    validate_parser.add_argument(
        '-c', '--config',
        required=True,
        help='Path to configuration file (YAML)'
    )
    validate_parser.set_defaults(func=cmd_validate)
    
    # Generate command
    generate_parser = subparsers.add_parser(
        'generate',
        help='Generate deployment scripts without deploying'
    )
    generate_parser.add_argument(
        '-c', '--config',
        required=True,
        help='Path to configuration file (YAML)'
    )
    generate_parser.add_argument(
        '-o', '--output',
        default='./generated',
        help='Output directory for generated scripts (default: ./generated)'
    )
    generate_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    generate_parser.set_defaults(func=cmd_generate)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Show help if no command
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())

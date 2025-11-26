"""
Command Line Interface for ML Experiment Tracker.

Provides CLI commands for managing experiments, viewing runs,
generating reports, and more.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, List
from datetime import datetime


class ExperimentTrackerCLI:
    """Command-line interface for ML Experiment Tracker."""
    
    def __init__(self):
        """Initialize CLI."""
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser with all commands."""
        parser = argparse.ArgumentParser(
            prog='experiment-tracker',
            description='ML Experiment Tracker - Track and manage ML experiments',
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # List runs command
        list_parser = subparsers.add_parser('list', help='List experiment runs')
        list_parser.add_argument('--project', type=str, help='Filter by project name')
        list_parser.add_argument('--limit', type=int, default=10, help='Number of runs to show')
        list_parser.add_argument('--status', type=str, help='Filter by status')
        
        # Show run command
        show_parser = subparsers.add_parser('show', help='Show run details')
        show_parser.add_argument('run_id', type=str, help='Run ID to display')
        show_parser.add_argument('--json', action='store_true', help='Output as JSON')
        
        # Compare runs command
        compare_parser = subparsers.add_parser('compare', help='Compare multiple runs')
        compare_parser.add_argument('run_ids', nargs='+', help='Run IDs to compare')
        compare_parser.add_argument('--output', type=str, help='Output file path')
        compare_parser.add_argument('--format', choices=['text', 'markdown', 'json'], 
                                   default='text', help='Output format')
        
        # Report command
        report_parser = subparsers.add_parser('report', help='Generate HTML report')
        report_parser.add_argument('run_id', type=str, help='Run ID')
        report_parser.add_argument('--output', type=str, help='Output file path')
        
        # Plot command
        plot_parser = subparsers.add_parser('plot', help='Generate plots')
        plot_parser.add_argument('run_id', type=str, help='Run ID')
        plot_parser.add_argument('--metrics', nargs='+', help='Metrics to plot')
        plot_parser.add_argument('--output', type=str, help='Output directory')
        
        # Init command
        init_parser = subparsers.add_parser('init', help='Initialize experiment tracker')
        init_parser.add_argument('--project', type=str, required=True, help='Project name')
        
        # Clean command
        clean_parser = subparsers.add_parser('clean', help='Clean old runs')
        clean_parser.add_argument('--project', type=str, help='Project name')
        clean_parser.add_argument('--days', type=int, default=30, help='Delete runs older than N days')
        clean_parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted')
        
        # Version command
        subparsers.add_parser('version', help='Show version')
        
        return parser
    
    def run(self, args: Optional[List[str]] = None):
        """Run CLI with given arguments."""
        parsed_args = self.parser.parse_args(args)
        
        if not parsed_args.command:
            self.parser.print_help()
            return
        
        # Route to appropriate handler
        handler = getattr(self, f'_handle_{parsed_args.command}', None)
        if handler:
            handler(parsed_args)
        else:
            print(f"Unknown command: {parsed_args.command}")
            sys.exit(1)
    
    def _handle_list(self, args):
        """Handle list command."""
        from experiment_tracker.core.run import RunManager
        
        # Find experiment logs directory
        logs_dir = Path('experiment_logs')
        if not logs_dir.exists():
            print("No experiments found. Run 'experiment-tracker init' first.")
            return
        
        # Collect all runs
        runs = []
        if args.project:
            project_dirs = [logs_dir / args.project]
        else:
            project_dirs = [d for d in logs_dir.iterdir() if d.is_dir()]
        
        for project_dir in project_dirs:
            if not project_dir.exists():
                continue
            
            for run_dir in project_dir.iterdir():
                if run_dir.is_dir():
                    run_json = run_dir / 'run.json'
                    if run_json.exists():
                        with open(run_json) as f:
                            run_data = json.load(f)
                        
                        # Filter by status if specified
                        if args.status and run_data.get('status') != args.status:
                            continue
                        
                        runs.append(run_data)
        
        # Sort by created_at
        runs.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        # Limit
        runs = runs[:args.limit]
        
        # Display
        if not runs:
            print("No runs found.")
            return
        
        print(f"\n{'Run ID':<40} {'Project':<20} {'Status':<12} {'Created':<20}")
        print("-" * 92)
        
        for run in runs:
            run_id = run.get('run_id', 'Unknown')[:38]
            project = run.get('project_name', 'Unknown')[:18]
            status = run.get('status', 'unknown')[:10]
            created = run.get('created_at', '')[:19]
            
            print(f"{run_id:<40} {project:<20} {status:<12} {created:<20}")
        
        print(f"\nTotal: {len(runs)} runs")
    
    def _handle_show(self, args):
        """Handle show command."""
        # Find run
        run_file = self._find_run_file(args.run_id)
        
        if not run_file:
            print(f"Run not found: {args.run_id}")
            sys.exit(1)
        
        with open(run_file) as f:
            run_data = json.load(f)
        
        if args.json:
            print(json.dumps(run_data, indent=2))
        else:
            self._display_run(run_data)
    
    def _handle_compare(self, args):
        """Handle compare command."""
        from experiment_tracker.utils.diff_generator import DiffGenerator
        
        # Load runs
        runs_data = []
        for run_id in args.run_ids:
            run_file = self._find_run_file(run_id)
            if not run_file:
                print(f"Warning: Run not found: {run_id}")
                continue
            
            with open(run_file) as f:
                runs_data.append(json.load(f))
        
        if len(runs_data) < 2:
            print("Need at least 2 runs to compare")
            sys.exit(1)
        
        # Compare first two runs
        diff_gen = DiffGenerator()
        comparison = diff_gen.compare_runs(runs_data[0], runs_data[1])
        
        # Generate report
        if args.format == 'text':
            report = diff_gen.generate_text_report(comparison)
        elif args.format == 'markdown':
            report = diff_gen.generate_markdown_report(comparison)
        else:  # json
            report = json.dumps(comparison.to_dict(), indent=2)
        
        # Output
        if args.output:
            Path(args.output).write_text(report)
            print(f"Comparison saved to: {args.output}")
        else:
            print(report)
    
    def _handle_report(self, args):
        """Handle report command."""
        from experiment_tracker.utils.html_reporter import HTMLReporter
        
        # Find run
        run_file = self._find_run_file(args.run_id)
        
        if not run_file:
            print(f"Run not found: {args.run_id}")
            sys.exit(1)
        
        with open(run_file) as f:
            run_data = json.load(f)
        
        # Generate report
        reporter = HTMLReporter()
        output_path = args.output or run_file.parent / 'report.html'
        
        reporter.generate_run_report(run_data, output_path)
        print(f"Report generated: {output_path}")
    
    def _handle_plot(self, args):
        """Handle plot command."""
        from experiment_tracker.utils.auto_plotter import AutoPlotter
        
        # Find run
        run_file = self._find_run_file(args.run_id)
        
        if not run_file:
            print(f"Run not found: {args.run_id}")
            sys.exit(1)
        
        with open(run_file) as f:
            run_data = json.load(f)
        
        # Generate plots
        plotter = AutoPlotter()
        
        if not plotter.matplotlib_available:
            print("matplotlib not installed. Install with: pip install matplotlib")
            sys.exit(1)
        
        output_dir = Path(args.output) if args.output else run_file.parent / 'plots'
        output_dir.mkdir(exist_ok=True)
        
        # Generate training curves
        plotter.plot_training_curves(
            run_data,
            metrics=args.metrics,
            output_path=output_dir / 'training_curves.png'
        )
        
        print(f"Plots generated in: {output_dir}")
    
    def _handle_init(self, args):
        """Handle init command."""
        from experiment_tracker.core.config import Config
        
        # Create config
        config = Config(project_name=args.project)
        config.save('experiment_config.yaml')
        
        print(f"✓ Initialized project: {args.project}")
        print(f"✓ Config saved to: experiment_config.yaml")
        print(f"\nGet started:")
        print(f"  from experiment_tracker import ExperimentTracker")
        print(f"  with ExperimentTracker(project_name='{args.project}') as tracker:")
        print(f"      tracker.log_params({{'lr': 0.001}})")
    
    def _handle_clean(self, args):
        """Handle clean command."""
        from datetime import datetime, timedelta
        
        logs_dir = Path('experiment_logs')
        if not logs_dir.exists():
            print("No experiments found.")
            return
        
        cutoff_date = datetime.now() - timedelta(days=args.days)
        deleted_count = 0
        
        # Find runs to delete
        if args.project:
            project_dirs = [logs_dir / args.project]
        else:
            project_dirs = [d for d in logs_dir.iterdir() if d.is_dir()]
        
        for project_dir in project_dirs:
            if not project_dir.exists():
                continue
            
            for run_dir in project_dir.iterdir():
                if run_dir.is_dir():
                    run_json = run_dir / 'run.json'
                    if run_json.exists():
                        with open(run_json) as f:
                            run_data = json.load(f)
                        
                        created_at_str = run_data.get('created_at', '')
                        if created_at_str:
                            try:
                                created_at = datetime.fromisoformat(created_at_str)
                                if created_at < cutoff_date:
                                    if args.dry_run:
                                        print(f"Would delete: {run_data.get('run_id')} ({created_at_str})")
                                    else:
                                        import shutil
                                        shutil.rmtree(run_dir)
                                        print(f"Deleted: {run_data.get('run_id')}")
                                    deleted_count += 1
                            except Exception as e:
                                print(f"Error processing {run_dir}: {e}")
        
        if args.dry_run:
            print(f"\nWould delete {deleted_count} runs older than {args.days} days")
            print("Run without --dry-run to actually delete")
        else:
            print(f"\nDeleted {deleted_count} runs older than {args.days} days")
    
    def _handle_version(self, args):
        """Handle version command."""
        from experiment_tracker import __version__
        print(f"ML Experiment Tracker v{__version__}")
    
    def _find_run_file(self, run_id: str) -> Optional[Path]:
        """Find run JSON file by run ID."""
        logs_dir = Path('experiment_logs')
        
        if not logs_dir.exists():
            return None
        
        # Search all projects
        for project_dir in logs_dir.iterdir():
            if not project_dir.is_dir():
                continue
            
            for run_dir in project_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                
                run_json = run_dir / 'run.json'
                if run_json.exists():
                    with open(run_json) as f:
                        run_data = json.load(f)
                    
                    if run_data.get('run_id', '').startswith(run_id):
                        return run_json
        
        return None
    
    def _display_run(self, run_data: dict):
        """Display run information."""
        print("\n" + "=" * 70)
        print("RUN DETAILS")
        print("=" * 70)
        
        print(f"\nRun ID: {run_data.get('run_id', 'Unknown')}")
        print(f"Project: {run_data.get('project_name', 'Unknown')}")
        print(f"Status: {run_data.get('status', 'unknown')}")
        print(f"Created: {run_data.get('created_at', 'Unknown')}")
        
        if run_data.get('description'):
            print(f"Description: {run_data.get('description')}")
        
        # Parameters
        params = run_data.get('params', {})
        if params:
            print("\nParameters:")
            for key, value in sorted(params.items()):
                print(f"  {key}: {value}")
        
        # Metrics
        metrics = run_data.get('metrics', {})
        if metrics:
            print("\nFinal Metrics:")
            for key, values in sorted(metrics.items()):
                if isinstance(values, list) and values:
                    final_value = values[-1].get('value', 'N/A')
                else:
                    final_value = values
                print(f"  {key}: {final_value}")
        
        # System info
        system_info = run_data.get('system_info', {})
        if system_info:
            print("\nSystem Info:")
            print(f"  Python: {system_info.get('python_version', 'Unknown')}")
            print(f"  Platform: {system_info.get('platform', 'Unknown')}")
        
        print()


def main():
    """Main entry point for CLI."""
    cli = ExperimentTrackerCLI()
    cli.run()


if __name__ == '__main__':
    main()
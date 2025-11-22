"""
Auto-Plotting module for ML Experiment Tracker.

Automatically generates plots and visualizations for experiment metrics,
training curves, and comparisons.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import json


class AutoPlotter:
    """
    Automatically generates plots for experiment metrics.
    
    Creates visualizations using matplotlib for training curves,
    metric comparisons, and parameter analysis.
    """
    
    def __init__(self, style: str = 'default'):
        """
        Initialize auto plotter.
        
        Args:
            style: Matplotlib style ('default', 'seaborn', 'ggplot')
        """
        self.style = style
        self.matplotlib_available = self._check_matplotlib()
    
    def _check_matplotlib(self) -> bool:
        """Check if matplotlib is available."""
        try:
            import matplotlib
            return True
        except ImportError:
            return False
    
    def plot_training_curves(
        self,
        run_data: Dict[str, Any],
        metrics: Optional[List[str]] = None,
        output_path: Optional[Union[str, Path]] = None,
        show: bool = False
    ) -> Optional[Path]:
        """
        Plot training curves for metrics over time.
        
        Args:
            run_data: Run data dictionary
            metrics: List of metric names to plot (None = all)
            output_path: Path to save plot
            show: Whether to display plot
        
        Returns:
            Path to saved plot or None
        """
        if not self.matplotlib_available:
            print("matplotlib not installed. Install with: pip install matplotlib")
            return None
        
        import matplotlib.pyplot as plt
        
        plt.style.use(self.style)
        
        # Get metrics from run data
        run_metrics = run_data.get('metrics', {})
        
        if not run_metrics:
            print("No metrics found in run data")
            return None
        
        # Filter metrics if specified
        if metrics:
            run_metrics = {k: v for k, v in run_metrics.items() if k in metrics}
        
        # Create subplots
        n_metrics = len(run_metrics)
        if n_metrics == 0:
            return None
        
        fig, axes = plt.subplots(
            (n_metrics + 1) // 2, 2 if n_metrics > 1 else 1,
            figsize=(12, 4 * ((n_metrics + 1) // 2))
        )
        
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Plot each metric
        for idx, (metric_name, values) in enumerate(sorted(run_metrics.items())):
            ax = axes[idx]
            
            if isinstance(values, list):
                steps = [v.get('step', i) for i, v in enumerate(values)]
                metric_values = [v.get('value', 0) for v in values]
                
                ax.plot(steps, metric_values, marker='o', linewidth=2)
                ax.set_xlabel('Step')
                ax.set_ylabel(metric_name)
                ax.set_title(f'{metric_name} over time')
                ax.grid(True, alpha=0.3)
            else:
                # Single value
                ax.bar([metric_name], [values])
                ax.set_title(metric_name)
                ax.set_ylabel('Value')
        
        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to: {output_path}")
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()
        
        return output_path
    
    def plot_metric_comparison(
        self,
        runs_data: List[Dict[str, Any]],
        metric_name: str,
        output_path: Optional[Union[str, Path]] = None,
        show: bool = False
    ) -> Optional[Path]:
        """
        Plot comparison of a metric across multiple runs.
        
        Args:
            runs_data: List of run data dictionaries
            metric_name: Name of metric to compare
            output_path: Path to save plot
            show: Whether to display plot
        
        Returns:
            Path to saved plot or None
        """
        if not self.matplotlib_available:
            print("matplotlib not installed. Install with: pip install matplotlib")
            return None
        
        import matplotlib.pyplot as plt
        
        plt.style.use(self.style)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot each run
        for run in runs_data:
            run_id = run.get('run_id', 'Unknown')
            metrics = run.get('metrics', {})
            
            if metric_name not in metrics:
                continue
            
            values = metrics[metric_name]
            
            if isinstance(values, list):
                steps = [v.get('step', i) for i, v in enumerate(values)]
                metric_values = [v.get('value', 0) for v in values]
                
                ax.plot(steps, metric_values, marker='o', label=run_id, linewidth=2)
        
        ax.set_xlabel('Step')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} Comparison Across Runs')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to: {output_path}")
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()
        
        return output_path
    
    def plot_parameter_impact(
        self,
        runs_data: List[Dict[str, Any]],
        param_name: str,
        metric_name: str,
        output_path: Optional[Union[str, Path]] = None,
        show: bool = False
    ) -> Optional[Path]:
        """
        Plot the impact of a parameter on a metric.
        
        Args:
            runs_data: List of run data dictionaries
            param_name: Parameter to analyze
            metric_name: Metric to compare
            output_path: Path to save plot
            show: Whether to display plot
        
        Returns:
            Path to saved plot or None
        """
        if not self.matplotlib_available:
            print("matplotlib not installed. Install with: pip install matplotlib")
            return None
        
        import matplotlib.pyplot as plt
        
        plt.style.use(self.style)
        
        # Extract parameter values and metric values
        param_values = []
        metric_values = []
        
        for run in runs_data:
            params = run.get('params', {})
            metrics = run.get('metrics', {})
            
            if param_name not in params or metric_name not in metrics:
                continue
            
            param_val = params[param_name]
            
            # Get final metric value
            metric_vals = metrics[metric_name]
            if isinstance(metric_vals, list) and metric_vals:
                metric_val = metric_vals[-1].get('value', 0)
            else:
                metric_val = metric_vals
            
            param_values.append(param_val)
            metric_values.append(metric_val)
        
        if not param_values:
            print(f"No data found for parameter '{param_name}' and metric '{metric_name}'")
            return None
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(param_values, metric_values, s=100, alpha=0.6)
        
        ax.set_xlabel(param_name)
        ax.set_ylabel(metric_name)
        ax.set_title(f'Impact of {param_name} on {metric_name}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to: {output_path}")
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()
        
        return output_path
    
    def plot_metrics_heatmap(
        self,
        runs_data: List[Dict[str, Any]],
        output_path: Optional[Union[str, Path]] = None,
        show: bool = False
    ) -> Optional[Path]:
        """
        Create heatmap of final metric values across runs.
        
        Args:
            runs_data: List of run data dictionaries
            output_path: Path to save plot
            show: Whether to display plot
        
        Returns:
            Path to saved plot or None
        """
        if not self.matplotlib_available:
            print("matplotlib not installed. Install with: pip install matplotlib")
            return None
        
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.style.use(self.style)
        
        # Collect all metrics
        all_metrics = set()
        for run in runs_data:
            all_metrics.update(run.get('metrics', {}).keys())
        
        all_metrics = sorted(all_metrics)
        run_ids = [run.get('run_id', f'Run {i}') for i, run in enumerate(runs_data)]
        
        # Create matrix
        matrix = []
        for run in runs_data:
            row = []
            metrics = run.get('metrics', {})
            
            for metric in all_metrics:
                if metric in metrics:
                    values = metrics[metric]
                    if isinstance(values, list) and values:
                        val = values[-1].get('value', 0)
                    else:
                        val = values
                else:
                    val = 0
                
                row.append(val)
            
            matrix.append(row)
        
        matrix = np.array(matrix)
        
        # Normalize columns
        matrix_normalized = (matrix - matrix.min(axis=0)) / (matrix.max(axis=0) - matrix.min(axis=0) + 1e-10)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, max(6, len(runs_data) * 0.5)))
        im = ax.imshow(matrix_normalized, cmap='YlOrRd', aspect='auto')
        
        # Set ticks
        ax.set_xticks(range(len(all_metrics)))
        ax.set_yticks(range(len(run_ids)))
        ax.set_xticklabels(all_metrics, rotation=45, ha='right')
        ax.set_yticklabels(run_ids)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Normalized Value')
        
        # Add values to cells
        for i in range(len(run_ids)):
            for j in range(len(all_metrics)):
                text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('Metrics Heatmap Across Runs')
        plt.tight_layout()
        
        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to: {output_path}")
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()
        
        return output_path
    
    def create_dashboard(
        self,
        run_data: Dict[str, Any],
        output_dir: Union[str, Path],
        metrics: Optional[List[str]] = None
    ) -> Path:
        """
        Create a complete dashboard with multiple plots.
        
        Args:
            run_data: Run data dictionary
            output_dir: Directory to save plots
            metrics: List of metrics to plot
        
        Returns:
            Path to output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training curves
        self.plot_training_curves(
            run_data,
            metrics=metrics,
            output_path=output_dir / "training_curves.png"
        )
        
        print(f"Dashboard created in: {output_dir}")
        return output_dir
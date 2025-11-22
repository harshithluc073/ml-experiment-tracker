"""
Diff Generator for ML Experiment Tracker.

Compares experiment runs and generates detailed difference reports
showing changes in parameters, metrics, configurations, and results.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import json


class DiffType:
    """Types of differences between runs."""
    
    ADDED = "added"
    REMOVED = "removed"
    CHANGED = "changed"
    UNCHANGED = "unchanged"


class Diff:
    """
    Represents a difference between two values.
    """
    
    def __init__(
        self,
        key: str,
        old_value: Any,
        new_value: Any,
        diff_type: str
    ):
        """
        Initialize a diff.
        
        Args:
            key: Name of the parameter/metric
            old_value: Value in old run
            new_value: Value in new run
            diff_type: Type of difference
        """
        self.key = key
        self.old_value = old_value
        self.new_value = new_value
        self.diff_type = diff_type
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'key': self.key,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'diff_type': self.diff_type
        }
    
    def __repr__(self) -> str:
        if self.diff_type == DiffType.ADDED:
            return f"Diff(+{self.key}={self.new_value})"
        elif self.diff_type == DiffType.REMOVED:
            return f"Diff(-{self.key}={self.old_value})"
        elif self.diff_type == DiffType.CHANGED:
            return f"Diff({self.key}: {self.old_value} → {self.new_value})"
        else:
            return f"Diff({self.key}={self.old_value})"


class RunComparison:
    """
    Comparison between two experiment runs.
    
    Analyzes and stores all differences between runs including
    parameters, metrics, system info, and metadata.
    """
    
    def __init__(
        self,
        run1_data: Dict[str, Any],
        run2_data: Dict[str, Any]
    ):
        """
        Initialize comparison.
        
        Args:
            run1_data: Data from first run
            run2_data: Data from second run
        """
        self.run1_data = run1_data
        self.run2_data = run2_data
        
        self.run1_id = run1_data.get('run_id', 'run1')
        self.run2_id = run2_data.get('run_id', 'run2')
        
        # Perform comparisons
        self.param_diffs = self._compare_params()
        self.metric_diffs = self._compare_metrics()
        self.system_diffs = self._compare_system_info()
        self.git_diffs = self._compare_git_info()
        
        # Summary stats
        self.summary = self._generate_summary()
    
    def _compare_params(self) -> List[Diff]:
        """Compare parameters between runs."""
        params1 = self.run1_data.get('params', {})
        params2 = self.run2_data.get('params', {})
        
        return self._compare_dicts(params1, params2)
    
    def _compare_metrics(self) -> List[Diff]:
        """Compare final metrics between runs."""
        metrics1 = self.run1_data.get('metrics', {})
        metrics2 = self.run2_data.get('metrics', {})
        
        # Get final values for each metric
        final_metrics1 = {}
        final_metrics2 = {}
        
        for key, values in metrics1.items():
            if isinstance(values, list) and values:
                final_metrics1[key] = values[-1]['value']
            elif isinstance(values, (int, float)):
                final_metrics1[key] = values
        
        for key, values in metrics2.items():
            if isinstance(values, list) and values:
                final_metrics2[key] = values[-1]['value']
            elif isinstance(values, (int, float)):
                final_metrics2[key] = values
        
        return self._compare_dicts(final_metrics1, final_metrics2)
    
    def _compare_system_info(self) -> List[Diff]:
        """Compare system information."""
        system1 = self.run1_data.get('system_info', {})
        system2 = self.run2_data.get('system_info', {})
        
        return self._compare_dicts(system1, system2)
    
    def _compare_git_info(self) -> List[Diff]:
        """Compare git information."""
        git1 = self.run1_data.get('git_info', {})
        git2 = self.run2_data.get('git_info', {})
        
        return self._compare_dicts(git1, git2)
    
    def _compare_dicts(
        self,
        dict1: Dict[str, Any],
        dict2: Dict[str, Any]
    ) -> List[Diff]:
        """
        Compare two dictionaries and return list of differences.
        
        Args:
            dict1: First dictionary
            dict2: Second dictionary
        
        Returns:
            List of Diff objects
        """
        diffs = []
        all_keys = set(dict1.keys()) | set(dict2.keys())
        
        for key in sorted(all_keys):
            if key in dict1 and key in dict2:
                # Both have the key
                if dict1[key] != dict2[key]:
                    diffs.append(Diff(
                        key=key,
                        old_value=dict1[key],
                        new_value=dict2[key],
                        diff_type=DiffType.CHANGED
                    ))
                else:
                    diffs.append(Diff(
                        key=key,
                        old_value=dict1[key],
                        new_value=dict2[key],
                        diff_type=DiffType.UNCHANGED
                    ))
            elif key in dict1:
                # Only in first dict (removed)
                diffs.append(Diff(
                    key=key,
                    old_value=dict1[key],
                    new_value=None,
                    diff_type=DiffType.REMOVED
                ))
            else:
                # Only in second dict (added)
                diffs.append(Diff(
                    key=key,
                    old_value=None,
                    new_value=dict2[key],
                    diff_type=DiffType.ADDED
                ))
        
        return diffs
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate comparison summary."""
        def count_diffs(diffs: List[Diff]) -> Dict[str, int]:
            return {
                'total': len(diffs),
                'added': sum(1 for d in diffs if d.diff_type == DiffType.ADDED),
                'removed': sum(1 for d in diffs if d.diff_type == DiffType.REMOVED),
                'changed': sum(1 for d in diffs if d.diff_type == DiffType.CHANGED),
                'unchanged': sum(1 for d in diffs if d.diff_type == DiffType.UNCHANGED)
            }
        
        return {
            'run1_id': self.run1_id,
            'run2_id': self.run2_id,
            'params': count_diffs(self.param_diffs),
            'metrics': count_diffs(self.metric_diffs),
            'system_info': count_diffs(self.system_diffs),
            'git_info': count_diffs(self.git_diffs)
        }
    
    def get_changed_params(self) -> List[Diff]:
        """Get only changed parameters."""
        return [d for d in self.param_diffs if d.diff_type == DiffType.CHANGED]
    
    def get_changed_metrics(self) -> List[Diff]:
        """Get only changed metrics."""
        return [d for d in self.metric_diffs if d.diff_type == DiffType.CHANGED]
    
    def has_differences(self) -> bool:
        """Check if there are any differences."""
        return (
            any(d.diff_type != DiffType.UNCHANGED for d in self.param_diffs) or
            any(d.diff_type != DiffType.UNCHANGED for d in self.metric_diffs) or
            any(d.diff_type != DiffType.UNCHANGED for d in self.system_diffs) or
            any(d.diff_type != DiffType.UNCHANGED for d in self.git_diffs)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert comparison to dictionary."""
        return {
            'summary': self.summary,
            'params': [d.to_dict() for d in self.param_diffs],
            'metrics': [d.to_dict() for d in self.metric_diffs],
            'system_info': [d.to_dict() for d in self.system_diffs],
            'git_info': [d.to_dict() for d in self.git_diffs]
        }


class DiffGenerator:
    """
    Generates difference reports between experiment runs.
    
    Supports comparing runs from JSON files, Run objects, or raw dictionaries.
    Can generate reports in multiple formats (text, JSON, markdown).
    """
    
    def __init__(self):
        """Initialize diff generator."""
        pass
    
    def compare_runs(
        self,
        run1: Union[str, Path, Dict[str, Any]],
        run2: Union[str, Path, Dict[str, Any]]
    ) -> RunComparison:
        """
        Compare two runs.
        
        Args:
            run1: First run (path to JSON or dict)
            run2: Second run (path to JSON or dict)
        
        Returns:
            RunComparison object
        """
        # Load run data
        run1_data = self._load_run_data(run1)
        run2_data = self._load_run_data(run2)
        
        # Create comparison
        return RunComparison(run1_data, run2_data)
    
    def _load_run_data(
        self,
        run: Union[str, Path, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Load run data from various sources."""
        if isinstance(run, dict):
            return run
        
        # Assume it's a path to JSON file
        run_path = Path(run)
        
        if not run_path.exists():
            raise FileNotFoundError(f"Run file not found: {run_path}")
        
        with open(run_path, 'r') as f:
            return json.load(f)
    
    def generate_text_report(
        self,
        comparison: RunComparison,
        show_unchanged: bool = False
    ) -> str:
        """
        Generate text report of comparison.
        
        Args:
            comparison: RunComparison object
            show_unchanged: Whether to show unchanged values
        
        Returns:
            Text report as string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("EXPERIMENT RUN COMPARISON")
        lines.append("=" * 70)
        lines.append("")
        
        # Summary
        lines.append(f"Run 1: {comparison.run1_id}")
        lines.append(f"Run 2: {comparison.run2_id}")
        lines.append("")
        
        # Parameters
        lines.append("PARAMETERS")
        lines.append("-" * 70)
        lines.extend(self._format_diffs(comparison.param_diffs, show_unchanged))
        lines.append("")
        
        # Metrics
        lines.append("METRICS")
        lines.append("-" * 70)
        lines.extend(self._format_diffs(comparison.metric_diffs, show_unchanged))
        lines.append("")
        
        # System Info
        if comparison.system_diffs:
            lines.append("SYSTEM INFORMATION")
            lines.append("-" * 70)
            lines.extend(self._format_diffs(comparison.system_diffs, show_unchanged))
            lines.append("")
        
        # Git Info
        if comparison.git_diffs:
            lines.append("GIT INFORMATION")
            lines.append("-" * 70)
            lines.extend(self._format_diffs(comparison.git_diffs, show_unchanged))
            lines.append("")
        
        # Summary stats
        lines.append("SUMMARY")
        lines.append("-" * 70)
        summary = comparison.summary
        lines.append(f"Parameters: {summary['params']['changed']} changed, "
                    f"{summary['params']['added']} added, "
                    f"{summary['params']['removed']} removed")
        lines.append(f"Metrics: {summary['metrics']['changed']} changed, "
                    f"{summary['metrics']['added']} added, "
                    f"{summary['metrics']['removed']} removed")
        lines.append("")
        
        return "\n".join(lines)
    
    def _format_diffs(
        self,
        diffs: List[Diff],
        show_unchanged: bool = False
    ) -> List[str]:
        """Format list of diffs as text lines."""
        lines = []
        
        for diff in diffs:
            if diff.diff_type == DiffType.UNCHANGED and not show_unchanged:
                continue
            
            if diff.diff_type == DiffType.ADDED:
                lines.append(f"  + {diff.key}: {diff.new_value}")
            elif diff.diff_type == DiffType.REMOVED:
                lines.append(f"  - {diff.key}: {diff.old_value}")
            elif diff.diff_type == DiffType.CHANGED:
                lines.append(f"  ~ {diff.key}: {diff.old_value} → {diff.new_value}")
            else:
                lines.append(f"    {diff.key}: {diff.old_value}")
        
        if not lines:
            lines.append("  (no differences)")
        
        return lines
    
    def generate_markdown_report(
        self,
        comparison: RunComparison,
        show_unchanged: bool = False
    ) -> str:
        """
        Generate markdown report of comparison.
        
        Args:
            comparison: RunComparison object
            show_unchanged: Whether to show unchanged values
        
        Returns:
            Markdown report as string
        """
        lines = []
        lines.append("# Experiment Run Comparison")
        lines.append("")
        lines.append(f"**Run 1:** `{comparison.run1_id}`  ")
        lines.append(f"**Run 2:** `{comparison.run2_id}`")
        lines.append("")
        
        # Parameters
        lines.append("## Parameters")
        lines.append("")
        lines.extend(self._format_diffs_markdown(comparison.param_diffs, show_unchanged))
        lines.append("")
        
        # Metrics
        lines.append("## Metrics")
        lines.append("")
        lines.extend(self._format_diffs_markdown(comparison.metric_diffs, show_unchanged))
        lines.append("")
        
        # System Info
        if comparison.system_diffs:
            lines.append("## System Information")
            lines.append("")
            lines.extend(self._format_diffs_markdown(comparison.system_diffs, show_unchanged))
            lines.append("")
        
        # Git Info
        if comparison.git_diffs:
            lines.append("## Git Information")
            lines.append("")
            lines.extend(self._format_diffs_markdown(comparison.git_diffs, show_unchanged))
            lines.append("")
        
        # Summary
        lines.append("## Summary")
        lines.append("")
        summary = comparison.summary
        lines.append(f"- **Parameters:** {summary['params']['changed']} changed, "
                    f"{summary['params']['added']} added, "
                    f"{summary['params']['removed']} removed")
        lines.append(f"- **Metrics:** {summary['metrics']['changed']} changed, "
                    f"{summary['metrics']['added']} added, "
                    f"{summary['metrics']['removed']} removed")
        lines.append("")
        
        return "\n".join(lines)
    
    def _format_diffs_markdown(
        self,
        diffs: List[Diff],
        show_unchanged: bool = False
    ) -> List[str]:
        """Format list of diffs as markdown lines."""
        lines = []
        
        for diff in diffs:
            if diff.diff_type == DiffType.UNCHANGED and not show_unchanged:
                continue
            
            if diff.diff_type == DiffType.ADDED:
                lines.append(f"- ✅ **{diff.key}**: `{diff.new_value}` *(added)*")
            elif diff.diff_type == DiffType.REMOVED:
                lines.append(f"- ❌ **{diff.key}**: `{diff.old_value}` *(removed)*")
            elif diff.diff_type == DiffType.CHANGED:
                lines.append(f"- 🔄 **{diff.key}**: `{diff.old_value}` → `{diff.new_value}`")
            else:
                lines.append(f"- **{diff.key}**: `{diff.old_value}`")
        
        if not lines:
            lines.append("*No differences*")
        
        return lines
    
    def save_report(
        self,
        comparison: RunComparison,
        output_path: Union[str, Path],
        format: str = "text"
    ) -> None:
        """
        Save comparison report to file.
        
        Args:
            comparison: RunComparison object
            output_path: Path to save report
            format: Report format (text, json, markdown)
        """
        output_path = Path(output_path)
        
        if format == "text":
            report = self.generate_text_report(comparison)
            output_path.write_text(report)
        elif format == "markdown":
            report = self.generate_markdown_report(comparison)
            output_path.write_text(report)
        elif format == "json":
            report = comparison.to_dict()
            output_path.write_text(json.dumps(report, indent=2))
        else:
            raise ValueError(f"Unknown format: {format}")
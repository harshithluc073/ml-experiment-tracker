"""
HTML Reporter for ML Experiment Tracker.

Generates beautiful, interactive HTML reports for experiment runs
with charts, tables, and detailed information.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json
import base64


class HTMLReporter:
    """
    Generates HTML reports for experiment runs.
    
    Creates self-contained HTML files with embedded CSS and JavaScript
    for visualizing experiment results, parameters, metrics, and more.
    """
    
    def __init__(self):
        """Initialize HTML reporter."""
        self.css = self._get_default_css()
        self.js = self._get_default_js()
    
    def generate_run_report(
        self,
        run_data: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Generate HTML report for a single run.
        
        Args:
            run_data: Run data dictionary
            output_path: Optional path to save HTML file
        
        Returns:
            HTML string
        """
        html = self._generate_html(run_data)
        
        if output_path:
            output_path = Path(output_path)
            output_path.write_text(html, encoding='utf-8')
        
        return html
    
    def generate_comparison_report(
        self,
        runs_data: List[Dict[str, Any]],
        output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Generate HTML comparison report for multiple runs.
        
        Args:
            runs_data: List of run data dictionaries
            output_path: Optional path to save HTML file
        
        Returns:
            HTML string
        """
        html = self._generate_comparison_html(runs_data)
        
        if output_path:
            output_path = Path(output_path)
            output_path.write_text(html, encoding='utf-8')
        
        return html
    
    def _generate_html(self, run_data: Dict[str, Any]) -> str:
        """Generate HTML for single run."""
        run_id = run_data.get('run_id', 'Unknown')
        project_name = run_data.get('project_name', 'Unknown')
        status = run_data.get('status', 'unknown')
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Experiment Report - {run_id}</title>
    <style>{self.css}</style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Experiment Report</h1>
            <div class="meta">
                <span class="project">{project_name}</span>
                <span class="run-id">{run_id}</span>
                <span class="status status-{status}">{status}</span>
            </div>
        </header>
        
        {self._generate_overview_section(run_data)}
        {self._generate_params_section(run_data)}
        {self._generate_metrics_section(run_data)}
        {self._generate_system_section(run_data)}
        {self._generate_artifacts_section(run_data)}
        
        <footer>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </footer>
    </div>
    <script>{self.js}</script>
</body>
</html>"""
        
        return html
    
    def _generate_comparison_html(self, runs_data: List[Dict[str, Any]]) -> str:
        """Generate HTML for multiple runs comparison."""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Experiment Comparison Report</title>
    <style>{self.css}</style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Experiment Comparison</h1>
            <p>{len(runs_data)} runs compared</p>
        </header>
        
        {self._generate_comparison_table(runs_data)}
        
        <footer>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </footer>
    </div>
    <script>{self.js}</script>
</body>
</html>"""
        
        return html
    
    def _generate_overview_section(self, run_data: Dict[str, Any]) -> str:
        """Generate overview section."""
        created_at = run_data.get('created_at', 'Unknown')
        duration = run_data.get('duration', 'N/A')
        description = run_data.get('description', '')
        
        html = f"""
        <section class="overview">
            <h2>Overview</h2>
            <div class="info-grid">
                <div class="info-item">
                    <span class="label">Created:</span>
                    <span class="value">{created_at}</span>
                </div>
                <div class="info-item">
                    <span class="label">Duration:</span>
                    <span class="value">{duration}</span>
                </div>
"""
        
        if description:
            html += f"""
                <div class="info-item full-width">
                    <span class="label">Description:</span>
                    <span class="value">{description}</span>
                </div>
"""
        
        html += """
            </div>
        </section>
"""
        
        return html
    
    def _generate_params_section(self, run_data: Dict[str, Any]) -> str:
        """Generate parameters section."""
        params = run_data.get('params', {})
        
        if not params:
            return ""
        
        html = """
        <section class="params">
            <h2>Parameters</h2>
            <table>
                <thead>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        for key, value in sorted(params.items()):
            html += f"""
                    <tr>
                        <td><code>{key}</code></td>
                        <td>{self._format_value(value)}</td>
                    </tr>
"""
        
        html += """
                </tbody>
            </table>
        </section>
"""
        
        return html
    
    def _generate_metrics_section(self, run_data: Dict[str, Any]) -> str:
        """Generate metrics section."""
        metrics = run_data.get('metrics', {})
        
        if not metrics:
            return ""
        
        html = """
        <section class="metrics">
            <h2>Metrics</h2>
            <div class="metrics-grid">
"""
        
        for key, values in sorted(metrics.items()):
            if isinstance(values, list) and values:
                final_value = values[-1].get('value', 'N/A')
            else:
                final_value = values
            
            html += f"""
                <div class="metric-card">
                    <div class="metric-name">{key}</div>
                    <div class="metric-value">{self._format_number(final_value)}</div>
                </div>
"""
        
        html += """
            </div>
        </section>
"""
        
        return html
    
    def _generate_system_section(self, run_data: Dict[str, Any]) -> str:
        """Generate system information section."""
        system_info = run_data.get('system_info', {})
        
        if not system_info:
            return ""
        
        html = """
        <section class="system">
            <h2>System Information</h2>
            <table>
                <tbody>
"""
        
        for key, value in sorted(system_info.items()):
            formatted_key = key.replace('_', ' ').title()
            html += f"""
                    <tr>
                        <td>{formatted_key}</td>
                        <td>{self._format_value(value)}</td>
                    </tr>
"""
        
        html += """
                </tbody>
            </table>
        </section>
"""
        
        return html
    
    def _generate_artifacts_section(self, run_data: Dict[str, Any]) -> str:
        """Generate artifacts section."""
        artifacts = run_data.get('artifacts', [])
        
        if not artifacts:
            return ""
        
        html = """
        <section class="artifacts">
            <h2>Artifacts</h2>
            <table>
                <thead>
                    <tr>
                        <th>Name</th>
                        <th>Type</th>
                        <th>Path</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        for artifact in artifacts:
            name = artifact.get('name', 'Unknown')
            artifact_type = artifact.get('type', 'unknown')
            path = artifact.get('path', '')
            
            html += f"""
                    <tr>
                        <td>{name}</td>
                        <td><span class="badge">{artifact_type}</span></td>
                        <td><code>{path}</code></td>
                    </tr>
"""
        
        html += """
                </tbody>
            </table>
        </section>
"""
        
        return html
    
    def _generate_comparison_table(self, runs_data: List[Dict[str, Any]]) -> str:
        """Generate comparison table for multiple runs."""
        # Collect all unique parameter keys
        all_params = set()
        for run in runs_data:
            all_params.update(run.get('params', {}).keys())
        
        # Collect all unique metric keys
        all_metrics = set()
        for run in runs_data:
            all_metrics.update(run.get('metrics', {}).keys())
        
        html = """
        <section class="comparison">
            <h2>Parameters Comparison</h2>
            <table>
                <thead>
                    <tr>
                        <th>Parameter</th>
"""
        
        # Header with run IDs
        for run in runs_data:
            run_id = run.get('run_id', 'Unknown')
            html += f"                        <th>{run_id}</th>\n"
        
        html += """
                    </tr>
                </thead>
                <tbody>
"""
        
        # Parameter rows
        for param in sorted(all_params):
            html += f"""
                    <tr>
                        <td><code>{param}</code></td>
"""
            for run in runs_data:
                value = run.get('params', {}).get(param, '-')
                html += f"                        <td>{self._format_value(value)}</td>\n"
            
            html += "                    </tr>\n"
        
        html += """
                </tbody>
            </table>
            
            <h2>Metrics Comparison</h2>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
"""
        
        # Header with run IDs
        for run in runs_data:
            run_id = run.get('run_id', 'Unknown')
            html += f"                        <th>{run_id}</th>\n"
        
        html += """
                    </tr>
                </thead>
                <tbody>
"""
        
        # Metric rows
        for metric in sorted(all_metrics):
            html += f"""
                    <tr>
                        <td><code>{metric}</code></td>
"""
            for run in runs_data:
                metrics = run.get('metrics', {})
                values = metrics.get(metric, [])
                
                if isinstance(values, list) and values:
                    final_value = values[-1].get('value', '-')
                else:
                    final_value = values
                
                html += f"                        <td>{self._format_number(final_value)}</td>\n"
            
            html += "                    </tr>\n"
        
        html += """
                </tbody>
            </table>
        </section>
"""
        
        return html
    
    def _format_value(self, value: Any) -> str:
        """Format value for display."""
        if isinstance(value, bool):
            return '✓' if value else '✗'
        elif isinstance(value, (int, float)):
            return self._format_number(value)
        elif value is None or value == '-':
            return '-'
        else:
            return str(value)
    
    def _format_number(self, value: Union[int, float]) -> str:
        """Format number for display."""
        if isinstance(value, float):
            if abs(value) < 0.01:
                return f"{value:.6f}"
            else:
                return f"{value:.4f}"
        return str(value)
    
    def _get_default_css(self) -> str:
        """Get default CSS styles."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        header {
            border-bottom: 2px solid #007bff;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        
        h1 {
            font-size: 32px;
            color: #007bff;
            margin-bottom: 10px;
        }
        
        h2 {
            font-size: 24px;
            color: #333;
            margin: 30px 0 15px 0;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
        }
        
        .meta {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }
        
        .meta span {
            padding: 5px 12px;
            background: #f0f0f0;
            border-radius: 4px;
            font-size: 14px;
        }
        
        .status {
            font-weight: bold;
        }
        
        .status-completed {
            background: #d4edda;
            color: #155724;
        }
        
        .status-failed {
            background: #f8d7da;
            color: #721c24;
        }
        
        .status-running {
            background: #fff3cd;
            color: #856404;
        }
        
        section {
            margin-bottom: 40px;
        }
        
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        
        .info-item {
            display: flex;
            flex-direction: column;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 4px;
        }
        
        .info-item.full-width {
            grid-column: 1 / -1;
        }
        
        .label {
            font-weight: bold;
            color: #666;
            font-size: 14px;
            margin-bottom: 5px;
        }
        
        .value {
            color: #333;
            font-size: 16px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        th {
            background: #f8f9fa;
            font-weight: 600;
            color: #333;
        }
        
        tr:hover {
            background: #f8f9fa;
        }
        
        code {
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 14px;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        
        .metric-card {
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 8px;
            color: white;
        }
        
        .metric-name {
            font-size: 14px;
            opacity: 0.9;
            margin-bottom: 8px;
        }
        
        .metric-value {
            font-size: 28px;
            font-weight: bold;
        }
        
        .badge {
            display: inline-block;
            padding: 4px 8px;
            background: #007bff;
            color: white;
            border-radius: 3px;
            font-size: 12px;
            font-weight: 600;
        }
        
        footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #666;
            font-size: 14px;
        }
        
        @media print {
            body {
                background: white;
                padding: 0;
            }
            
            .container {
                box-shadow: none;
                padding: 20px;
            }
        }
"""
    
    def _get_default_js(self) -> str:
        """Get default JavaScript."""
        return """
        // Add any interactive features here
        console.log('Experiment report loaded');
"""
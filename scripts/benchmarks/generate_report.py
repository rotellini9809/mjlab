"""Generate benchmark report from evaluation metrics.

This script runs policy evaluation on nightly runs and generates a static HTML
dashboard for tracking policy performance over time.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import tyro
import wandb

from mjlab.tasks.tracking.scripts.evaluate import EvaluateConfig, run_evaluate

# Metrics to display: (key, label, unit, scale, higher_is_better)
METRICS = [
  ("success_rate", "Success Rate", "%", 100, True),
  ("mpkpe", "MPKPE", "m", 1, False),
  ("r_mpkpe", "R-MPKPE", "m", 1, False),
  ("ee_pos_error", "EE Position Error", "m", 1, False),
  ("ee_ori_error", "EE Orientation Error", "rad", 1, False),
  ("joint_vel_error", "Joint Velocity Error", "rad/s", 1, False),
]


def evaluate_run(run_path: str, num_envs: int = 1024) -> dict:
  """Evaluate a single run and return metrics with metadata."""
  api = wandb.Api()
  run = api.run(run_path)

  print(f"Evaluating run: {run.name} ({run.id})")

  cfg = EvaluateConfig(wandb_run_path=run_path, num_envs=num_envs)
  metrics = run_evaluate("Mjlab-Tracking-Flat-Unitree-G1", cfg)

  # Get commit SHA from run metadata.
  commit = run.commit or run.config.get("commit", "unknown")

  return {
    "id": run.id,
    "name": run.name,
    "url": run.url,
    "created_at": run.created_at,
    "commit": commit[:7] if len(commit) > 7 else commit,
    "metrics": metrics,
  }


def generate_html_report(runs: list[dict], output_dir: Path) -> None:
  """Generate static HTML dashboard from evaluation data."""
  output_dir.mkdir(parents=True, exist_ok=True)

  # Save raw data.
  with open(output_dir / "data.json", "w") as f:
    json.dump(runs, f, indent=2, default=str)

  html = generate_dashboard_html(runs)
  with open(output_dir / "index.html", "w") as f:
    f.write(html)

  print(f"Report generated at {output_dir / 'index.html'}")


def generate_dashboard_html(runs: list[dict]) -> str:
  """Generate the HTML dashboard content."""
  runs_json = json.dumps(runs, default=str)
  metrics_json = json.dumps(METRICS)
  timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

  return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MJLab Nightly Tracking Benchmark</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
    <style>
        :root {{
            --bg: #0d1117;
            --bg-card: #161b22;
            --text: #c9d1d9;
            --text-dim: #8b949e;
            --border: #30363d;
            --accent: #58a6ff;
            --green: #3fb950;
            --red: #f85149;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: var(--bg);
            color: var(--text);
            padding: 2rem;
            max-width: 1200px;
            margin: 0 auto;
        }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border);
        }}
        h1 {{ font-size: 1.5rem; }}
        .timestamp {{ color: var(--text-dim); font-size: 0.875rem; }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .stat {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
        }}
        .stat-label {{ font-size: 0.75rem; color: var(--text-dim); text-transform: uppercase; }}
        .stat-value {{ font-size: 1.5rem; font-weight: 600; margin-top: 0.25rem; }}
        .stat-value.good {{ color: var(--green); }}
        .stat-value.bad {{ color: var(--red); }}
        .charts {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        .chart-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
        }}
        .chart-title {{
            font-size: 0.875rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
            display: flex;
            justify-content: space-between;
        }}
        .chart-value {{ color: var(--text-dim); }}
        .chart-container {{ height: 180px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            font-size: 0.75rem;
            color: var(--text-dim);
            text-transform: uppercase;
        }}
        a {{ color: var(--accent); text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <header>
        <h1>MJLab Nightly Tracking Benchmark</h1>
        <span class="timestamp">Updated: {timestamp}</span>
    </header>

    <div class="summary" id="summary"></div>
    <div class="charts" id="charts"></div>

    <table>
        <thead>
            <tr>
                <th>Date</th>
                <th>Commit</th>
                <th>Run</th>
                <th>Success Rate</th>
                <th>MPKPE (m)</th>
                <th>EE Pos Error (m)</th>
            </tr>
        </thead>
        <tbody id="table-body"></tbody>
    </table>

    <script>
        const runs = {runs_json};
        const METRICS = {metrics_json};

        // Sort by date ascending for charts.
        runs.sort((a, b) => new Date(a.created_at) - new Date(b.created_at));

        const colors = {{
            success_rate: '#3fb950',
            mpkpe: '#58a6ff',
            r_mpkpe: '#a371f7',
            ee_pos_error: '#f0883e',
            ee_ori_error: '#f85149',
            joint_vel_error: '#79c0ff'
        }};

        // Summary cards
        const summary = document.getElementById('summary');
        const latest = runs[runs.length - 1];
        if (latest) {{
            summary.innerHTML = `
                <div class="stat">
                    <div class="stat-label">Total Runs</div>
                    <div class="stat-value">${{runs.length}}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Success Rate</div>
                    <div class="stat-value ${{latest.metrics.success_rate >= 0.95 ? 'good' : latest.metrics.success_rate < 0.8 ? 'bad' : ''}}">
                        ${{(latest.metrics.success_rate * 100).toFixed(1)}}%
                    </div>
                </div>
                <div class="stat">
                    <div class="stat-label">MPKPE</div>
                    <div class="stat-value">${{(latest.metrics.mpkpe * 100).toFixed(1)}} cm</div>
                </div>
                <div class="stat">
                    <div class="stat-label">EE Position Error</div>
                    <div class="stat-value">${{(latest.metrics.ee_pos_error * 100).toFixed(1)}} cm</div>
                </div>
            `;
        }}

        // Charts
        const chartsContainer = document.getElementById('charts');
        Chart.defaults.color = '#8b949e';
        Chart.defaults.borderColor = '#30363d';

        METRICS.forEach(([key, label, unit, scale, higherIsBetter]) => {{
            const data = runs.map(r => ({{
                x: new Date(r.created_at),
                y: r.metrics[key] * scale,
                commit: r.commit,
                name: r.name
            }}));

            const latestVal = data[data.length - 1]?.y;
            const arrow = higherIsBetter ? '↑' : '↓';
            const tooltip = higherIsBetter ? 'Higher is better' : 'Lower is better';

            const card = document.createElement('div');
            card.className = 'chart-card';
            card.innerHTML = `
                <div class="chart-title">
                    <span>${{label}} <span title="${{tooltip}}" style="cursor:help;opacity:0.6">${{arrow}}</span></span>
                    <span class="chart-value">${{latestVal?.toFixed(3)}} ${{unit}}</span>
                </div>
                <div class="chart-container"><canvas></canvas></div>
            `;
            chartsContainer.appendChild(card);

            new Chart(card.querySelector('canvas'), {{
                type: 'line',
                data: {{
                    datasets: [{{
                        data: data,
                        borderColor: colors[key] || '#58a6ff',
                        backgroundColor: (colors[key] || '#58a6ff') + '20',
                        borderWidth: 2,
                        pointRadius: 4,
                        tension: 0.1,
                        fill: true
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: false }},
                        tooltip: {{
                            callbacks: {{
                                title: (items) => {{
                                    const d = items[0]?.raw;
                                    return d ? `${{d.name}} (${{d.commit}})` : '';
                                }},
                                label: (item) => {{
                                    const d = item.raw;
                                    return `${{label}}: ${{d.y?.toFixed(4)}} ${{unit}}`;
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            type: 'time',
                            time: {{ unit: 'day' }},
                            ticks: {{ maxTicksLimit: 5 }}
                        }},
                        y: {{ ticks: {{ maxTicksLimit: 5 }} }}
                    }}
                }}
            }});
        }});

        // Table
        const tbody = document.getElementById('table-body');
        [...runs].reverse().forEach(run => {{
            tbody.innerHTML += `
                <tr>
                    <td>${{new Date(run.created_at).toLocaleDateString()}}</td>
                    <td><code>${{run.commit}}</code></td>
                    <td><a href="${{run.url}}" target="_blank">${{run.name}}</a></td>
                    <td>${{(run.metrics.success_rate * 100).toFixed(1)}}%</td>
                    <td>${{run.metrics.mpkpe.toFixed(4)}}</td>
                    <td>${{run.metrics.ee_pos_error.toFixed(4)}}</td>
                </tr>
            `;
        }});
    </script>
</body>
</html>
"""


def load_cached_results(output_dir: Path) -> dict[str, dict]:
  """Load previously evaluated results from cache."""
  data_file = output_dir / "data.json"
  if not data_file.exists():
    return {}

  with open(data_file) as f:
    runs = json.load(f)

  return {run["id"]: run for run in runs}


def main(
  run_paths: list[str] | None = None,
  entity: str = "gcbc_researchers",
  project: str = "mjlab",
  tag: str = "nightly",
  limit: int = 30,
  num_envs: int = 1024,
  output_dir: Path = Path("benchmark_results"),
) -> None:
  """Generate benchmark report by evaluating nightly runs.

  Args:
    run_paths: Specific run paths to evaluate (entity/project/run_id).
    entity: WandB entity.
    project: WandB project name.
    tag: Filter runs by tag.
    limit: Maximum number of runs to evaluate.
    num_envs: Number of envs for evaluation.
    output_dir: Output directory for generated report.
  """
  # Load cached results to avoid re-evaluating old runs.
  cached = load_cached_results(output_dir)
  print(f"Loaded {len(cached)} cached evaluation results")

  eval_results = []

  if run_paths:
    for run_path in run_paths:
      run_id = run_path.split("/")[-1]
      if run_id in cached:
        print(f"Using cached result for {run_id}")
        eval_results.append(cached[run_id])
      else:
        result = evaluate_run(run_path, num_envs)
        eval_results.append(result)
  else:
    api = wandb.Api()
    print(f"Fetching runs from {entity}/{project} with tag '{tag}'...")
    runs = api.runs(f"{entity}/{project}", filters={"tags": tag}, order="-created_at")

    for i, run in enumerate(runs):
      if i >= limit:
        break
      if run.state != "finished":
        continue

      if run.id in cached:
        print(f"Using cached result for {run.name} ({run.id})")
        eval_results.append(cached[run.id])
      else:
        run_path = f"{entity}/{project}/{run.id}"
        result = evaluate_run(run_path, num_envs)
        eval_results.append(result)

  print(f"Total runs: {len(eval_results)}")
  generate_html_report(eval_results, output_dir)


if __name__ == "__main__":
  tyro.cli(main)

"""
HTML report generator for evaluation results.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List

from ..logger import get_logger

logger = get_logger()


class HTMLReporter:
    """Generate HTML reports for evaluation results."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_report(self, results: Dict[str, Any], filename: str = None) -> str:
        """
        Generate HTML report from evaluation results.

        Args:
            results: Dictionary containing evaluation results for all datasets
            filename: Optional filename for the report

        Returns:
            str: Path to the generated HTML file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_report_{timestamp}.html"

        filepath = os.path.join(self.output_dir, filename)

        html_content = self._generate_html_content(results)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"HTML report generated: {filepath}")
        return filepath

    def _generate_html_content(self, results: Dict[str, Any]) -> str:
        """Generate the HTML content for the report."""

        # Generate dataset cards
        dataset_cards = ""
        for dataset_name, dataset_data in results.items():
            dataset_cards += self._generate_dataset_card(dataset_name, dataset_data)

        # Generate summary statistics
        summary_stats = self._generate_summary_stats(results)

        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAGQA Evaluation Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        .header {{
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .summary-section {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .summary-item {{
            text-align: center;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }}
        
        .summary-item h3 {{
            color: #667eea;
            margin-bottom: 10px;
        }}
        
        .summary-item .value {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}
        
        .datasets-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 25px;
        }}
        
        .dataset-card {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            cursor: pointer;
        }}
        
        .dataset-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.15);
        }}
        
        .dataset-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }}
        
        .dataset-name {{
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
            margin-left: -8px;
        }}
        
        .dataset-type {{
            background: #667eea;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
        }}
        
        .scores-section {{
            margin-bottom: 20px;
        }}
        
        .score-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }}
        
        .score-item:last-child {{
            border-bottom: none;
        }}
        
        .score-label {{
            font-weight: 500;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .score-value {{
            font-weight: bold;
            font-size: 1.1em;
            color: #333;
        }}
        
        .score-value.high {{
            color: #28a745;
        }}
        
        .score-value.medium {{
            color: #ffc107;
        }}
        
        .score-value.low {{
            color: #dc3545;
        }}
        
        .error-section {{
            margin-top: 20px;
        }}
        
        .error-title {{
            font-weight: 500;
            color: #666;
            margin-bottom: 10px;
        }}
        
        .error-list {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            max-height: 180px;
            overflow-y: auto;
        }}
        
        /* Expand error list more when details is open */
        details[open] .error-list {{
            max-height: 480px;
        }}
        
        .error-detail {{
            white-space: pre-wrap;
            word-break: break-word;
            line-height: 1.4;
        }}
        
        .error-item {{
            background: #fff3cd;
            color: #856404;
            padding: 5px 10px;
            margin: 2px 0;
            border-radius: 4px;
            font-size: 0.9em;
            border-left: 3px solid #ffc107;
            cursor: pointer;
        }}
        
        .timestamp {{
            text-align: center;
            color: white;
            margin-top: 30px;
            opacity: 0.8;
        }}
        
        @media (max-width: 768px) {{
            .datasets-grid {{
                grid-template-columns: 1fr;
            }}
            
            .summary-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 RAGQA Evaluation Report</h1>
            <p>Comprehensive evaluation results for all datasets</p>
        </div>
        
        {summary_stats}
        
        <div class="datasets-grid">
            {dataset_cards}
        </div>
        
        <div class="timestamp">
            <p>Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
    </div>
    
    <script>
        // Add click functionality to dataset cards
        document.querySelectorAll('.dataset-card').forEach(card => {{
            card.addEventListener('click', function() {{
                // Add visual feedback
                this.style.transform = 'scale(0.98)';
                setTimeout(() => {{
                    this.style.transform = 'translateY(-5px)';
                }}, 150);
            }});
        }});
        
        // Add hover effects for score values
        document.querySelectorAll('.score-value').forEach(score => {{
            const value = parseFloat(score.textContent);
            if (!isNaN(value)) {{
                if (value >= 0.8) {{
                    score.classList.add('high');
                }} else if (value >= 0.6) {{
                    score.classList.add('medium');
                }} else {{
                    score.classList.add('low');
                }}
            }}
        }});
    </script>
</body>
</html>
        """

        return html_template

    def _generate_dataset_card(
        self, dataset_name: str, dataset_data: Dict[str, Any]
    ) -> str:
        """Generate HTML for a single dataset card."""

        # Generate scores HTML
        scores_html = ""
        if "scores" in dataset_data:
            preferred_order = ["acc", "f1", "em"]
            metrics_to_render = []
            scores_dict = dataset_data["scores"]

            # Build a lookup for case-insensitive match while preserving first-seen original key
            lower_to_entry = {}
            for metric, val in scores_dict.items():
                if val is None or val == "":
                    continue
                lower = str(metric).lower()
                if lower not in lower_to_entry:
                    try:
                        num_val = float(val)
                    except Exception:
                        continue
                    lower_to_entry[lower] = (metric, num_val)

            added_lowers = set()
            # Add preferred metrics first if present
            for key in preferred_order:
                if key in lower_to_entry:
                    orig_key, num_val = lower_to_entry[key]
                    metrics_to_render.append((orig_key, num_val))
                    added_lowers.add(key)

            # Add remaining metrics in original order (case-insensitive de-dup)
            for metric, val in scores_dict.items():
                lower = str(metric).lower()
                if lower in added_lowers:
                    continue
                if val is None or val == "":
                    continue
                try:
                    num_val = float(val)
                except Exception:
                    continue
                metrics_to_render.append((metric, num_val))
                added_lowers.add(lower)

            for metric, score in metrics_to_render:
                score_class = self._get_score_class(score)
                scores_html += f"""
                <div class=\"score-item\">\n                    <span class=\"score-label\">{str(metric).upper()}</span>\n                    <span class=\"score-value {score_class}\">{score:.3f}</span>\n                </div>
                """

        # Generate errors HTML
        errors_html = ""
        if "error_id" in dataset_data and dataset_data["error_id"]:
            error_items = ""
            # Build rich details lookup if provided
            details = dataset_data.get("error_details", {}) or {}
            # Show up to 10 errors
            for error_id in dataset_data["error_id"][:10]:
                safe_id = json.dumps(error_id)
                detail = details.get(error_id)
                if detail:
                    q = json.dumps(detail.get("query", ""))
                    gt = json.dumps(detail.get("ground_truth", ""))
                    pd = json.dumps(detail.get("prediction", ""))
                    error_items += (
                        f'<div class="error-item">'
                        f"{error_id}"
                        f'<div class="error-detail" style="display:none;margin-top:6px;background:#fff;border:1px solid #eee;border-radius:6px;padding:8px;">'
                        f'<div><strong>Query:</strong> <span class="err-q"></span></div>'
                        f'<div><strong>GT:</strong> <span class="err-gt"></span></div>'
                        f'<div><strong>Pred:</strong> <span class="err-pd"></span></div>'
                        f"</div>"
                        f"<script>(function(){{"
                        f"var parent = document.currentScript.parentElement; "
                        f"var wrap = parent.querySelector('.error-detail'); "
                        f"var q = {q}; var gt = {gt}; var pd = {pd}; "
                        f"parent.addEventListener('click', function(evt){{ "
                        f"  if(evt.target.closest('.error-detail')) return; "
                        f'  wrap.style.display = (wrap.style.display===\'none\' ? "block" : "none"); '
                        f"  wrap.querySelector('.err-q').textContent = q; "
                        f"  wrap.querySelector('.err-gt').textContent = gt; "
                        f"  wrap.querySelector('.err-pd').textContent = pd; "
                        f"}});"
                        f"}})();</script>"
                        f"</div>"
                    )
                else:
                    error_items += f'<div class="error-item">{error_id}</div>'

            if len(dataset_data["error_id"]) > 10:
                error_items += f'<div class="error-item">... and {len(dataset_data["error_id"]) - 10} more</div>'

            errors_html = f"""
            <details class="error-section">
                <summary class="error-title">Error IDs ({len(dataset_data["error_id"])}) - click an ID to view details</summary>
                <div class="error-list">
                    {error_items}
                </div>
            </details>
            """

        return f"""
        <div class="dataset-card">
            <div class="dataset-header">
                <div class="dataset-name">{dataset_name.upper()}</div>
                <div class="dataset-type">{dataset_data.get("class", "Unknown")}</div>
            </div>
            
            <div class="scores-section">
                {scores_html}
            </div>
            
            {errors_html}
        </div>
        """

    def _generate_summary_stats(self, results: Dict[str, Any]) -> str:
        """Generate summary statistics section."""

        total_datasets = len(results)
        total_errors = sum(len(data.get("error_id", [])) for data in results.values())

        # Calculate total samples (total number of samples across all datasets)
        total_samples = 0
        for data in results.values():
            # Try to get sample count from different possible fields
            if "total" in data:
                total_samples += data["total"]
            elif "sample_count" in data:
                total_samples += data["sample_count"]
            elif "scores" in data and any(data["scores"].values()):
                # If no explicit count, assume 1 sample per dataset
                total_samples += 1

        # Calculate average accuracy scores across all datasets (only acc, not f1/em)
        all_acc_scores = []
        for data in results.values():
            if "scores" in data:
                for metric, score in data["scores"].items():
                    if (
                        score is not None
                        and score != ""
                        and str(metric).lower() == "acc"
                    ):
                        all_acc_scores.append(float(score))

        avg_acc_score = (
            sum(all_acc_scores) / len(all_acc_scores) if all_acc_scores else 0
        )

        # Calculate average accuracy scores grouped by dataset type (class) - only acc, not f1/em
        type_to_acc_scores: Dict[str, List[float]] = {}
        for data in results.values():
            dataset_type = data.get("class", "Unknown")
            if "scores" in data:
                for metric, score in data["scores"].items():
                    if (
                        score is not None
                        and score != ""
                        and str(metric).lower() == "acc"
                    ):
                        type_to_acc_scores.setdefault(dataset_type, []).append(
                            float(score)
                        )

        type_avg_items = ""
        for dataset_type, acc_scores in sorted(
            type_to_acc_scores.items(), key=lambda x: x[0]
        ):
            type_avg = (sum(acc_scores) / len(acc_scores)) if acc_scores else 0
            type_avg_items += f"""
                <div class=\"summary-item\">
                    <h3>{dataset_type}</h3>
                    <div class=\"value\">{type_avg:.3f}</div>
                </div>
            """

        type_avg_section = (
            f"""
            <div style=\"margin-top: 20px;\">
                <h2>🧩 Average Accuracy Score by Type (Acc Only)</h2>
                <div class=\"summary-grid\">
                    {type_avg_items}
                </div>
            </div>
        """
            if type_avg_items
            else ""
        )

        # Build dataset-level accuracy averages per type for charting (only acc, not f1/em)
        type_to_dataset_avgs: Dict[str, List[Dict[str, Any]]] = {}
        for dataset_name, data in results.items():
            dataset_type = data.get("class", "Unknown")
            acc_score = None
            if "scores" in data:
                for metric, s in data["scores"].items():
                    if s is not None and s != "" and str(metric).lower() == "acc":
                        acc_score = float(s)
                        break
            if acc_score is not None:
                type_to_dataset_avgs.setdefault(dataset_type, []).append(
                    {"name": dataset_name, "score": round(acc_score, 4)}
                )

        # Compute per-type average based on dataset averages (for chart reference line)
        type_chart_avgs: Dict[str, float] = {}
        for t, items in type_to_dataset_avgs.items():
            if items:
                type_chart_avgs[t] = round(
                    sum(i["score"] for i in items) / len(items), 4
                )
            else:
                type_chart_avgs[t] = 0.0

        # Prepare per-type, per-dataset metric details for grouped bars (acc, f1, em)
        type_to_dataset_metrics: Dict[str, List[Dict[str, Any]]] = {}
        type_to_metric_order: Dict[str, List[str]] = {}
        preferred_metrics = ["acc", "f1", "em"]
        for dataset_name, data in results.items():
            dataset_type = data.get("class", "Unknown")
            metrics_map: Dict[str, float] = {}
            for m, v in (data.get("scores", {}) or {}).items():
                if v is None or v == "":
                    continue
                try:
                    metrics_map[str(m)] = float(v)
                except Exception:
                    continue
            if metrics_map:
                type_to_dataset_metrics.setdefault(dataset_type, []).append(
                    {"name": dataset_name, "metrics": metrics_map}
                )
        # Metric order per type: preferred first (if present), then remaining alpha
        for t, items in type_to_dataset_metrics.items():
            present_keys = set()
            for it in items:
                present_keys.update(k for k in it.get("metrics", {}).keys())
            ordered = [
                m
                for m in preferred_metrics
                if m in {k.lower(): k for k in present_keys}.keys()
            ]
            # Map lower->original to preserve case of first occurrence
            lower_to_orig = {}
            for it in items:
                for k in it.get("metrics", {}).keys():
                    lk = k.lower()
                    if lk not in lower_to_orig:
                        lower_to_orig[lk] = k
            # Build ordered list with preserved original keys
            ordered_preserved = []
            for m in preferred_metrics:
                for lk, ok in lower_to_orig.items():
                    if lk == m and ok not in ordered_preserved:
                        ordered_preserved.append(ok)
            remaining = sorted(
                [ok for lk, ok in lower_to_orig.items() if lk not in preferred_metrics],
                key=lambda x: x.lower(),
            )
            type_to_metric_order[t] = ordered_preserved + remaining

        # Compute per-type per-metric averages (keys stored in lowercase for easy lookup)
        type_metric_avg_map: Dict[str, Dict[str, float]] = {}
        for t, items in type_to_dataset_metrics.items():
            sums: Dict[str, float] = {}
            counts: Dict[str, int] = {}
            for it in items:
                for m, v in (it.get("metrics", {}) or {}).items():
                    lk = str(m).lower()
                    sums[lk] = sums.get(lk, 0.0) + float(v)
                    counts[lk] = counts.get(lk, 0) + 1
            type_metric_avg_map[t] = {
                mk: (sums[mk] / counts[mk])
                for mk in sums.keys()
                if counts.get(mk, 0) > 0
            }

        # Prepare chart HTML + inline script
        if type_to_dataset_avgs:
            import json as _json

            type_chart_data_json = _json.dumps(type_to_dataset_avgs)
            type_chart_avg_json = _json.dumps(type_chart_avgs)
            type_order = list(sorted(type_to_dataset_avgs.keys()))
            type_order_json = _json.dumps(type_order)
            chart_metrics_json = _json.dumps(type_to_dataset_metrics)
            chart_metric_order_json = _json.dumps(type_to_metric_order)
            type_metric_avg_json = _json.dumps(type_metric_avg_map)
            type_tabs_html = "".join(
                [
                    f'<button class="type-tab" data-type="{t}" style="padding:8px 12px;margin:6px;border:1px solid #667eea;border-radius:16px;background:#fff;color:#667eea;cursor:pointer;">{t}</button>'
                    for t in type_order
                ]
            )
            type_chart_section = f"""
            <div class=\"type-chart-section\" style=\"margin-top: 24px;\">
                <h2>📈 Scores by Type</h2>
                <div class=\"type-tabs\" style=\"margin: 10px 0;\">
                    {type_tabs_html}
                </div>
                <div class=\"chart-container\" style=\"width: 100%; background: #fff; border-radius: 10px; padding: 12px; border: 1px solid #eee;\">
                    <div id=\"type-legend\" class=\"type-legend\" style=\"display:flex;flex-wrap:wrap;gap:10px;margin-bottom:8px;align-items:center;\"></div>
                    <svg id=\"type-bar-chart\" width=\"100%\" height=\"420\"></svg>
                </div>
                <script>
                (function() {{
                    const data = {type_chart_data_json};
                    const typeAverages = {type_chart_avg_json};
                    const typeOrder = {type_order_json};
                    const chartMetrics = {chart_metrics_json};
                    const typeMetricOrder = {chart_metric_order_json};
                    const typeMetricAverages = {type_metric_avg_json};
                    const container = document.querySelector('.type-chart-section');
                    const svg = container.querySelector('#type-bar-chart');
                    const tabs = container.querySelectorAll('.type-tab');
                    const legendDiv = container.querySelector('#type-legend');

                    function setActiveTab(activeType) {{
                        tabs.forEach(btn => {{
                            if (btn.dataset.type === activeType) {{
                                btn.style.background = '#667eea';
                                btn.style.color = '#fff';
                            }} else {{
                                btn.style.background = '#fff';
                                btn.style.color = '#667eea';
                            }}
                        }});
                    }}

                    function renderLegend() {{
                        if (!legendDiv) return;
                        legendDiv.innerHTML = '';
                        const defs = [
                            {{ color: '#28a745', label: '>= AVG' }},
                            {{ color: '#dc3545', label: '< AVG' }},
                        ];
                        defs.forEach(def => {{
                            const item = document.createElement('div');
                            item.style.display = 'flex';
                            item.style.alignItems = 'center';
                            item.style.gap = '6px';
                            const sw = document.createElement('span');
                            sw.style.display = 'inline-block';
                            sw.style.width = '12px';
                            sw.style.height = '12px';
                            sw.style.background = def.color;
                            sw.style.borderRadius = '2px';
                            const label = document.createElement('span');
                            label.style.fontSize = '12px';
                            label.style.color = '#495057';
                            label.textContent = def.label;
                            item.appendChild(sw);
                            item.appendChild(label);
                            legendDiv.appendChild(item);
                        }});
                    }}

                    function renderTypeChart(typeKey) {{
                        const items = (chartMetrics[typeKey] || []).slice();
                        const allOrder = typeMetricOrder[typeKey] || [];
                        const pref = ['acc','f1','em'];
                        // Metrics on X-axis
                        const metricsAxis = pref.map(k => (allOrder.find(m => String(m).toLowerCase() === k))).filter(Boolean);
                        const finalMetrics = metricsAxis.length ? metricsAxis : allOrder;
                        const datasets = items.map(d => d.name);

                        const W = svg.clientWidth || svg.getBoundingClientRect().width || 900;
                        const left = 60, right = 20, top = 20, bottom = 110;
                        const H = 420;
                        svg.setAttribute('height', H);
                        const innerW = Math.max(200, W - left - right);
                        const innerH = Math.max(100, H - top - bottom);

                        // Clear
                        while (svg.firstChild) svg.removeChild(svg.firstChild);

                        // Y grid lines 0, 0.5, 1.0
                        [0, 0.5, 1].forEach(tick => {{
                            const y = top + innerH * (1 - tick);
                            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                            line.setAttribute('x1', left);
                            line.setAttribute('y1', y);
                            line.setAttribute('x2', left + innerW);
                            line.setAttribute('y2', y);
                            line.setAttribute('stroke', '#e9ecef');
                            line.setAttribute('stroke-width', '1');
                            svg.appendChild(line);
                            const lbl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                            lbl.setAttribute('x', left - 8);
                            lbl.setAttribute('y', y + 4);
                            lbl.setAttribute('text-anchor', 'end');
                            lbl.setAttribute('font-size', '11');
                            lbl.setAttribute('fill', '#6c757d');
                            lbl.textContent = String(tick.toFixed(1));
                            svg.appendChild(lbl);
                        }});

                        // Compute layout per metric group
                        const groupCount = Math.max(1, finalMetrics.length);
                        const groupPad = 24;
                        const groupW = Math.max(40, (innerW - groupPad * (groupCount - 1)) / groupCount);
                        const numDatasets = Math.max(1, datasets.length);
                        const innerGap = 4;
                        const barW = Math.max(6, (groupW - innerGap * (numDatasets - 1)) / numDatasets);

                        // Legend: threshold meaning
                        renderLegend();

                        // Draw bars
                        finalMetrics.forEach((metric, gi) => {{
                            const groupX = left + gi * (groupW + groupPad);
                            const avgMap = typeMetricAverages[typeKey] || {{}};
                            const metricAvg = avgMap[String(metric).toLowerCase()] || 0;

                            // Metric label under group
                            const mLbl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                            mLbl.setAttribute('x', groupX + groupW / 2);
                            mLbl.setAttribute('y', top + innerH + 20);
                            mLbl.setAttribute('text-anchor', 'middle');
                            mLbl.setAttribute('font-size', '12');
                            mLbl.setAttribute('fill', '#495057');
                            mLbl.textContent = String(metric).toUpperCase();
                            svg.appendChild(mLbl);

                            // Average line for this metric
                            const yAvg = top + innerH * (1 - Math.max(0, Math.min(1, metricAvg)));
                            const mLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                            mLine.setAttribute('x1', groupX);
                            mLine.setAttribute('y1', yAvg);
                            mLine.setAttribute('x2', groupX + groupW);
                            mLine.setAttribute('y2', yAvg);
                            mLine.setAttribute('stroke', '#6c757d');
                            mLine.setAttribute('stroke-dasharray', '4,3');
                            mLine.setAttribute('stroke-width', '1.5');
                            svg.appendChild(mLine);

                            // Bars per dataset
                            datasets.forEach((ds, j) => {{
                                const item = items[j];
                                const val = (item && item.metrics && Object.prototype.hasOwnProperty.call(item.metrics, metric)) ? item.metrics[metric] : null;
                                if (val == null) return;
                                const v = Math.max(0, Math.min(1, val));
                                const x = groupX + j * (barW + innerGap);
                                const y = top + innerH * (1 - v);
                                const h = innerH * v;
                                const color = v >= metricAvg ? '#28a745' : '#dc3545';
                                const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                                rect.setAttribute('x', x);
                                rect.setAttribute('y', y);
                                rect.setAttribute('width', barW);
                                rect.setAttribute('height', Math.max(1, h));
                                rect.setAttribute('fill', color);
                                rect.setAttribute('opacity', '0.9');
                                svg.appendChild(rect);

                                // Dataset label (rotated) under x-axis, once per group for clarity
                                const lbl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                                const lx = x + barW / 2;
                                const ly = top + innerH + 36;
                                lbl.setAttribute('x', lx);
                                lbl.setAttribute('y', ly);
                                lbl.setAttribute('text-anchor', 'end');
                                lbl.setAttribute('font-size', '15');
                                lbl.setAttribute('fill', '#6c757d');
                                lbl.textContent = ds;
                                lbl.setAttribute('transform', 'rotate(-45 ' + lx + ' ' + ly + ')');
                                svg.appendChild(lbl);
                            }});
                        }});
                    }}
                 
                    // Bind events
                    tabs.forEach(btn => {{
                        btn.addEventListener('click', () => {{
                            const t = btn.dataset.type;
                            setActiveTab(t);
                            renderTypeChart(t);
                        }});
                    }});
                 
                    // Initial render
                    if (typeOrder && typeOrder.length) {{
                        setActiveTab(typeOrder[0]);
                        renderTypeChart(typeOrder[0]);
                    }}
                }})();
                </script>
            </div>
            """
        else:
            type_chart_section = ""

        return f"""
        <div class=\"summary-section\">\n            <h2>📊 Summary Statistics</h2>\n            <div class=\"summary-grid\">\n                <div class=\"summary-item\">\n                    <h3>Total Datasets</h3>\n                    <div class=\"value\">{total_datasets}</div>\n                </div>\n                <div class=\"summary-item\">\n                    <h3>Total Samples</h3>\n                    <div class=\"value\">{total_samples}</div>\n                </div>\n                <div class=\"summary-item\">\n                    <h3>Total Errors</h3>\n                    <div class=\"value\">{total_errors}</div>\n                </div>\n                <div class=\"summary-item\">\n                    <h3>Average Accuracy Score (Acc Only)</h3>\n                    <div class=\"value\">{avg_acc_score:.3f}</div>\n                </div>\n            </div>\n            {type_avg_section}\n            {type_chart_section}\n        </div>\n        """

    def _get_score_class(self, score: float) -> str:
        """Get CSS class for score styling."""
        if score >= 0.8:
            return "high"
        elif score >= 0.6:
            return "medium"
        else:
            return "low"

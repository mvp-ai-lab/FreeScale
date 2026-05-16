import json
import pandas as pd
import os
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import re


def extract_keyword_from_path(experiment_path):
    """Extract keyword from experiment path using common patterns."""
    experiment_name = os.path.basename(experiment_path)

    # Common patterns for extracting keywords
    patterns = [
        r'(.+)_(keyword\d+)$',  # test1_keyword1
        r'(.+)_([a-zA-Z]+\d+)$',  # test1_param1
        r'(.+)-([a-zA-Z]+\d+)$',  # test1-param1
        r'(.+)-([a-zA-Z]+_\d+)$',  # test1-param_1
    ]

    for pattern in patterns:
        match = re.match(pattern, experiment_name)
        if match:
            base_name, keyword = match.groups()
            return keyword, base_name

    # If no pattern matches, try to split by common separators
    for separator in ['_', '-']:
        if separator in experiment_name:
            parts = experiment_name.split(separator)
            if len(parts) >= 2:
                keyword = parts[-1]
                base_name = separator.join(parts[:-1])
                return keyword, base_name

    # If no keyword found, use the entire name as base and no keyword
    return "default", experiment_name


def extract_last_iteration_metrics(json_path):
    """Extract metrics from the last iteration in the metrics JSON file."""
    try:
        with open(json_path, 'r') as f:
            content = f.read().strip()
            if '}{' in content:
                json_objects = content.split('}{')
                json_objects = [json_objects[0] + '}'] + [
                    '{' + obj + '}' for obj in json_objects[1:-1]] + ['{' + json_objects[-1]]
            else:
                json_objects = [content]

            all_iterations = []
            for json_str in json_objects:
                try:
                    data = json.loads(json_str)
                    if "0" in data:
                        iteration_data = data["0"]
                        all_iterations.append(iteration_data)
                    else:
                        all_iterations.append(data)
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse JSON object: {e}")
                    continue

            if not all_iterations:
                print(f"Warning: No valid iterations found in {json_path}")
                return {}

            last_iteration = all_iterations[-1]

            desired_metrics = {}
            for metric in ['psnr', 'ssim', 'lpips', 'time', 'memory', 'points']:
                if metric in last_iteration:
                    desired_metrics[f'eval_{metric}'] = last_iteration[metric]

            if 'iteration' in last_iteration:
                desired_metrics['eval_iteration'] = last_iteration['iteration']

            return desired_metrics

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not read {json_path}: {e}")
        return {}
    
import json

def extract_first_iteration_metrics(json_path):
    """Extract metrics from the first iteration in the metrics JSON file."""
    try:
        with open(json_path, 'r') as f:
            content = f.read().strip()
            if '}{' in content:
                json_objects = content.split('}{')
                json_objects = [json_objects[0] + '}'] + [
                    '{' + obj + '}' for obj in json_objects[1:-1]] + ['{' + json_objects[-1]]
            else:
                json_objects = [content]

            for json_str in json_objects:
                try:
                    data = json.loads(json_str)
                    
                    if "0" in data:
                        first_iteration = data["0"]
                    else:
                        first_iteration = data["global"]
                    desired_metrics = {}
                    for metric in ['psnr', 'ssim', 'lpips', 'time', 'memory', 'points']:
                        if metric in first_iteration:
                            desired_metrics[f'eval_{metric}'] = first_iteration[metric]

                    if 'iteration' in first_iteration:
                        desired_metrics['eval_iteration'] = first_iteration['iteration']
                    return desired_metrics

                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse JSON object: {e}")
                    continue

            print(f"Warning: No valid iterations found in {json_path}")
            return {}

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not read {json_path}: {e}")
        return {}


def find_metrics_json(experiment_path):
    """Find metrics.json in the experiment directory structure."""
    possible_paths = [
        os.path.join(experiment_path, "eval", "val", "metrics.json"),
        os.path.join(experiment_path, "metrics.json"),
        os.path.join(experiment_path, "metrics_denoise.json"),
        os.path.join(experiment_path, "evaluation", "metrics.json"),
        os.path.join(experiment_path, "results", "metrics.json"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    for root, dirs, files in os.walk(experiment_path):
        if "metrics.json" in files:
            return os.path.join(root, "metrics.json")

    return None


def extract_training_metrics(train_json_path):
    """Extract training metrics from train_metrics.json."""
    try:
        with open(train_json_path, 'r') as f:
            train_data = json.load(f)

        train_metrics = {}

        field_mapping = {
            "Start Date": "start_date",
            "End Date": "end_date",
            "Training Time": "training_time",
            "Peak GPU Memory": "peak_gpu_memory"
        }

        for json_key, metric_key in field_mapping.items():
            if json_key in train_data:
                train_metrics[f'train_{metric_key}'] = train_data[json_key]

        if "Start Date" in train_data and "End Date" in train_data:
            try:
                from datetime import datetime
                start = datetime.strptime(
                    train_data["Start Date"], "%Y-%m-%d %H:%M:%S.%f")
                end = datetime.strptime(
                    train_data["End Date"], "%Y-%m-%d %H:%M:%S.%f")
                total_duration = (end - start).total_seconds()
                train_metrics['train_total_duration_seconds'] = total_duration
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not parse dates: {e}")

        return train_metrics

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(
            f"Warning: Could not read training metrics from {train_json_path}: {e}")
        return {}


def collect_experiment_metrics(experiment_path):
    """Collect metrics from a single experiment."""
    experiment_name = os.path.basename(experiment_path)
    keyword, base_name = extract_keyword_from_path(experiment_path)

    metrics = {
        'experiment_name': experiment_name,
        'experiment_path': experiment_path,
        'keyword': keyword,
        'base_name': base_name
    }

    # Extract training metrics
    train_json_path = os.path.join(experiment_path, "train_metrics.json")
    if os.path.exists(train_json_path):
        train_metrics = extract_training_metrics(train_json_path)
        if train_metrics:
            metrics.update(train_metrics)

    # Extract evaluation metrics
    metrics_json_path = find_metrics_json(experiment_path)
    if metrics_json_path:
        eval_metrics = extract_last_iteration_metrics(metrics_json_path)
        # eval_metrics = extract_first_iteration_metrics(metrics_json_path)
        if eval_metrics:
            metrics.update(eval_metrics)
        metrics['eval_path'] = metrics_json_path
    else:
        metrics['eval_path'] = 'Not found'

    return metrics


def collect_all_experiments(root_dir):
    """Collect metrics from all experiments in the root directory."""
    all_metrics = []
    root_path = Path(root_dir)

    if not root_path.exists():
        print(f"Error: Root directory {root_dir} does not exist")
        return all_metrics

    experiments = [d for d in root_path.iterdir() if d.is_dir()]

    print(f"Found {len(experiments)} experiments to process...")

    for i, exp_path in enumerate(experiments):
        print(f"Processing {i+1}/{len(experiments)}: {exp_path.name}")

        metrics = collect_experiment_metrics(str(exp_path))
        all_metrics.append(metrics)

    return all_metrics


def compute_keyword_statistics(metrics_data):
    """Compute averaged metrics for each keyword group."""
    keyword_groups = defaultdict(list)

    # Group metrics by keyword
    for metrics in metrics_data:
        keyword = metrics.get('keyword', 'default')
        keyword_groups[keyword].append(metrics)

    # Compute statistics for each keyword group
    keyword_stats = {}
    for keyword, experiments in keyword_groups.items():
        stats = {
            'keyword': keyword,
            'experiment_count': len(experiments),
            'experiment_names': [exp['experiment_name'] for exp in experiments]
        }

        # Define metrics to average
        metrics_to_average = [
            'eval_psnr', 'eval_ssim', 'eval_lpips', 'eval_points',
            'eval_time', 'train_training_time', 'train_peak_gpu_memory'
        ]

        for metric in metrics_to_average:
            values = [exp.get(metric)
                      for exp in experiments if exp.get(metric) is not None]
            if values:
                stats[f'{metric}_mean'] = np.mean(values)
                stats[f'{metric}_std'] = np.std(values)
                stats[f'{metric}_min'] = np.min(values)
                stats[f'{metric}_max'] = np.max(values)

        keyword_stats[keyword] = stats

    return keyword_stats


def generate_comparison_charts(keyword_stats, output_dir):
    """Generate line charts comparing different metrics across keywords."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define chart configurations
    chart_configs = [
        {
            'metric': 'eval_psnr',
            'title': 'PSNR Comparison by Parameter',
            'ylabel': 'PSNR (higher is better)',
            'invert': False
        },
        {
            'metric': 'eval_ssim',
            'title': 'SSIM Comparison by Parameter',
            'ylabel': 'SSIM (higher is better)',
            'invert': False
        },
        {
            'metric': 'eval_lpips',
            'title': 'LPIPS Comparison by Parameter',
            'ylabel': 'LPIPS (lower is better)',
            'invert': True
        },
        {
            'metric': 'eval_points',
            'title': 'Number of Points Comparison by Parameter',
            'ylabel': 'Number of Points',
            'invert': False
        },
        {
            'metric': 'eval_time',
            'title': 'Evaluation Time Comparison by Parameter',
            'ylabel': 'Evaluation Time (seconds)',
            'invert': True
        },
        {
            'metric': 'train_training_time',
            'title': 'Training Time Comparison by Parameter',
            'ylabel': 'Training Time (seconds)',
            'invert': True
        },
        {
            'metric': 'train_peak_gpu_memory',
            'title': 'Peak GPU Memory Comparison by Parameter',
            'ylabel': 'GPU Memory (GB)',
            'invert': True
        }
    ]

    # Prepare data for plotting
    keywords = list(keyword_stats.keys())

    for config in chart_configs:
        metric = config['metric']

        # Check if we have data for this metric
        means = []
        stds = []
        valid_keywords = []

        for keyword in keywords:
            mean_key = f'{metric}_mean'
            std_key = f'{metric}_std'

            if mean_key in keyword_stats[keyword]:
                means.append(keyword_stats[keyword][mean_key])
                stds.append(keyword_stats[keyword][std_key])
                valid_keywords.append(keyword)

        if not means:
            print(f"Warning: No data found for metric {metric}")
            continue

        # Create the chart
        plt.figure(figsize=(12, 8))

        # Create positions for bars
        x_pos = np.arange(len(valid_keywords))

        # Plot with error bars
        if config['invert']:
            # For metrics where lower is better, we might want to show the inverse
            plt.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7,
                    color=['red', 'blue', 'green', 'orange', 'purple'][:len(valid_keywords)])
        else:
            plt.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7,
                    color=['blue', 'orange', 'green', 'red', 'purple'][:len(valid_keywords)])

        plt.xlabel('Parameters')
        plt.ylabel(config['ylabel'])
        plt.title(config['title'])
        plt.xticks(x_pos, valid_keywords, rotation=45)

        # Add value labels on top of bars
        for i, v in enumerate(means):
            plt.text(i, v + (0.02 * max(means)), f'{v:.3f}',
                     ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        # Save the chart
        chart_filename = f"{metric}_comparison.png"
        chart_path = os.path.join(output_dir, chart_filename)
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Generated chart: {chart_filename}")


def save_to_excel(metrics_data, keyword_stats, output_file):
    """Save collected metrics and statistics to an Excel file."""
    if not metrics_data:
        print("No metrics data to save.")
        return

    # Create main DataFrame
    df = pd.DataFrame(metrics_data)

    # Define column order
    basic_columns = ['experiment_name', 'base_name',
                     'keyword', 'experiment_path', 'eval_path']
    key_eval_metrics = ['eval_psnr', 'eval_ssim', 'eval_lpips']
    other_eval_metrics = [col for col in df.columns if col.startswith(
        'eval_') and col not in basic_columns + key_eval_metrics]
    train_columns = [col for col in df.columns if col.startswith('train_')]
    other_columns = [col for col in df.columns if col not in basic_columns +
                     key_eval_metrics + other_eval_metrics + train_columns]

    # Sort and filter columns
    key_eval_metrics.sort()
    other_eval_metrics.sort()
    train_columns.sort()
    other_columns.sort()

    final_columns = basic_columns + key_eval_metrics + \
        other_eval_metrics + train_columns + other_columns
    final_columns = [col for col in final_columns if col in df.columns]

    df = df[final_columns]

    # Create keyword statistics DataFrame
    stats_data = []
    for keyword, stats in keyword_stats.items():
        stats_row = {'keyword': keyword,
                     'experiment_count': stats['experiment_count']}
        for key, value in stats.items():
            if key not in ['keyword', 'experiment_count', 'experiment_names']:
                stats_row[key] = value
        stats_data.append(stats_row)

    stats_df = pd.DataFrame(stats_data)

    # Save to Excel with multiple sheets
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # All experiments data - sort by keyword for better organization
        df_sorted = df.sort_values(['keyword', 'base_name'])
        df_sorted.to_excel(writer, sheet_name='All_Experiments', index=False)

        # Keyword statistics
        if not stats_df.empty:
            stats_df.to_excel(
                writer, sheet_name='Keyword_Statistics', index=False)

        # Grouped by keyword
        # for keyword in keyword_stats.keys():
        #     keyword_experiments = [
        #         exp for exp in metrics_data if exp.get('keyword') == keyword]
        #     if keyword_experiments:
        #         keyword_df = pd.DataFrame(keyword_experiments)[final_columns]
        #         sheet_name = f"Keyword_{keyword}"[
        #             :31]  # Excel sheet name limit
        #         keyword_df.to_excel(writer, sheet_name=sheet_name, index=False)

        # Overview sheet with key metrics - grouped by keyword in consecutive rows
        overview_cols = ['experiment_name', 'base_name', 'keyword',
                         'eval_psnr', 'eval_ssim', 'eval_lpips', 'eval_points']
        overview_cols.extend([col for col in [
                             'train_training_time', 'train_peak_gpu_memory'] if col in df.columns])
        overview_cols = [col for col in overview_cols if col in df.columns]

        if len(overview_cols) > 1:
            # Create overview DataFrame and sort by keyword to group same parameters together
            overview_df = df[overview_cols].sort_values(
                ['keyword', 'base_name'])
            overview_df.to_excel(writer, sheet_name='Overview', index=False)

            # Add a summary row for each keyword group
            summary_rows = []
            for keyword in overview_df['keyword'].unique():
                keyword_data = overview_df[overview_df['keyword'] == keyword]

                # Create summary row
                summary_row = {
                    'experiment_name': f'SUMMARY - {keyword}', 'base_name': '', 'keyword': keyword}

                # Calculate averages for numeric columns
                numeric_cols = [col for col in overview_cols if col not in [
                    'experiment_name', 'base_name', 'keyword']]
                for col in numeric_cols:
                    if col in keyword_data.columns and \
                       pd.api.types.is_numeric_dtype(keyword_data[col]):
                        summary_row[col] = keyword_data[col].mean()
                    else:
                        summary_row[col] = ''

                summary_rows.append(summary_row)

            # Create summary DataFrame
            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)

                # Combine original data with summary rows
                combined_overview = pd.concat(
                    [overview_df, summary_df], ignore_index=True)

                # Write the combined data to a new sheet
                combined_overview.to_excel(
                    writer, sheet_name='Overview_With_Summary', index=False)

    print(f"Metrics saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract and compare 3DGS experiment metrics by parameters')
    parser.add_argument('--root_dir', type=str, default='/home/user/exp',
                        help='Root directory containing experiments')
    parser.add_argument('--output_dir', type=str, default='3dgs_metrics.xlsx',
                        help='Output Excel file name')

    args = parser.parse_args()

    assert os.path.exists(
        args.root_dir), f"root_dir {args.root_dir} not exists!"

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    print(f"Starting metrics extraction from: {args.root_dir}")
    print("Grouping experiments by keywords and generating comparison charts...")

    # Collect all metrics
    all_metrics = collect_all_experiments(args.root_dir)

    if all_metrics:
        # Compute keyword statistics
        excel_path = os.path.join(args.output_dir, "3dgs_metrics.xlsx")
        keyword_stats = compute_keyword_statistics(all_metrics)

        # Print summary
        print("\n=== EXPERIMENT GROUPS SUMMARY ===")
        for keyword, stats in keyword_stats.items():
            print(
                f"Parameter '{keyword}': {stats['experiment_count']} experiments")
            if 'eval_psnr_mean' in stats:
                print(
                    f"  PSNR: {stats['eval_psnr_mean']:.3f} ± {stats['eval_psnr_std']:.3f}")
            if 'eval_ssim_mean' in stats:
                print(
                    f"  SSIM: {stats['eval_ssim_mean']:.3f} ± {stats['eval_ssim_std']:.3f}")
            if 'eval_lpips_mean' in stats:
                print(
                    f"  LPIPS: {stats['eval_lpips_mean']:.3f} ± {stats['eval_lpips_std']:.3f}")

        # Save to Excel
        save_to_excel(all_metrics, keyword_stats, excel_path)
        exit()

        # Generate comparison charts
        print("\nGenerating comparison charts...")
        generate_comparison_charts(keyword_stats, args.output_dir)

        print(f"\nSuccessfully processed {len(all_metrics)} experiments")
        print(f"Results saved to: {args.output_dir}")
        print(f"Charts saved to: {args.output_dir}")

    else:
        print("No experiment metrics were collected.")


if __name__ == "__main__":
    main()

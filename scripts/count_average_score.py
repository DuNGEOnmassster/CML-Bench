import os
import sys
import csv
import glob
import argparse
from pathlib import Path
from collections import defaultdict

META_COLS = {"movie_name", "imdb_id", "line_number"}

def find_csv_files(inputs):
    """Takes a list of files or directories as input, returns a list of all csv file paths"""
    csv_files = []
    for inp in inputs:
        p = Path(inp)
        if p.is_file() and p.suffix == ".csv":
            csv_files.append(str(p.resolve()))
        elif p.is_dir():
            for f in p.glob("*.csv"):
                csv_files.append(str(f.resolve()))
    return csv_files

def count_average_per_file(csv_files):
    """Calculate average scores for each metric per CSV file, return a list of dicts containing filename and metric averages"""
    results = []
    all_metrics = set()
    for csv_path in csv_files:
        sum_dict = defaultdict(float)
        count_dict = defaultdict(int)
        header = None
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            for row in reader:
                for k, v in row.items():
                    if k in META_COLS:
                        continue
                    try:
                        val = float(v)
                        sum_dict[k] += val
                        count_dict[k] += 1
                        all_metrics.add(k)
                    except Exception:
                        continue
        avg_dict = {k: (sum_dict[k] / count_dict[k] if count_dict[k] else 0.0) for k in sum_dict}
        avg_dict['file'] = os.path.basename(csv_path)
        results.append(avg_dict)
    return results, sorted(list(all_metrics))

def main():
    parser = argparse.ArgumentParser(description="Calculate average scores for each metric per CSV file, return a list of dicts containing filename and metric averages")
    parser.add_argument('inputs', nargs='+', help='csv file or directory containing csv files, supports multiple')
    parser.add_argument('--output', '-o', default='average_score.csv', help='output statistics csv file name')
    args = parser.parse_args()
    csv_files = find_csv_files(args.inputs)
    if not csv_files:
        print("No csv files found!")
        sys.exit(1)
    results, metrics = count_average_per_file(csv_files)
    out_cols = ['file'] + metrics
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(out_cols)
        for avg_dict in results:
            row = [avg_dict.get(c, 0.0) for c in out_cols]
            writer.writerow(row)
    print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main() 
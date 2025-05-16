import json
import csv
import os
import argparse
from collections import defaultdict

def load_imdbid_to_genres(info_json_path):
    imdbid_to_genres = {}
    with open(info_json_path, 'r', encoding='utf-8') as f:
        info = json.load(f)
        for item in info['individual_results']:
            imdb_id = item['imdb_id']
            genres = item['genres']
            imdbid_to_genres[imdb_id] = genres
    return imdbid_to_genres

def group_rows_by_genre(csv_path, imdbid_to_genres):
    genre_to_rows = defaultdict(list)
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        for row in reader:
            imdb_id = row['imdb_id']
            genres = imdbid_to_genres.get(imdb_id, [])
            for genre in genres:
                genre_to_rows[genre].append(row)
    return genre_to_rows, header

def save_genre_csvs(genre_to_rows, header, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for genre, rows in genre_to_rows.items():
        out_path = os.path.join(output_dir, f"{genre}.csv")
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {out_path} ({len(rows)} rows)")

def main():
    parser = argparse.ArgumentParser(description="Filter CSV by genre and output to separate files")
    parser.add_argument('--info_json', default='CML-Bench/gt_100_info.json')
    # parser.add_argument('--csv', default='/data/nas/mingzhe/code-release/MovieLLM/scripts/output_benchmark_v5_gt.csv')
    parser.add_argument('--csv', default='scripts/results/gt/output_benchmark_v4_gt_100.csv')
    parser.add_argument('--output_dir', default='scripts/genres_csv_0515', help='Output directory')
    args = parser.parse_args()

    imdbid_to_genres = load_imdbid_to_genres(args.info_json)
    genre_to_rows, header = group_rows_by_genre(args.csv, imdbid_to_genres)
    save_genre_csvs(genre_to_rows, header, args.output_dir)

if __name__ == '__main__':
    main()
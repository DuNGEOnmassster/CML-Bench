# CML-Bench

## How to run

### Install the dependencies

```bash
conda create -n cml-bench python=3.10
conda activate cml-bench
pip install -r requirements.txt
```

### Download the data

```bash
sh download_data.sh
```

### Run the benchmark

```bash
sh run_benchmark_v5_base2.sh [GPU_IDs]
```

Example:

```bash
sh run_benchmark_v5_base2.sh 0,1,2,3
```

If no GPU_IDs are provided, CUDA_VISIBLE_DEVICES will not be explicitly set, and PyTorch/Transformers will use its default behavior.

```bash
sh run_benchmark_v5_base2.sh
```

### Count the average score

```bash
python scripts/count_average_score.py <csv_file_or_directory_1> <csv_file_or_directory_2> ... -o <output_csv_file>
```

Example:

```bash
python scripts/count_average_score.py scripts/results/gt/output_benchmark_v4_gt_100.csv /data/nas/mingzhe/code-release/MovieLLM/CML-Bench/scripts/results/base -o scripts/results/average_score_gt_base.csv
```



### Draw the ablation score

```bash
python scripts/draw_ablation_score.py
```

### Draw the genres score

First select the genres you want to draw, secure you have the `gt_100_info.json` in the data directory.

```bash
python scripts/select_genres.py
```

By default, the script will save the selected genres from `scripts/results/gt/output_benchmark_v4_gt_100.csv` to `scripts/genres_csv_0515`.

Then draw the plot/boxplot/mean bar chart.

```bash
python scripts/draw_genres_error_bar.py
```

By default, the script will save the plot/boxplot/mean bar chart to `scripts/genres_charts_0515`.
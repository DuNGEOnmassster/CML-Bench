import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Metric groups
DC_METRICS = ["dc1_adjacent_similarity", "dc2_qa_relevance", "dc3_topic_concentration"]
CC_METRICS = ["cc1_emotional_stability", "cc2_linguistic_consistency", "cc3_action_intention_alignment"]
PR_METRICS = ["pr1_scene_similarity", "pr2_event_coherence"]

sns.set(style="whitegrid", palette="Set2", font_scale=1.2)


def load_genre_scores(genre_csv_dir, metrics):
    genre_to_scores = {}
    for csv_file in glob.glob(os.path.join(genre_csv_dir, "*.csv")):
        genre = os.path.splitext(os.path.basename(csv_file))[0]
        df = pd.read_csv(csv_file)
        # Only keep needed metrics
        scores = {metric: df[metric].dropna().astype(float).tolist() for metric in metrics if metric in df.columns}
        genre_to_scores[genre] = scores
    return genre_to_scores


def plot_boxplot_with_errorbar(genre_to_scores, metrics, group_name, out_path):
    plt.figure(figsize=(14, 7))
    genres = sorted(genre_to_scores.keys())
    n_metrics = len(metrics)
    n_genres = len(genres)
    # Prepare DataFrame for seaborn
    records = []
    for genre in genres:
        for metric in metrics:
            for score in genre_to_scores[genre].get(metric, []):
                records.append({"Genre": genre, "Metric": metric, "Score": score})
    df = pd.DataFrame(records)
    # Draw boxplot
    ax = sns.boxplot(
        data=df,
        x="Metric",
        y="Score",
        hue="Genre",
        palette="Set2",
        showmeans=True,
        meanprops={"marker":"o","markerfacecolor":"white","markeredgecolor":"black","markersize":"7"}
    )
    # Draw error bars (mean ± std)
    for i, metric in enumerate(metrics):
        for j, genre in enumerate(genres):
            scores = df[(df["Metric"] == metric) & (df["Genre"] == genre)]["Score"]
            if len(scores) == 0:
                continue
            mean = scores.mean()
            std = scores.std()
            ax.errorbar(i - 0.3 + j * 0.6 / (n_genres-1 if n_genres>1 else 1), mean, yerr=std, fmt='o', color='black', capsize=4, alpha=0.7)
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.title(f"{group_name} metrics by genre (boxplot with error bars)")
    plt.legend(title="Genre", bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

# 新增：画每个genre在该group下所有metric的均值和std的bar+errorbar图
def plot_bar_mean_with_errorbar(genre_to_scores, metrics, group_name, out_path):
    genres = sorted(genre_to_scores.keys())
    means = []
    stds = []
    for genre in genres:
        # 合并所有metric的score
        all_scores = []
        for metric in metrics:
            all_scores.extend(genre_to_scores[genre].get(metric, []))
        if all_scores:
            means.append(pd.Series(all_scores).mean())
            stds.append(pd.Series(all_scores).std())
        else:
            means.append(0)
            stds.append(0)
    plt.figure(figsize=(14, 6))
    bar = plt.bar(genres, means, yerr=stds, capsize=6, color=sns.color_palette("Set2", len(genres)))
    plt.xlabel("Genre")
    plt.ylabel("Mean Score (all metrics)")
    plt.title(f"{group_name} mean score by genre (all metrics, with error bars)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

# Add: Draw boxplot of mean scores for each genre in the group
def plot_boxplot_mean_by_genre(genre_to_scores, metrics, group_name, out_path):
    genres = sorted(genre_to_scores.keys())
    # Collect mean scores for each movie (each row in csv represents one movie)
    records = []
    for genre in genres:
        # Find the mean score for each movie (row) across all metrics for this genre
        # First locate the CSV file for this genre
        csv_path = None
        for f in glob.glob(os.path.join(args.input_dir, "*.csv")):
            if os.path.splitext(os.path.basename(f))[0] == genre:
                csv_path = f
                break
        if csv_path is None:
            continue
        df = pd.read_csv(csv_path)
        # Only keep metrics columns
        subdf = df[[m for m in metrics if m in df.columns]]
        # Calculate mean score for each row (movie)
        for mean_score in subdf.mean(axis=1, skipna=True):
            records.append({"Genre": genre, "MeanScore": mean_score})
    mean_df = pd.DataFrame(records)
    plt.figure(figsize=(14, 7))
    ax = sns.boxplot(
        data=mean_df,
        x="Genre",
        y="MeanScore",
        palette="Set2"
    )
    # Mean points
    sns.stripplot(
        data=mean_df,
        x="Genre",
        y="MeanScore",
        color="black",
        size=4,
        jitter=True,
        alpha=0.7,
        ax=ax
    )
    plt.xlabel("Genre")
    plt.ylabel("Mean Score (all metrics)")
    plt.title(f"{group_name} mean score by genre (boxplot)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Draw boxplots with error bars for DC/CC/PR metrics by genre.")
    parser.add_argument('--input_dir', default='./scripts/genres_csv_0515', help='Input csv directory')
    parser.add_argument('--output_dir', default='./scripts/genres_charts_0515', help='Output image directory')
    global args
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for metrics, group_name in zip([DC_METRICS, CC_METRICS, PR_METRICS], ["DC", "CC", "PR"]):
        genre_to_scores = load_genre_scores(args.input_dir, metrics)
        out_path1 = os.path.join(args.output_dir, f"{group_name}_genres_boxplot.png")
        plot_boxplot_with_errorbar(genre_to_scores, metrics, group_name, out_path1)
        # Add: Mean bar chart
        out_path2 = os.path.join(args.output_dir, f"{group_name}_genres_bar_mean.png")
        plot_bar_mean_with_errorbar(genre_to_scores, metrics, group_name, out_path2)
        # Add: Mean boxplot
        out_path3 = os.path.join(args.output_dir, f"{group_name}_genres_boxplot_mean.png")
        plot_boxplot_mean_by_genre(genre_to_scores, metrics, group_name, out_path3)

if __name__ == '__main__':
    main()

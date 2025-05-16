import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import numpy as np

# Ablation types
ABLATION_TYPES = ["base", "rag", "instruction"]
# Metric groups
CC_METRICS = ["cc1_emotional_stability", "cc2_linguistic_consistency", "cc3_action_intention_alignment"]
DC_METRICS = ["dc1_adjacent_similarity", "dc2_qa_relevance", "dc3_topic_concentration"]
PR_METRICS = ["pr1_scene_similarity", "pr2_event_coherence"]

MODEL_PATTERN = re.compile(r"^(.*?)_(base|rag|instruction)\.csv$")

# Updated color and marker scheme
MODEL_BASE_COLORS_CMAP = plt.get_cmap('tab10')  # Use plt.get_cmap for compatibility
BASE_MARKER = 'o'
RAG_MARKER = 's'
INSTRUCTION_MARKER = '^'

DEFAULT_SCATTER_SIZE = 120
HOLLOW_MARKER_LINEWIDTH = 1.5

def load_ablation_data(csv_path):
    df = pd.read_csv(csv_path)
    model_dict = {}
    for _, row in df.iterrows():
        fname = row["file"]
        m = MODEL_PATTERN.match(fname)
        if not m:
            continue
        model_key, ablation = m.group(1), m.group(2)
        if model_key not in model_dict:
            model_dict[model_key] = {}
        if ablation in ABLATION_TYPES:
            model_dict[model_key][ablation] = row
    return model_dict

def plot_group(model_data_full, metrics, group_name, out_path, include_base=True):
    model_names_present = sorted([
        model for model, ablations in model_data_full.items()
        if any(ab_type in ablations for ab_type in ABLATION_TYPES)
    ])

    if not model_names_present:
        print(f"No valid model data found for group {group_name}, skipping plot.")
        return

    num_models = len(model_names_present)
    num_metrics = len(metrics)

    fig_width = max(12, num_metrics * num_models * 0.3)
    plt.figure(figsize=(fig_width, 7))

    x_metric_indices = np.arange(num_metrics)
    model_base_colors = [MODEL_BASE_COLORS_CMAP(i / float(max(1, num_models - 1))) for i in range(num_models)]

    group_width_on_x = 0.6 
    model_offsets = np.linspace(-group_width_on_x / 2, group_width_on_x / 2, num_models if num_models > 1 else 1)
    if num_models == 1: model_offsets = [0]

    # Collect scores for dynamic y-axis scaling if not including base
    scores_for_y_scaling = []

    for i, metric_name in enumerate(metrics):
        for j, model_name in enumerate(model_names_present):
            model_ablations = model_data_full.get(model_name, {})
            model_color = model_base_colors[j]
            current_x_base = x_metric_indices[i] + model_offsets[j]

            base_score = float(model_ablations.get("base", {}).get(metric_name, np.nan))
            rag_score = float(model_ablations.get("rag", {}).get(metric_name, np.nan))
            instruction_score = float(model_ablations.get("instruction", {}).get(metric_name, np.nan))
            
            if not include_base:
                if not np.isnan(rag_score): scores_for_y_scaling.append(rag_score)
                if not np.isnan(instruction_score): scores_for_y_scaling.append(instruction_score)
            else: # For include_base=True, consider all scores for the default 0-1.05 range
                if not np.isnan(base_score): scores_for_y_scaling.append(base_score)
                if not np.isnan(rag_score): scores_for_y_scaling.append(rag_score)
                if not np.isnan(instruction_score): scores_for_y_scaling.append(instruction_score)

            if include_base and not np.isnan(base_score):
                plt.scatter(current_x_base, base_score, facecolors='none', edgecolors=model_color, 
                            marker=BASE_MARKER, s=DEFAULT_SCATTER_SIZE, linewidths=HOLLOW_MARKER_LINEWIDTH, alpha=0.9)
            if not np.isnan(rag_score):
                plt.scatter(current_x_base, rag_score, facecolors='none', edgecolors=model_color, 
                            marker=RAG_MARKER, s=DEFAULT_SCATTER_SIZE, linewidths=HOLLOW_MARKER_LINEWIDTH, alpha=0.9)
            if not np.isnan(instruction_score):
                plt.scatter(current_x_base, instruction_score, facecolors='none', edgecolors=model_color, 
                            marker=INSTRUCTION_MARKER, s=DEFAULT_SCATTER_SIZE, linewidths=HOLLOW_MARKER_LINEWIDTH, alpha=0.9)

    plt.xlabel("Metrics")
    plt.ylabel("Score")
    title_suffix = "(Scatter Plot)" if include_base else "(RAG & Instruction Only, Scatter)"
    plt.title(f"{group_name} Scores by Model and Ablation {title_suffix}")
    plt.xticks(x_metric_indices, metrics, rotation=25, ha="right")
    
    if not include_base and scores_for_y_scaling:
        min_score = min(scores_for_y_scaling)
        max_score = max(scores_for_y_scaling)
        padding = (max_score - min_score) * 0.10 # 10% padding
        if padding < 0.02: padding = 0.02 # Ensure some minimal padding
        if max_score == min_score : # if all scores are the same
             y_lower = max(0, min_score - 0.1) 
             y_upper = min(1.05, max_score + 0.1)
        else:
            y_lower = max(0, min_score - padding)
            y_upper = min(1.05, max_score + padding)
        if y_upper <= y_lower : # handle cases where scores are too close or at limits
            y_upper = y_lower + 0.1 if y_lower < 0.95 else 1.05
            y_lower = y_upper -0.1 if y_upper > 0.1 else 0.0
        plt.ylim(y_lower, y_upper)
    else:
        plt.ylim(0, 1.05) # Default for plots including base or if no scores

    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    model_legend_handles = []
    simplified_model_names = []
    for j, model_name_full in enumerate(model_names_present):
        name_part = model_name_full.split('-')[0]
        if name_part == "internlm3" and "instruct" in model_name_full:
            simplified_name = "internlm3-instruct"
        elif name_part == "meta_llama" and "instruct" in model_name_full:
            sub_parts = model_name_full.split('-')
            simplified_name = f"llama-{sub_parts[2]}" if len(sub_parts) > 2 else "llama"
        elif name_part.startswith(("o3", "o4")):
            simplified_name = name_part
        else:
            simplified_name = name_part
        model_legend_handles.append(plt.Line2D([0], [0], color=model_base_colors[j], lw=HOLLOW_MARKER_LINEWIDTH+1, 
                                               linestyle='-', markersize=0, label=simplified_name))
 
    ablation_legend_items = []
    if include_base:
        ablation_legend_items.append(plt.Line2D([0], [0], marker=BASE_MARKER, color='grey', linestyle='None', 
                                                markersize=8, markerfacecolor='none', markeredgecolor='grey', 
                                                markeredgewidth=HOLLOW_MARKER_LINEWIDTH, label='Base'))
    ablation_legend_items.extend([
        plt.Line2D([0], [0], marker=RAG_MARKER, color='grey', linestyle='None', 
                   markersize=8, markerfacecolor='none', markeredgecolor='grey',
                   markeredgewidth=HOLLOW_MARKER_LINEWIDTH, label='RAG'),
        plt.Line2D([0], [0], marker=INSTRUCTION_MARKER, color='grey', linestyle='None', 
                   markersize=8, markerfacecolor='none', markeredgecolor='grey',
                   markeredgewidth=HOLLOW_MARKER_LINEWIDTH, label='Instruction')
    ])

    leg1 = plt.legend(handles=model_legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1), title="Models")
    plt.gca().add_artist(leg1)
    plt.legend(handles=ablation_legend_items, loc='lower left', bbox_to_anchor=(1.02, 0), title="Ablations")

    plt.tight_layout(rect=[0, 0, 0.82, 1]) 
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "average_score.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: average_score.csv not found at {csv_path}")
        return

    model_data = load_ablation_data(csv_path)
    
    if not model_data:
        print("Failed to load model data. Check average_score.csv and MODEL_PATTERN.")
        return

    output_dir = os.path.join(script_dir, "ablation_scatter_charts")
    os.makedirs(output_dir, exist_ok=True)

    # Plots including Base
    plot_group(model_data, CC_METRICS, "CC", os.path.join(output_dir, "cc_ablation_scatter_all.png"), include_base=True)
    plot_group(model_data, DC_METRICS, "DC", os.path.join(output_dir, "dc_ablation_scatter_all.png"), include_base=True)
    plot_group(model_data, PR_METRICS, "PR", os.path.join(output_dir, "pr_ablation_scatter_all.png"), include_base=True)
    print(f"Ablation scatter plots (all) saved to: {output_dir}")

    # Plots for RAG and Instruction only
    plot_group(model_data, CC_METRICS, "CC", os.path.join(output_dir, "cc_ablation_scatter_rag_instruction_only.png"), include_base=False)
    plot_group(model_data, DC_METRICS, "DC", os.path.join(output_dir, "dc_ablation_scatter_rag_instruction_only.png"), include_base=False)
    plot_group(model_data, PR_METRICS, "PR", os.path.join(output_dir, "pr_ablation_scatter_rag_instruction_only.png"), include_base=False)
    print(f"Ablation scatter plots (RAG & Instruction only) saved to: {output_dir}")

if __name__ == "__main__":
    main() 
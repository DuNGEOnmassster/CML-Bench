from scipy.stats import spearmanr

CML_Scores_DC = [0.87, 0.00, 0.03, 0.55, 0.56, 0.31, 0.06, 0.19, 0.86, 0.89, 0.88, 0.89, 0.90, 0.88, 0.90]
Human_Ratings_DC = [3.00, 1.67, 1.46, 1.42, 1.13, 1.46, 1.29, 1.25, 3.50, 3.58, 3.50, 3.42, 3.50, 3.42, 3.17]

CML_Scores_CC = [0.84, 0.00, 0.02, 0.43, 0.43, 0.14, 0.01, 0.12, 0.76, 0.82, 0.81, 0.83, 0.82, 0.82, 0.77]
Human_Ratings_CC = [2.66, 1.21, 1.33, 1.29, 1.04, 1.33, 1.21, 1.67, 3.33, 3.58, 3.42, 3.42, 3.42, 3.50, 3.25]

CML_Scores_PR = [0.91, 0.45, 0.46, 0.58, 0.55, 0.75, 0.51, 0.60, 0.93, 0.93, 0.93, 0.92, 0.92, 0.92, 0.90]
Human_Ratings_PR = [3.33, 1.21, 1.21, 1.25, 1.42, 1.29, 1.25, 1.21, 3.25, 3.33, 3.50, 3.50, 3.58, 3.50, 3.42]

# Spearman correlation for DC
corr_dc, p_value_dc = spearmanr(CML_Scores_DC, Human_Ratings_DC)
print(f"Spearman correlation for DC: {corr_dc:.2f}")

# Spearman correlation for CC
corr_cc, p_value_cc = spearmanr(CML_Scores_CC, Human_Ratings_CC)
print(f"Spearman correlation for CC: {corr_cc:.2f}")

# Spearman correlation for PR
corr_pr, p_value_pr = spearmanr(CML_Scores_PR, Human_Ratings_PR)
print(f"Spearman correlation for PR: {corr_pr:.2f}")

# Calculate Overall Average Scores
Overall_CML_Scores = []
Overall_Human_Ratings = []
num_models = len(CML_Scores_DC)

for i in range(num_models):
    avg_cml = (CML_Scores_DC[i] + CML_Scores_CC[i] + CML_Scores_PR[i]) / 3.0
    avg_human = (Human_Ratings_DC[i] + Human_Ratings_CC[i] + Human_Ratings_PR[i]) / 3.0
    Overall_CML_Scores.append(avg_cml)
    Overall_Human_Ratings.append(avg_human)

# Spearman correlation for Overall Average Scores
corr_overall, p_value_overall = spearmanr(Overall_CML_Scores, Overall_Human_Ratings)
print(f"Overall Spearman correlation (CML Avg vs Human Avg): {corr_overall:.2f} (p-value: {p_value_overall:.4f})")
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from scipy.stats import chi2, probplot, ttest_ind
from scipy.spatial.distance import pdist, squareform
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.multivariate.manova import MANOVA
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
from skbio.stats.distance import permanova, DistanceMatrix
from itertools import combinations

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
# Define inputs and output directory
input_base = r"NEW_RUNS_LMForge_RUN04_Lab_Machine\Generated_Results\Phi-4-mini-instruct"
file_path = f"{input_base}/scores.csv"

output_dir = "Analysis_Results"
os.makedirs(output_dir, exist_ok=True)

# Open a text file to log console outputs
log_file_path = os.path.join(output_dir, "statistical_summary.txt")
log_file = open(log_file_path, "w")

def log_print(title, content=""):
    """Helper function to print to console and write to log file."""
    output = f"\n{'-'*50}\n{title}\n{'-'*50}\n{content}\n"
    print(output)
    log_file.write(output)

if os.path.exists(file_path):
    print(f"Loading data from: {file_path}")
else:
    raise FileNotFoundError(f"File {file_path} does not exist. Please check the path.")

scores_df = pd.read_csv(file_path)
scores_df = scores_df.drop_duplicates(subset=["chunk_size", "max_tokens"], keep="first").drop(columns=["elapsed_time", "questions_num"])
scores_df["group"] = scores_df["chunk_size"].astype(str)

dv_columns = [
    "rouge1", "rouge2", "rougeL", "rougeLsum",
    "bert_score_P", "bert_score_R", "bert_score_F1", "sts_score"
]

# ==========================================
# 2. OUTLIER DETECTION & NORMALITY
# ==========================================
X = scores_df[dv_columns].dropna()

# Compute robust Mahalanobis distances
robust_cov = MinCovDet().fit(X)
mahalanobis_distances = robust_cov.mahalanobis(X)

# Chi-squared Q-Q plot
plt.figure()
probplot(mahalanobis_distances, dist="chi2", sparams=(len(dv_columns),), plot=plt)
plt.title("Q-Q Plot of Mahalanobis Distances")
plt.savefig(os.path.join(output_dir, "qq_plot_mahalanobis.png"), bbox_inches='tight')
plt.close()

# Formal multivariate normality test
normality_test = pg.multivariate_normality(scores_df[dv_columns].dropna(), alpha=0.05)
log_print("Henze-Zirkler Multivariate Normality Test", normality_test)

# Calculate thresholds and find outliers
threshold_99 = chi2.ppf(0.99, df=len(dv_columns))
threshold_999 = chi2.ppf(0.999, df=len(dv_columns))

outliers = mahalanobis_distances > threshold_99
outlier_flags = mahalanobis_distances > threshold_999

outlier_indices = np.where(outliers)[0]
outlier_rows = scores_df.iloc[outlier_indices]
outlier_rows.to_csv(os.path.join(output_dir, "outliers_detected.csv"), index=False)

outlier_summary = pd.DataFrame([{
    "Total Observations": len(mahalanobis_distances),
    "Outliers Detected (p < 0.001)": np.sum(outlier_flags),
    "Percentage Outliers": 100 * np.sum(outlier_flags) / len(mahalanobis_distances)
}])
outlier_summary.to_csv(os.path.join(output_dir, "outlier_summary.csv"), index=False)
log_print("Outlier Summary", outlier_summary.to_string())

# ==========================================
# 3. HOMOGENEITY, CORRELATION & MULTICOLLINEARITY
# ==========================================
df_clean = scores_df.dropna(subset=dv_columns + ["group"]).copy()

# Box's M test
box_m = pg.box_m(data=df_clean, dvs=dv_columns, group='group')
log_print("Box's M Test for Homogeneity of Covariance", box_m)

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(scores_df[dv_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation between DVs")
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), bbox_inches='tight')
plt.close()

# VIF for Multicollinearity
X_with_const = add_constant(X)
vif_data = pd.DataFrame({
    "feature": X_with_const.columns,
    "VIF": [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
})
vif_data.to_csv(os.path.join(output_dir, "vif_multicollinearity.csv"), index=False)

# ==========================================
# 4. MANOVA & ANOVA
# ==========================================
formula_simple = " + ".join(dv_columns) + " ~ group"
maov_simple = MANOVA.from_formula(formula_simple, data=df_clean)
log_print("MANOVA Results", maov_simple.mv_test())

# Type III ANOVA for each DV
anova_frames = []
for column in dv_columns:
    formula = f"{column} ~ group"
    model = ols(formula, data=df_clean).fit()
    anova_res = anova_lm(model, typ=3)
    anova_res['DV'] = column # Tag with Dependent Variable name
    anova_frames.append(anova_res)

anova_combined = pd.concat(anova_frames)
anova_combined.to_csv(os.path.join(output_dir, "anova_type3_results.csv"))

# ==========================================
# 5. PERMANOVA & PCA
# ==========================================
X_clean_vals = df_clean[dv_columns].values
distance_matrix = DistanceMatrix(squareform(pdist(X_clean_vals, metric='euclidean')))
perm_chunk = permanova(distance_matrix, grouping=df_clean["group"].values, permutations=999)
log_print("PERMANOVA (Chunk Size)", perm_chunk)

# PCA + MANOVA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[dv_columns])
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_clean["PC1"] = X_pca[:, 0]
df_clean["PC2"] = X_pca[:, 1]

maov_pca = MANOVA.from_formula("PC1 + PC2 ~ group", data=df_clean)
log_print("MANOVA on PCA Components", maov_pca.mv_test())

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_clean, x="PC1", y="PC2", hue="group", style="group")
plt.title("PCA of Dependent Variables by Chunk Size")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Chunk Size")
plt.savefig(os.path.join(output_dir, "pca_scatter.png"), bbox_inches='tight')
plt.close()

# ==========================================
# 6. VISUALIZATIONS (F1 Scores)
# ==========================================
plt.figure(figsize=(10, 6))
sns.boxplot(x='chunk_size', y='bert_score_F1', hue='max_tokens', data=scores_df, palette="Set2")
plt.xlabel("Chunk Size")
plt.ylabel("BERTScore F1")
plt.xticks(rotation=45)
plt.legend(title="Max Tokens")
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "boxplot_bert_f1.png"))
plt.close()

plt.figure(figsize=(12, 6))
sns.lineplot(data=scores_df, x='chunk_size', y='bert_score_F1', hue='max_tokens', marker='o')
plt.xlabel("Chunk Size")
plt.ylabel("BERTScore F1")
plt.title("BERTScore F1 by Chunk Size and Max Tokens")
plt.xticks(rotation=45)
plt.legend(title="Max Tokens")
plt.grid()
plt.savefig(os.path.join(output_dir, "lineplot_bert_f1.png"), bbox_inches='tight')
plt.close()

# ==========================================
# 7. PAIRWISE TESTING & MEAN DIFFERENCES
# ==========================================
# Pairwise PERMANOVA
pairwise_results = []
for grp1, grp2 in combinations(scores_df["group"].dropna().unique(), 2):
    subset = scores_df[scores_df["group"].isin([grp1, grp2])].copy().reset_index(drop=True)
    X_pair = subset[dv_columns].values
    dist_mat = DistanceMatrix(squareform(pdist(X_pair, metric='euclidean')), ids=[str(i) for i in subset.index])
    grouping_df = pd.DataFrame({'group': subset["group"].values}, index=[str(i) for i in subset.index])
    
    result = permanova(dist_mat, grouping=grouping_df, column='group', permutations=999)
    pairwise_results.append({
        "Group 1": grp1, "Group 2": grp2,
        "Pseudo-F": result["test statistic"], "p-value": result["p-value"]
    })

pd.DataFrame(pairwise_results).to_csv(os.path.join(output_dir, "pairwise_permanova.csv"), index=False)

# Mean Differences per DV
group_means = scores_df.groupby('chunk_size')[dv_columns].mean().sort_index()
group_means.to_csv(os.path.join(output_dir, "group_means.csv"))

mean_diffs = []
chunk_sizes = group_means.index.tolist()
for dv in dv_columns:
    for grp1, grp2 in combinations(chunk_sizes, 2):
        mean_diffs.append({
            "Dependent Variable": dv,
            "Group 1": grp1, "Group 2": grp2,
            "Mean Difference": group_means.loc[grp2, dv] - group_means.loc[grp1, dv]
        })
pd.DataFrame(mean_diffs).to_csv(os.path.join(output_dir, "mean_differences.csv"), index=False)

# Pairwise T-Tests with FDR correction
ttest_results = []
for dv in dv_columns:
    for grp1, grp2 in combinations(scores_df["group"].dropna().unique(), 2):
        vals1 = scores_df[scores_df["group"] == grp1][dv].dropna()
        vals2 = scores_df[scores_df["group"] == grp2][dv].dropna()
        t_stat, p_val = ttest_ind(vals1, vals2, equal_var=False)
        ttest_results.append({
            "Outcome": dv, "Group 1": grp1, "Group 2": grp2,
            "t-statistic": t_stat, "raw p-value": p_val
        })

ttest_df = pd.DataFrame(ttest_results)
reject, fdr_pvals, _, _ = multipletests(ttest_df["raw p-value"].fillna(1), method='fdr_bh')
ttest_df["FDR (BH) p-value"] = fdr_pvals
ttest_df["Significant (FDR < 0.05)"] = reject
ttest_df.to_csv(os.path.join(output_dir, "pairwise_ttests_fdr.csv"), index=False)

# Close text log
log_file.close()
print(f"\nAll analysis complete! Outputs, plots, and CSVs have been saved to the '{output_dir}' folder.")
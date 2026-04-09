"""
Cross-model MANOVA / PERMANOVA: stack scores from multiple decoder models and use
`model` as the grouping factor (analogous to `chunk_size` in the per-model script).

Expected layout (one LMForge run per model on the lab machine):

  Codes_Outputs/NEW_RUNS_LMForge_RUN01_Lab_Machine/Generated_Results/gemma-7b-spark/scores.csv
  Codes_Outputs/NEW_RUNS_LMForge_RUN02_Lab_Machine/Generated_Results/Qwen2.5-7B-Instruct/scores.csv
  Codes_Outputs/NEW_RUNS_LMForge_RUN03_Lab_Machine/Generated_Results/Mistral-7B-Instruct-v0.3/scores.csv
  Codes_Outputs/NEW_RUNS_LMForge_RUN04_Lab_Machine/Generated_Results/Phi-4-mini-instruct/scores.csv

Edit `MODEL_RUN_SOURCES` if run numbers or folder names change. Each `scores.csv` must share
the same metric columns as the per-model pipeline.
"""
import os
import re
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
# 1. SETUP & DATA LOADING (multi-model)
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
CODES_OUTPUTS = os.path.join(PROJECT_ROOT, "Codes_Outputs")

# (run folder under Codes_Outputs, model subfolder under that run's Generated_Results)
MODEL_RUN_SOURCES = [
    ("NEW_RUNS_LMForge_RUN01_Lab_Machine", "gemma-7b-spark"),
    ("NEW_RUNS_LMForge_RUN02_Lab_Machine", "Qwen2.5-7B-Instruct"),
    ("NEW_RUNS_LMForge_RUN03_Lab_Machine", "Mistral-7B-Instruct-v0.3"),
    ("NEW_RUNS_LMForge_RUN04_Lab_Machine", "Phi-4-mini-instruct"),
]

# Optional: set to a single (chunk_size, max_tokens) tuple to compare models only
# in that experimental cell (e.g. (512, 512)). None = use all rows (pooled).
filter_chunk_max_tokens = None

output_dir = os.path.join(SCRIPT_DIR, "Cross_Model_Analysis_Results")
os.makedirs(output_dir, exist_ok=True)

log_file_path = os.path.join(output_dir, "cross_model_statistical_summary.txt")
log_file = open(log_file_path, "w", encoding="utf-8")


def log_print(title, content=""):
    output = f"\n{'-' * 50}\n{title}\n{'-' * 50}\n{content}\n"
    print(output)
    log_file.write(output)


def sanitize_model_key(name: str) -> str:
    """Safe factor level names for formula / CSV (avoid patsy parsing issues)."""
    return re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_") or "model"


def load_scores_from_run_sources(sources: list, codes_outputs: str) -> pd.DataFrame:
    frames = []
    loaded_paths = []
    for run_folder, model_subfolder in sources:
        path = os.path.join(
            codes_outputs,
            run_folder,
            "Generated_Results",
            model_subfolder,
            "scores.csv",
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing scores.csv for {model_subfolder} ({run_folder}):\n  {path}"
            )
        df = pd.read_csv(path)
        df["model"] = model_subfolder
        df["model_key"] = sanitize_model_key(model_subfolder)
        df["lmforge_run"] = run_folder
        frames.append(df)
        loaded_paths.append(path)

    if len(frames) < 2:
        raise RuntimeError(
            "Need at least 2 models in MODEL_RUN_SOURCES; found "
            f"{len(frames)}. Check paths under {codes_outputs}."
        )

    log_print(
        "Loaded scores (cross-run)",
        "\n".join(f"  {p}" for p in loaded_paths),
    )
    return pd.concat(frames, ignore_index=True)


scores_df = load_scores_from_run_sources(MODEL_RUN_SOURCES, CODES_OUTPUTS)

drop_cols = [c for c in ["elapsed_time", "questions_num"] if c in scores_df.columns]
scores_df = scores_df.drop_duplicates(subset=["model", "chunk_size", "max_tokens"], keep="first")
scores_df = scores_df.drop(columns=drop_cols, errors="ignore")

if filter_chunk_max_tokens is not None:
    cs, mt = filter_chunk_max_tokens
    scores_df = scores_df[
        (scores_df["chunk_size"] == cs) & (scores_df["max_tokens"] == mt)
    ].copy()
    log_print(
        "Filter applied",
        f"chunk_size={cs}, max_tokens={mt}; n rows = {len(scores_df)}",
    )

# Grouping factor for cross-model tests
scores_df["group"] = scores_df["model_key"]

dv_columns = [
    "rouge1",
    "rouge2",
    "rougeL",
    "rougeLsum",
    "bert_score_P",
    "bert_score_R",
    "bert_score_F1",
    "sts_score",
]

missing = [c for c in dv_columns if c not in scores_df.columns]
if missing:
    raise ValueError(f"scores.csv missing expected columns: {missing}")

# ==========================================
# 2. OUTLIER DETECTION & NORMALITY
# ==========================================
X = scores_df[dv_columns].dropna()
robust_cov = MinCovDet().fit(X)
mahalanobis_distances = robust_cov.mahalanobis(X)

plt.figure()
probplot(mahalanobis_distances, dist="chi2", sparams=(len(dv_columns),), plot=plt)
plt.title("Q-Q Plot of Mahalanobis Distances (cross-model pooled)")
plt.savefig(os.path.join(output_dir, "qq_plot_mahalanobis.png"), bbox_inches="tight")
plt.close()

normality_test = pg.multivariate_normality(scores_df[dv_columns].dropna(), alpha=0.05)
log_print("Henze-Zirkler Multivariate Normality Test", normality_test)

threshold_99 = chi2.ppf(0.99, df=len(dv_columns))
threshold_999 = chi2.ppf(0.999, df=len(dv_columns))
outliers = mahalanobis_distances > threshold_99
outlier_flags = mahalanobis_distances > threshold_999
outlier_indices = np.where(outliers)[0]
outlier_rows = scores_df.iloc[outlier_indices]
outlier_rows.to_csv(os.path.join(output_dir, "outliers_detected.csv"), index=False)

outlier_summary = pd.DataFrame(
    [
        {
            "Total Observations": len(mahalanobis_distances),
            "Outliers Detected (p < 0.001)": int(np.sum(outlier_flags)),
            "Percentage Outliers": 100 * np.sum(outlier_flags) / len(mahalanobis_distances),
        }
    ]
)
outlier_summary.to_csv(os.path.join(output_dir, "outlier_summary.csv"), index=False)
log_print("Outlier Summary", outlier_summary.to_string())

# ==========================================
# 3. HOMOGENEITY, CORRELATION & MULTICOLLINEARITY
# ==========================================
df_clean = scores_df.dropna(subset=dv_columns + ["group"]).copy()

box_m = pg.box_m(data=df_clean, dvs=dv_columns, group="group")
log_print("Box's M Test for Homogeneity of Covariance (by model)", box_m)

plt.figure(figsize=(10, 8))
sns.heatmap(scores_df[dv_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation between DVs (all models pooled)")
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), bbox_inches="tight")
plt.close()

X_with_const = add_constant(X)
vif_data = pd.DataFrame(
    {
        "feature": X_with_const.columns,
        "VIF": [
            variance_inflation_factor(X_with_const.values, i)
            for i in range(X_with_const.shape[1])
        ],
    }
)
vif_data.to_csv(os.path.join(output_dir, "vif_multicollinearity.csv"), index=False)

# ==========================================
# 4. MANOVA & ANOVA (group = model)
# ==========================================
formula_simple = " + ".join(dv_columns) + " ~ C(group)"
maov_simple = MANOVA.from_formula(formula_simple, data=df_clean)
log_print("MANOVA Results (multivariate effect of model)", maov_simple.mv_test())

anova_frames = []
for column in dv_columns:
    formula = f"{column} ~ C(group)"
    model = ols(formula, data=df_clean).fit()
    anova_res = anova_lm(model, typ=3)
    anova_res["DV"] = column
    anova_frames.append(anova_res)

anova_combined = pd.concat(anova_frames)
anova_combined.to_csv(os.path.join(output_dir, "anova_type3_results.csv"))

# ==========================================
# 5. PERMANOVA & PCA (group = model)
# ==========================================
X_clean_vals = df_clean[dv_columns].values
distance_matrix = DistanceMatrix(squareform(pdist(X_clean_vals, metric="euclidean")))
perm_model = permanova(
    distance_matrix, grouping=df_clean["group"].values, permutations=999
)
log_print("PERMANOVA (model as grouping factor)", perm_model)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[dv_columns])
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_clean["PC1"] = X_pca[:, 0]
df_clean["PC2"] = X_pca[:, 1]

maov_pca = MANOVA.from_formula("PC1 + PC2 ~ C(group)", data=df_clean)
log_print("MANOVA on PCA Components (by model)", maov_pca.mv_test())

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_clean, x="PC1", y="PC2", hue="model", style="model")
plt.title("PCA of DVs by model")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pca_scatter.png"), bbox_inches="tight")
plt.close()

# ==========================================
# 6. VISUALIZATIONS
# ==========================================
plt.figure(figsize=(12, 6))
sns.boxplot(data=scores_df, x="model", y="bert_score_F1", palette="Set2")
plt.xlabel("Model")
plt.ylabel("BERTScore F1")
plt.xticks(rotation=25, ha="right")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "boxplot_bert_f1_by_model.png"))
plt.close()

# ==========================================
# 7. PAIRWISE TESTING & MEAN DIFFERENCES
# ==========================================
pairwise_results = []
for grp1, grp2 in combinations(df_clean["group"].dropna().unique(), 2):
    subset = df_clean[df_clean["group"].isin([grp1, grp2])].copy().reset_index(drop=True)
    X_pair = subset[dv_columns].values
    dist_mat = DistanceMatrix(
        squareform(pdist(X_pair, metric="euclidean")),
        ids=[str(i) for i in subset.index],
    )
    grouping_df = pd.DataFrame({"group": subset["group"].values}, index=[str(i) for i in subset.index])
    result = permanova(dist_mat, grouping=grouping_df, column="group", permutations=999)
    pairwise_results.append(
        {
            "Model key 1": grp1,
            "Model key 2": grp2,
            "Pseudo-F": result["test statistic"],
            "p-value": result["p-value"],
        }
    )

pd.DataFrame(pairwise_results).to_csv(
    os.path.join(output_dir, "pairwise_permanova.csv"), index=False
)

# Means by model (human-readable `model` column)
group_means = scores_df.groupby("model")[dv_columns].mean().sort_index()
group_means.to_csv(os.path.join(output_dir, "group_means_by_model.csv"))

mean_diffs = []
models = group_means.index.tolist()
for dv in dv_columns:
    for m1, m2 in combinations(models, 2):
        mean_diffs.append(
            {
                "Dependent Variable": dv,
                "Model 1": m1,
                "Model 2": m2,
                "Mean Difference": group_means.loc[m2, dv] - group_means.loc[m1, dv],
            }
        )
pd.DataFrame(mean_diffs).to_csv(os.path.join(output_dir, "mean_differences.csv"), index=False)

ttest_results = []
for dv in dv_columns:
    for m1, m2 in combinations(scores_df["model"].dropna().unique(), 2):
        vals1 = scores_df[scores_df["model"] == m1][dv].dropna()
        vals2 = scores_df[scores_df["model"] == m2][dv].dropna()
        t_stat, p_val = ttest_ind(vals1, vals2, equal_var=False)
        ttest_results.append(
            {
                "Outcome": dv,
                "Model 1": m1,
                "Model 2": m2,
                "t-statistic": t_stat,
                "raw p-value": p_val,
            }
        )

ttest_df = pd.DataFrame(ttest_results)
reject, fdr_pvals, _, _ = multipletests(ttest_df["raw p-value"].fillna(1), method="fdr_bh")
ttest_df["FDR (BH) p-value"] = fdr_pvals
ttest_df["Significant (FDR < 0.05)"] = reject
ttest_df.to_csv(os.path.join(output_dir, "pairwise_ttests_fdr.csv"), index=False)

log_file.close()
print(
    f"\nCross-model analysis complete. Outputs saved to:\n  {output_dir}\n"
    f"(MANOVA/PERMANOVA grouping factor = model; pooled over chunk_size/max_tokens unless filter is set.)"
)

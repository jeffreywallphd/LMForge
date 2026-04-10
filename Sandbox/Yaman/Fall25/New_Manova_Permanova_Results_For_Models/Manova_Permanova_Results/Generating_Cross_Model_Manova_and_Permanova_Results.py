"""
Cross-model MANOVA / PERMANOVA by Chunk Size.
This script compares multiple models WITHIN each specific chunk size.
It aggregates the results into master CSV files.
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
from sklearn.covariance import MinCovDet, EmpiricalCovariance
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
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==========================================
# 1. SETUP & DATA LOADING (multi-model)
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
CODES_OUTPUTS = os.path.join(PROJECT_ROOT, "Codes_Outputs")

MODEL_RUN_SOURCES = [
    ("NEW_RUNS_LMForge_RUN01_Lab_Machine", "gemma-7b-spark"),
    ("NEW_RUNS_LMForge_RUN02_Lab_Machine", "Qwen2.5-7B-Instruct"),
    ("NEW_RUNS_LMForge_RUN03_Lab_Machine", "Mistral-7B-Instruct-v0.3"),
    ("NEW_RUNS_LMForge_RUN04_Lab_Machine", "Phi-4-mini-instruct"),
]

output_dir = os.path.join(SCRIPT_DIR, "Cross_Model_Analysis_Results")
os.makedirs(output_dir, exist_ok=True)

log_file_path = os.path.join(output_dir, "cross_model_statistical_summary.txt")
log_file = open(log_file_path, "w", encoding="utf-8")

def log_print(title, content=""):
    output = f"\n{'-' * 50}\n{title}\n{'-' * 50}\n{content}\n"
    print(output)
    log_file.write(output)

def sanitize_model_key(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_") or "model"

def load_scores_from_run_sources(sources: list, codes_outputs: str) -> pd.DataFrame:
    frames = []
    for run_folder, model_subfolder in sources:
        path = os.path.join(codes_outputs, run_folder, "Generated_Results", model_subfolder, "scores.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing scores.csv for {model_subfolder}:\n  {path}")
        df = pd.read_csv(path)
        df["model"] = model_subfolder
        df["model_key"] = sanitize_model_key(model_subfolder)
        df["lmforge_run"] = run_folder
        frames.append(df)
    
    return pd.concat(frames, ignore_index=True)

scores_df = load_scores_from_run_sources(MODEL_RUN_SOURCES, CODES_OUTPUTS)

# Clean up dataframe
drop_cols = [c for c in ["elapsed_time", "questions_num"] if c in scores_df.columns]
scores_df = scores_df.drop(columns=drop_cols, errors="ignore")
scores_df["group"] = scores_df["model_key"]

dv_columns = [
    "rouge1", "rouge2", "rougeL", "rougeLsum",
    "bert_score_P", "bert_score_R", "bert_score_F1", "sts_score"
]

# Identify unique chunk sizes
chunk_sizes = sorted(scores_df["chunk_size"].dropna().unique())
log_print("Detected Chunk Sizes", str(chunk_sizes))

# Master lists for collecting results across all chunk sizes
all_permanova_results = []
all_pairwise_permanova = []
all_ttest_results = []
all_mean_diffs = []
all_anova_results = []

# ==========================================
# MASTER LOOP: Analyze per Chunk Size
# ==========================================
for chunk in chunk_sizes:
    log_print(f"STARTING ANALYSIS FOR CHUNK SIZE: {chunk}")
    
    # Subset the data for the specific chunk
    df_chunk = scores_df[scores_df["chunk_size"] == chunk].dropna(subset=dv_columns + ["group"]).copy()
    
    if len(df_chunk["group"].unique()) < 2:
        log_print(f"Skipping Chunk {chunk}", "Not enough models to compare.")
        continue
        
    X_chunk = df_chunk[dv_columns].values
    grouping_chunk = df_chunk["group"].values

    # ------------------------------------------
    # A. OUTLIER DETECTION (Safe Covariance)
    # ------------------------------------------
    try:
        robust_cov = MinCovDet().fit(X_chunk)
        mahalanobis_distances = robust_cov.mahalanobis(X_chunk)
    except ValueError:
        # Fallback if metrics are too highly correlated causing a singular matrix
        emp_cov = EmpiricalCovariance().fit(X_chunk)
        mahalanobis_distances = emp_cov.mahalanobis(X_chunk)

    # ------------------------------------------
    # B. MAIN PERMANOVA & MANOVA
    # ------------------------------------------
    # Scale data before Euclidean distance computation!
    scaler_perm = StandardScaler()
    X_scaled = scaler_perm.fit_transform(X_chunk)
    
    # PERMANOVA
    dist_matrix = DistanceMatrix(squareform(pdist(X_scaled, metric="euclidean")))
    perm_res = permanova(dist_matrix, grouping=grouping_chunk, permutations=999)
    
    all_permanova_results.append({
        "Chunk Size": chunk,
        "Method": "PERMANOVA",
        "Test Statistic (Pseudo-F)": perm_res["test statistic"],
        "p-value": perm_res["p-value"],
        "Sample Size": perm_res["sample size"]
    })
    
    # MANOVA
    try:
        formula_simple = " + ".join(dv_columns) + " ~ C(group)"
        maov_simple = MANOVA.from_formula(formula_simple, data=df_chunk)
        log_print(f"MANOVA Results (Chunk {chunk})", maov_simple.mv_test())
    except Exception as e:
        log_print(f"MANOVA Failed for Chunk {chunk} (Likely Multicollinearity)", str(e))

    # ANOVA (Type III)
    for column in dv_columns:
        formula = f"{column} ~ C(group)"
        model_fit = ols(formula, data=df_chunk).fit()
        anova_res = anova_lm(model_fit, typ=3)
        anova_res["DV"] = column
        anova_res["Chunk Size"] = chunk
        all_anova_results.append(anova_res)

    # ------------------------------------------
    # C. PAIRWISE PERMANOVA (Post-Hoc)
    # ------------------------------------------
    for grp1, grp2 in combinations(df_chunk["group"].unique(), 2):
        subset = df_chunk[df_chunk["group"].isin([grp1, grp2])].copy().reset_index(drop=True)
        X_pair_scaled = StandardScaler().fit_transform(subset[dv_columns].values)
        
        pair_dist_mat = DistanceMatrix(
            squareform(pdist(X_pair_scaled, metric="euclidean")), 
            ids=[str(i) for i in subset.index]
        )
        grouping_df = pd.DataFrame({"group": subset["group"].values}, index=[str(i) for i in subset.index])
        
        pair_res = permanova(pair_dist_mat, grouping=grouping_df, column="group", permutations=999)
        all_pairwise_permanova.append({
            "Chunk Size": chunk,
            "Model 1": grp1,
            "Model 2": grp2,
            "Pseudo-F": pair_res["test statistic"],
            "raw p-value": pair_res["p-value"],
        })

    # ------------------------------------------
    # D. T-TESTS & MEAN DIFFERENCES
    # ------------------------------------------
    group_means = df_chunk.groupby("model")[dv_columns].mean().sort_index()
    models = group_means.index.tolist()
    
    for dv in dv_columns:
        # Mean diffs
        for m1, m2 in combinations(models, 2):
            all_mean_diffs.append({
                "Chunk Size": chunk,
                "Dependent Variable": dv,
                "Model 1": m1,
                "Model 2": m2,
                "Mean Difference (M2 - M1)": group_means.loc[m2, dv] - group_means.loc[m1, dv]
            })
            
        # T-tests
        for m1, m2 in combinations(df_chunk["model"].unique(), 2):
            vals1 = df_chunk[df_chunk["model"] == m1][dv].dropna()
            vals2 = df_chunk[df_chunk["model"] == m2][dv].dropna()
            t_stat, p_val = ttest_ind(vals1, vals2, equal_var=False)
            all_ttest_results.append({
                "Chunk Size": chunk,
                "Outcome": dv,
                "Model 1": m1,
                "Model 2": m2,
                "t-statistic": t_stat,
                "raw p-value": p_val,
            })

# ==========================================
# FINAL COMPILATION & FDR CORRECTIONS
# ==========================================

# 1. Main PERMANOVA Results
pd.DataFrame(all_permanova_results).to_csv(os.path.join(output_dir, "master_permanova_by_chunk.csv"), index=False)

# 2. Pairwise PERMANOVA (Apply FDR Correction)
perm_df = pd.DataFrame(all_pairwise_permanova)
if not perm_df.empty:
    reject_perm, fdr_pvals_perm, _, _ = multipletests(perm_df["raw p-value"].fillna(1), method="fdr_bh")
    perm_df["FDR (BH) p-value"] = fdr_pvals_perm
    perm_df["Significant (FDR < 0.05)"] = reject_perm
    perm_df.to_csv(os.path.join(output_dir, "master_pairwise_permanova.csv"), index=False)

# 3. Pairwise T-Tests (Apply FDR Correction)
ttest_df = pd.DataFrame(all_ttest_results)
if not ttest_df.empty:
    reject_t, fdr_pvals_t, _, _ = multipletests(ttest_df["raw p-value"].fillna(1), method="fdr_bh")
    ttest_df["FDR (BH) p-value"] = fdr_pvals_t
    ttest_df["Significant (FDR < 0.05)"] = reject_t
    ttest_df.to_csv(os.path.join(output_dir, "master_pairwise_ttests.csv"), index=False)

# 4. Mean Differences
pd.DataFrame(all_mean_diffs).to_csv(os.path.join(output_dir, "master_mean_differences.csv"), index=False)

# 5. ANOVA Results
if all_anova_results:
    pd.concat(all_anova_results).to_csv(os.path.join(output_dir, "master_anova_type3.csv"))

log_file.close()
print(f"\nAnalysis complete! Outputs generated in:\n  {output_dir}")
print("Results are now safely partitioned and analyzed by specific chunk sizes.")
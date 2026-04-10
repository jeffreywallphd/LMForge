import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _here(*parts: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _format_p(p: float) -> str:
    if pd.isna(p):
        return ""
    try:
        p = float(p)
    except Exception:
        return str(p)
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def _format_num(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return ""
    try:
        x = float(x)
    except Exception:
        return str(x)
    return f"{x:.{digits}f}"


def _latex_escape(s: str) -> str:
    return (
        str(s)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


def _write_tex(path: str, content: str) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(content.rstrip() + "\n")


MODEL_KEY_TO_DISPLAY = {
    "gemma_7b_spark": "gemma-7b-spark",
    "Qwen2_5_7B_Instruct": "Qwen2.5-7B-Instruct",
    "Mistral_7B_Instruct_v0_3": "Mistral-7B-Instruct-v0.3",
    "Phi_4_mini_instruct": "Phi-4-mini-instruct",
}


@dataclass(frozen=True)
class ManovaRow:
    chunk_size: int
    test: str
    value: float
    num_df: float
    den_df: float
    f_value: float
    p_value: float


def parse_cross_model_manova_summary(txt_path: str) -> List[ManovaRow]:
    """
    Parse `cross_model_statistical_summary.txt` for the `C(group)` MANOVA block
    for each chunk size. This is a lightweight parser that looks for the
    `MANOVA Results (Chunk <chunk>)` block and then parses the `C(group)` table.
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = re.split(r"STARTING ANALYSIS FOR CHUNK SIZE:\s*(\d+)", text)
    # split returns: [preamble, chunk1, block1, chunk2, block2, ...]
    rows: List[ManovaRow] = []
    for i in range(1, len(chunks), 2):
        chunk = int(chunks[i])
        block = chunks[i + 1]

        # Find the C(group) table inside the MANOVA results for this chunk.
        # We match the rows of the form:
        # Wilks' lambda     0.0000 24.0000 26.7040  35.9730 0.0000
        # Note: spacing varies; p-values are printed as 0.0000 in this log.
        cgroup_match = re.search(
            r"\n\s*C\(group\)\s*.*?\n-+\n(?P<table>.*?)(?:\n=+|\n-+\n)",
            block,
            flags=re.DOTALL,
        )
        if not cgroup_match:
            continue

        table = cgroup_match.group("table")
        for line in table.splitlines():
            line = line.strip()
            if not line or line.startswith("-") or line.startswith("="):
                continue
            # Expect: <test> <value> <numdf> <dendf> <f> <p>
            m = re.match(
                r"^(Wilks' lambda|Pillai's trace|Hotelling-Lawley trace|Roy's greatest root)\s+"
                r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$",
                line,
            )
            if not m:
                continue
            test, value, num_df, den_df, f_value, p_value = m.groups()
            rows.append(
                ManovaRow(
                    chunk_size=chunk,
                    test=test,
                    value=float(value),
                    num_df=float(num_df),
                    den_df=float(den_df),
                    f_value=float(f_value),
                    p_value=float(p_value),
                )
            )

    return rows


def build_tables() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tables_dir = _here("tables")
    _ensure_dir(tables_dir)

    # 1) Omnibus PERMANOVA by chunk
    perm = pd.read_csv(_here("master_permanova_by_chunk.csv"))
    perm = perm.sort_values("Chunk Size")
    perm_table = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Cross-model omnibus PERMANOVA by chunk size (group = model; Euclidean distance on z-scored metrics; 999 permutations).}",
        r"\label{tab:cross-permanova-omnibus}",
        r"\begin{tabular}{rccc}",
        r"\toprule",
        r"Chunk size & Pseudo-$F$ & $p$ & $N$ \\",
        r"\midrule",
    ]
    for _, row in perm.iterrows():
        perm_table.append(
            f"{int(row['Chunk Size'])} & {_format_num(row['Test Statistic (Pseudo-F)'], 3)} & {_format_p(row['p-value'])} & {int(row['Sample Size'])} \\\\"
        )
    perm_table += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write_tex(os.path.join(tables_dir, "cross_permanova_omnibus.tex"), "\n".join(perm_table))

    # 2) MANOVA mv_test summary by chunk (C(group))
    manova_rows = parse_cross_model_manova_summary(_here("cross_model_statistical_summary.txt"))
    if manova_rows:
        dfm = pd.DataFrame([r.__dict__ for r in manova_rows])
        dfm["p_fmt"] = dfm["p_value"].map(_format_p)
        dfm = dfm.sort_values(["chunk_size", "test"])
        # Wide-ish table: one row per chunk, show Wilks + Pillai only (compact) with F and p.
        keep_tests = ["Wilks' lambda", "Pillai's trace"]
        dfm2 = dfm[dfm["test"].isin(keep_tests)].copy()
        pivot = dfm2.pivot(index="chunk_size", columns="test", values=["value", "f_value", "p_fmt"])
        # Ensure deterministic column order
        chunks_sorted = sorted(pivot.index.tolist())
        man_table = [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Cross-model MANOVA tests by chunk size for $C(\mathrm{group})$ (group = model). Reported from statsmodels \texttt{mv\\_test()}; p-values in the log are printed as 0.0000 when extremely small.}",
            r"\label{tab:cross-manova-mvtest}",
            r"\small",
            r"\begin{tabular}{rcccccc}",
            r"\toprule",
            r"Chunk & Wilks' $\Lambda$ & $F$ & $p$ & Pillai & $F$ & $p$ \\",
            r"\midrule",
        ]
        for ch in chunks_sorted:
            wilks = pivot.loc[ch, ("value", "Wilks' lambda")]
            wilks_f = pivot.loc[ch, ("f_value", "Wilks' lambda")]
            wilks_p = pivot.loc[ch, ("p_fmt", "Wilks' lambda")]
            pillai = pivot.loc[ch, ("value", "Pillai's trace")]
            pillai_f = pivot.loc[ch, ("f_value", "Pillai's trace")]
            pillai_p = pivot.loc[ch, ("p_fmt", "Pillai's trace")]
            man_table.append(
                f"{int(ch)} & {_format_num(wilks, 4)} & {_format_num(wilks_f, 3)} & {wilks_p} & {_format_num(pillai, 4)} & {_format_num(pillai_f, 3)} & {pillai_p} \\\\"
            )
        man_table += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        _write_tex(os.path.join(tables_dir, "cross_manova_mvtest.tex"), "\n".join(man_table))

    # 3) Pairwise PERMANOVA (significant only)
    pair = pd.read_csv(_here("master_pairwise_permanova.csv"))
    pair["Model 1"] = pair["Model 1"].map(lambda x: MODEL_KEY_TO_DISPLAY.get(str(x), str(x)))
    pair["Model 2"] = pair["Model 2"].map(lambda x: MODEL_KEY_TO_DISPLAY.get(str(x), str(x)))
    pair = pair.sort_values(["Chunk Size", "FDR (BH) p-value", "Pseudo-F"], ascending=[True, True, False])
    sig = pair[pair["Significant (FDR < 0.05)"] == True].copy()

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Cross-model pairwise PERMANOVA results (BH-FDR corrected within the full set of pairwise contrasts across all chunk sizes as exported). Only significant pairs shown.}",
        r"\label{tab:cross-permanova-pairwise-sig}",
        r"\small",
        r"\begin{tabular}{rllcc}",
        r"\toprule",
        r"Chunk & Model 1 & Model 2 & Pseudo-$F$ & $p_{FDR}$ \\",
        r"\midrule",
    ]
    if sig.empty:
        lines.append(r"\multicolumn{5}{c}{No significant pairs after BH-FDR.} \\")
    else:
        for _, r in sig.iterrows():
            lines.append(
                f"{int(r['Chunk Size'])} & {_latex_escape(r['Model 1'])} & {_latex_escape(r['Model 2'])} & {_format_num(r['Pseudo-F'], 3)} & {_format_p(r['FDR (BH) p-value'])} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write_tex(os.path.join(tables_dir, "cross_pairwise_permanova_sig.tex"), "\n".join(lines))

    # 4) Type-III ANOVA p-values + partial eta^2 for C(group), per DV and chunk
    anova = pd.read_csv(_here("master_anova_type3.csv"))
    # Normalize column names (this file includes an unnamed first column sometimes)
    if anova.columns[0] == "Unnamed: 0" or anova.columns[0] == "":
        anova = anova.rename(columns={anova.columns[0]: "term"})
    else:
        anova = anova.rename(columns={anova.columns[0]: "term"})

    # keep only effect + residual for eta
    effect = anova[anova["term"].isin(["C(group)", "Residual"])].copy()
    # compute eta_p^2 by DV+chunk
    eff = effect.pivot_table(
        index=["DV", "Chunk Size"],
        columns="term",
        values="sum_sq",
        aggfunc="first",
    ).reset_index()
    eff["partial_eta_sq"] = eff["C(group)"] / (eff["C(group)"] + eff["Residual"])

    pvals = (
        anova[anova["term"] == "C(group)"][["DV", "Chunk Size", "PR(>F)"]]
        .rename(columns={"PR(>F)": "p_value"})
        .copy()
    )
    merged = pd.merge(pvals, eff[["DV", "Chunk Size", "partial_eta_sq"]], on=["DV", "Chunk Size"], how="left")

    d_order = [
        "rouge1",
        "rouge2",
        "rougeL",
        "rougeLsum",
        "bert_score_P",
        "bert_score_R",
        "bert_score_F1",
        "sts_score",
    ]
    merged["DV"] = pd.Categorical(merged["DV"], categories=d_order, ordered=True)
    merged = merged.sort_values(["DV", "Chunk Size"])

    # Two compact tables: p-values and eta_p^2
    p_wide = merged.pivot(index="DV", columns="Chunk Size", values="p_value").reindex(d_order)
    eta_wide = merged.pivot(index="DV", columns="Chunk Size", values="partial_eta_sq").reindex(d_order)

    def _wide_table(title: str, label: str, wide: pd.DataFrame, fmt) -> str:
        cols = [int(c) for c in wide.columns.tolist()]
        header = "DV & " + " & ".join(str(c) for c in cols) + r" \\"
        lines2 = [
            r"\begin{table}[H]",
            r"\centering",
            rf"\caption{{{title}}}",
            rf"\label{{{label}}}",
            r"\small",
            r"\begin{tabular}{l" + ("c" * len(cols)) + r"}",
            r"\toprule",
            header,
            r"\midrule",
        ]
        for dv, row in wide.iterrows():
            vals = [fmt(row[c]) for c in wide.columns]
            lines2.append(f"{_latex_escape(dv)} & " + " & ".join(vals) + r" \\")
        lines2 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        return "\n".join(lines2)

    p_table_tex = _wide_table(
        title="Cross-model Type-III ANOVA $p$-values for $C(\\mathrm{group})$ (group = model), by metric and chunk size.",
        label="tab:cross-anova-pvalues",
        wide=p_wide,
        fmt=_format_p,
    )
    _write_tex(os.path.join(tables_dir, "cross_anova_pvalues.tex"), p_table_tex)

    eta_table_tex = _wide_table(
        title="Cross-model partial $\\eta^2$ for $C(\\mathrm{group})$ from Type-III ANOVA (per metric, per chunk).",
        label="tab:cross-anova-eta",
        wide=eta_wide,
        fmt=lambda x: _format_num(x, 3) if not pd.isna(x) else "",
    )
    _write_tex(os.path.join(tables_dir, "cross_anova_eta.tex"), eta_table_tex)


def build_plots() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Cross-model plots
    perm = pd.read_csv(_here("master_permanova_by_chunk.csv")).sort_values("Chunk Size")

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(perm["Chunk Size"].astype(str), perm["Test Statistic (Pseudo-F)"], color="#4C78A8")
    ax.set_xlabel("Chunk size")
    ax.set_ylabel("PERMANOVA pseudo-$F$")
    ax.set_title("Cross-model omnibus PERMANOVA by chunk size")
    fig.tight_layout()
    fig.savefig(_here("cross_model_permanova_omnibus_bar.png"), dpi=200)
    plt.close(fig)

    pair = pd.read_csv(_here("master_pairwise_permanova.csv"))
    pair["Model 1"] = pair["Model 1"].map(lambda x: MODEL_KEY_TO_DISPLAY.get(str(x), str(x)))
    pair["Model 2"] = pair["Model 2"].map(lambda x: MODEL_KEY_TO_DISPLAY.get(str(x), str(x)))

    models = [
        "gemma-7b-spark",
        "Qwen2.5-7B-Instruct",
        "Mistral-7B-Instruct-v0.3",
        "Phi-4-mini-instruct",
    ]

    for chunk, dfc in pair.groupby("Chunk Size"):
        mat = pd.DataFrame(np.nan, index=models, columns=models)
        sig = pd.DataFrame(False, index=models, columns=models)
        for _, r in dfc.iterrows():
            m1, m2 = r["Model 1"], r["Model 2"]
            if m1 not in models or m2 not in models:
                continue
            mat.loc[m1, m2] = r["Pseudo-F"]
            mat.loc[m2, m1] = r["Pseudo-F"]
            s = bool(r["Significant (FDR < 0.05)"])
            sig.loc[m1, m2] = s
            sig.loc[m2, m1] = s
        np.fill_diagonal(mat.values, 0.0)
        np.fill_diagonal(sig.values, False)

        fig, ax = plt.subplots(figsize=(7.4, 6.2))
        sns.heatmap(
            mat,
            ax=ax,
            cmap="viridis",
            annot=True,
            fmt=".1f",
            cbar_kws={"label": "Pseudo-$F$"},
            linewidths=0.5,
            linecolor="white",
        )
        # Mark significant cells
        for i in range(len(models)):
            for j in range(len(models)):
                if i == j:
                    continue
                if sig.iat[i, j]:
                    ax.text(j + 0.5, i + 0.5, "*", ha="center", va="center", color="white", fontsize=14)
        ax.set_title(f"Pairwise PERMANOVA pseudo-$F$ (chunk {int(chunk)})\\n* = BH-FDR < 0.05")
        fig.tight_layout()
        fig.savefig(_here(f"cross_model_pairwise_permanova_heatmap_chunk_{int(chunk)}.png"), dpi=200)
        plt.close(fig)

    # ANOVA p-values heatmap
    anova = pd.read_csv(_here("master_anova_type3.csv"))
    if anova.columns[0] == "Unnamed: 0" or anova.columns[0] == "":
        anova = anova.rename(columns={anova.columns[0]: "term"})
    else:
        anova = anova.rename(columns={anova.columns[0]: "term"})
    p = anova[anova["term"] == "C(group)"][["DV", "Chunk Size", "PR(>F)"]].copy()
    d_order = [
        "rouge1",
        "rouge2",
        "rougeL",
        "rougeLsum",
        "bert_score_P",
        "bert_score_R",
        "bert_score_F1",
        "sts_score",
    ]
    p["DV"] = pd.Categorical(p["DV"], categories=d_order, ordered=True)
    pv = p.pivot(index="DV", columns="Chunk Size", values="PR(>F)").reindex(d_order)
    pv = -np.log10(pv.astype(float))

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    sns.heatmap(pv, ax=ax, cmap="magma", annot=True, fmt=".2f", cbar_kws={"label": "$-\\log_{10}(p)$"})
    ax.set_xlabel("Chunk size")
    ax.set_ylabel("Metric (DV)")
    ax.set_title("Cross-model ANOVA significance by metric and chunk size")
    fig.tight_layout()
    fig.savefig(_here("cross_model_anova_pvalues_heatmap.png"), dpi=200)
    plt.close(fig)


def build_within_model_plots() -> None:
    """
    Regenerate within-model plots referenced by `LATEX_writeup/decoder_models_lab_progress_report.tex`.
    The original report expects these PNGs under:
      Manova_Permanova_Results/<model>/{lineplot_bert_f1,boxplot_bert_f1,correlation_heatmap,qq_plot_mahalanobis,pca_scatter}.png
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import probplot
    from scipy.spatial.distance import pdist, squareform
    from sklearn.covariance import MinCovDet
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    project_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    codes_outputs = os.path.join(project_root, "Codes_Outputs")
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

    model_sources: List[Tuple[str, str]] = [
        ("NEW_RUNS_LMForge_RUN01_Lab_Machine", "gemma-7b-spark"),
        ("NEW_RUNS_LMForge_RUN02_Lab_Machine", "Qwen2.5-7B-Instruct"),
        ("NEW_RUNS_LMForge_RUN03_Lab_Machine", "Mistral-7B-Instruct-v0.3"),
        ("NEW_RUNS_LMForge_RUN04_Lab_Machine", "Phi-4-mini-instruct"),
    ]

    sns.set_theme(style="whitegrid")

    for run_folder, model_name in model_sources:
        scores_path = os.path.join(codes_outputs, run_folder, "Generated_Results", model_name, "scores.csv")
        if not os.path.isfile(scores_path):
            continue

        df = pd.read_csv(scores_path)
        for c in ["elapsed_time", "questions_num"]:
            if c in df.columns:
                df = df.drop(columns=[c])

        # match the within-model script behavior
        if "chunk_size" in df.columns and "max_tokens" in df.columns:
            df = df.drop_duplicates(subset=["chunk_size", "max_tokens"], keep="first")

        out_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", model_name))
        _ensure_dir(out_dir)

        X = df[dv_columns].dropna()
        if len(X) >= 3:
            try:
                robust_cov = MinCovDet().fit(X.values)
                mahal = robust_cov.mahalanobis(X.values)
            except Exception:
                mahal = np.full(len(X), np.nan)

            fig = plt.figure(figsize=(6.4, 4.2))
            probplot(mahal, dist="chi2", sparams=(len(dv_columns),), plot=plt)
            plt.title("Q-Q Plot of Mahalanobis Distances")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "qq_plot_mahalanobis.png"), dpi=200)
            plt.close(fig)

        # correlation heatmap
        fig, ax = plt.subplots(figsize=(7.2, 6.0))
        corr = df[dv_columns].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation between DVs")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "correlation_heatmap.png"), dpi=200)
        plt.close(fig)

        # boxplot and lineplot for BERT F1
        if {"chunk_size", "max_tokens", "bert_score_F1"}.issubset(df.columns):
            fig, ax = plt.subplots(figsize=(8.2, 4.8))
            sns.boxplot(x="chunk_size", y="bert_score_F1", hue="max_tokens", data=df, palette="Set2", ax=ax)
            ax.set_xlabel("Chunk size")
            ax.set_ylabel("BERTScore F1")
            ax.set_title("BERTScore F1 by chunk size and max tokens")
            ax.legend(title="Max tokens", fontsize=8, title_fontsize=9, loc="best")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "boxplot_bert_f1.png"), dpi=200)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8.2, 4.8))
            sns.lineplot(data=df, x="chunk_size", y="bert_score_F1", hue="max_tokens", marker="o", ax=ax)
            ax.set_xlabel("Chunk size")
            ax.set_ylabel("BERTScore F1")
            ax.set_title("BERTScore F1 by chunk size and max tokens")
            ax.legend(title="Max tokens", fontsize=8, title_fontsize=9, loc="best")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "lineplot_bert_f1.png"), dpi=200)
            plt.close(fig)

        # PCA scatter (z-scored DVs)
        clean = df.dropna(subset=dv_columns + (["chunk_size"] if "chunk_size" in df.columns else []))
        if len(clean) >= 3 and "chunk_size" in clean.columns:
            scaler = StandardScaler()
            Xs = scaler.fit_transform(clean[dv_columns].values)
            pca = PCA(n_components=2)
            pcs = pca.fit_transform(Xs)
            pca_df = pd.DataFrame({"PC1": pcs[:, 0], "PC2": pcs[:, 1], "chunk_size": clean["chunk_size"].astype(str).values})

            fig, ax = plt.subplots(figsize=(7.4, 4.8))
            sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="chunk_size", style="chunk_size", ax=ax)
            ax.set_title("PCA of dependent variables by chunk size")
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.legend(title="Chunk size", fontsize=8, title_fontsize=9, loc="best")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "pca_scatter.png"), dpi=200)
            plt.close(fig)


def main() -> None:
    build_tables()
    build_plots()
    build_within_model_plots()
    print("Done. Wrote cross-model tables/plots and regenerated within-model plots.")


if __name__ == "__main__":
    main()


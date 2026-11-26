import logging
import os
from typing import Optional, TypedDict

import pandas as pd
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from scipy.stats import chi2


logger = logging.getLogger(__name__)
WORKFLOW_NAME = "eda_workflow"
LOG_PATH = os.path.join(os.getcwd(), "logs/")
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompt_path = os.path.join(PROMPTS_DIR, filename)
    with open(prompt_path, "r") as f:
        return f.read()


class EDAWorkflow:
    """
    Exploratory Data Analysis workflow that performs consistent, first-pass analysis of datasets.
    
    Uses a fixed set of predefined analysis tools to produce structured, tabular outputs.
    Operates sequentially and deterministically through baseline EDA steps.
    
    Parameters
    ----------
    model : LLM, optional
        Language model for synthesizing findings.
    log : bool, default=False
        Whether to save analysis results to a file.
    log_path : str, optional
        Directory for log files.
    checkpointer : Checkpointer, optional
        LangGraph checkpointer for saving workflow state.
    
    Attributes
    ----------
    response : dict or None
        Stores the full response after invoke_workflow() is called.
    """
    
    def __init__(
        self,
        model=None,
        log=False,
        log_path=None,
        checkpointer: Optional[object] = None
    ):
        self.model = model
        self.log = log
        self.log_path = log_path
        self.checkpointer = checkpointer
        self.response = None
        self._compiled_graph = make_eda_baseline_workflow(
            model=model,
            log=log,
            log_path=log_path,
            checkpointer=checkpointer
        )
    
    def invoke_workflow(self, filepath: str, **kwargs):
        """
        Run EDA analysis on the provided dataset.
        
        Parameters
        ----------
        filepath : str
            Path to the dataset file.
        **kwargs
            Additional arguments passed to the underlying graph invoke method.
        
        Returns
        -------
        None
            Results are stored in self.response and accessed via getter methods.
        """
        df = pd.read_csv(filepath)
        
        response = self._compiled_graph.invoke({
            "dataframe": df.to_dict(),
            "results": {},
            "observations": {},
            "current_step": "",
            "summary": "",
            "recommendations": [],
        }, **kwargs)
        
        self.response = response
        return None
    
    def get_summary(self):
        """Retrieves the analysis summary."""
        if self.response:
            return self.response.get("summary")
    
    def get_recommendations(self):
        """Retrieves the recommendations."""
        if self.response:
            return self.response.get("recommendations")
    
    def get_results(self):
        """Retrieves the full analysis results."""
        if self.response:
            return self.response.get("results")
    
    def get_observations(self):
        """Retrieves all observations from analysis steps."""
        if self.response:
            return self.response.get("observations")


def make_eda_baseline_workflow(
    model=None,
    log=False,
    log_path=None,
    checkpointer: Optional[object] = None
):
    """
    Factory function that creates a compiled LangGraph workflow for baseline EDA.
    
    Performs automated first-pass analysis with fixed analysis steps.
    
    Parameters
    ----------
    model : LLM, optional
        Language model for synthesizing findings.
    log : bool, default=False
        Whether to save analysis results to a file.
    log_path : str, optional
        Directory for log files.
    checkpointer : Checkpointer, optional
        LangGraph checkpointer for saving workflow state.
    
    Returns
    -------
    CompiledStateGraph
        Compiled LangGraph workflow ready to process EDA requests.
    """
    if log:
        if log_path is None:
            log_path = LOG_PATH
        if not os.path.exists(log_path):
            os.makedirs(log_path)
    
    class EDAState(TypedDict):
        dataframe: dict
        results: dict
        observations: dict[str, list[str]]
        current_step: str
        summary: str
        recommendations: list[str]
    
    def profile_dataset_node(state: EDAState):
        """Generate dataset profile with basic statistics."""
        logger.info("Profiling dataset")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})
        
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        profile = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "numeric_summary": (
                df[numeric_cols].describe().to_dict() if numeric_cols else {}
            ),
            "categorical_summary": {
                col: df[col].value_counts().head(10).to_dict()
                for col in categorical_cols
            },
        }
        
        results["profile_dataset"] = profile
        
        return {
            "current_step": "profile_dataset",
            "results": results,
        }
    
    def analyze_missingness_node(state: EDAState):
        """Analyze missing values in the dataset."""
        logger.info("Analyzing missingness")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})
        
        missing_count = df.isnull().sum().to_dict()
        missing_pct = (
            (df.isnull().sum() / len(df) * 100).round(2).to_dict()
        )
        
        high_missing = {col: pct for col, pct in missing_pct.items() if pct > 20}
        
        missingness = {
            "total_rows": len(df),
            "missing_count": missing_count,
            "missing_percentage": missing_pct,
            "high_missing_columns": high_missing,
            "complete_rows": int(df.dropna().shape[0]),
            "complete_rows_pct": (
                round(df.dropna().shape[0] / len(df) * 100, 2)
                if len(df) > 0 else 0
            ),
        }
        
        results["analyze_missingness"] = missingness
        
        return {
            "current_step": "analyze_missingness",
            "results": results,
        }
    
    def compute_aggregates_node(state: EDAState):
        """Compute group-by aggregates on key columns.

        Produces three kinds of results, stored in results["compute_aggregates"]:
        - numeric_stats: descriptive statistics for each numeric column.
        - categorical_stats: frequency statistics for each categorical column
          (missing values are reported separately, never as a category).
        - grouped_stats: numeric statistics per category, for categorical
          columns with a manageable number of categories.
        """
        logger.info("Computing aggregates")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})

        max_categories = 10         # categories listed per categorical column
        max_group_cardinality = 10  # max unique values to group by a column

        def to_py(value, digits=4):
            """Convert a numeric scalar to a rounded float; NaN/None -> None."""
            if value is None or pd.isna(value):
                return None
            return round(float(value), digits)

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        aggregates = {
            "numeric_stats": {},
            "categorical_stats": {},
            "grouped_stats": {},
            "grouping_columns": [],
            "skipped_grouping_columns": {},
            "notes": [],
        }

        if df.empty:
            aggregates["notes"].append("Dataset is empty; no aggregates computed.")
            results["compute_aggregates"] = aggregates
            return {
                "current_step": "compute_aggregates",
                "results": results,
            }

        # A. Descriptive statistics for numeric columns
        for col in numeric_cols:
            s = df[col]
            if s.count() == 0:  # all values missing: nothing to compute
                aggregates["numeric_stats"][str(col)] = {
                    stat: (0 if stat == "count" else None)
                    for stat in ["count", "mean", "median", "std", "min", "q1", "q3", "max"]
                }
                continue
            aggregates["numeric_stats"][str(col)] = {
                "count": int(s.count()),
                "mean": to_py(s.mean()),
                "median": to_py(s.median()),
                "std": to_py(s.std()),
                "min": to_py(s.min()),
                "q1": to_py(s.quantile(0.25)),
                "q3": to_py(s.quantile(0.75)),
                "max": to_py(s.max()),
            }
        if not numeric_cols:
            aggregates["notes"].append("No numeric columns found.")

        # B. Frequency statistics for categorical columns
        for col in categorical_cols:
            s = df[col]
            counts = s.value_counts(dropna=True)
            n_valid = int(counts.sum())
            n_missing = int(s.isna().sum())
            top = counts.head(max_categories)
            aggregates["categorical_stats"][str(col)] = {
                "unique_count": int(len(counts)),
                "missing_count": n_missing,
                "missing_pct": round(n_missing / len(df) * 100, 2),
                "top_category": str(counts.index[0]) if n_valid else None,
                "top_count": int(counts.iloc[0]) if n_valid else None,
                "top_pct": (
                    round(int(counts.iloc[0]) / n_valid * 100, 2) if n_valid else None
                ),
                "frequencies": {str(k): int(v) for k, v in top.items()},
                "percentages": {
                    str(k): round(int(v) / n_valid * 100, 2) for k, v in top.items()
                },
                "other_count": n_valid - int(top.sum()),
            }
        if not categorical_cols:
            aggregates["notes"].append("No categorical columns found.")

        # C. Group-by aggregates (numeric stats per category)
        if numeric_cols and categorical_cols:
            stat_names = ["mean", "median", "std", "min", "max"]
            for col in categorical_cols:
                n_unique = aggregates["categorical_stats"][str(col)]["unique_count"]
                if n_unique < 2:
                    aggregates["skipped_grouping_columns"][str(col)] = (
                        "fewer than 2 categories"
                    )
                    continue
                if n_unique > max_group_cardinality:
                    aggregates["skipped_grouping_columns"][str(col)] = (
                        f"high cardinality ({n_unique} categories > {max_group_cardinality})"
                    )
                    continue

                grouped = df.groupby(col, observed=True)
                stats = grouped[numeric_cols].agg(stat_names)
                sizes = grouped.size()
                group_result = {}
                for category in stats.index:
                    entry = {"count": int(sizes[category])}
                    for num_col in numeric_cols:
                        entry[str(num_col)] = {
                            stat: to_py(stats.loc[category, (num_col, stat)])
                            for stat in stat_names
                        }
                    group_result[str(category)] = entry

                aggregates["grouped_stats"][str(col)] = group_result
                aggregates["grouping_columns"].append(str(col))

            if not aggregates["grouping_columns"]:
                aggregates["notes"].append(
                    "No categorical column suitable for grouping."
                )
        elif numeric_cols or categorical_cols:
            aggregates["notes"].append(
                "Group-by aggregates need both numeric and categorical columns."
            )

        results["compute_aggregates"] = aggregates
        
        return {
            "current_step": "compute_aggregates",
            "results": results,
        }

    
    def analyze_relationships_node(state: EDAState):
        """Analyze relationships between variables.

        Produces three kinds of results, stored in results["analyze_relationships"]:
        - numeric_numeric: Pearson and Spearman correlations (pairwise-complete
          observations) between eligible numeric columns, with the strongest
          positive and negative pairs, strong pairs (|r| >= 0.7) and pairs where
          Pearson and Spearman diverge.
        - categorical_numeric: per-category count, mean, median, min and max of
          each numeric column, plus eta-squared as an effect size, for
          categorical columns with a manageable number of categories.
        - categorical_categorical: contingency tables, row proportions, the
          chi-square test of independence (flagged valid only when expected
          counts are adequate) and Cramer's V for pairs of categorical columns.

        Also contains "skipped" (columns/pairs excluded, with reasons), "summary"
        (counts) and "notes". Missing values are never treated as a category and
        never imputed. If SciPy is unavailable, chi-square p-values are None.
        """
        logger.info("Analyzing relationships")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})

        min_observations = 10       # min paired observations for any pair
        max_categories = 10         # max unique values for a categorical column
        top_n = 5                   # strongest positive/negative correlations
        min_group_size = 5          # groups smaller than this are flagged
        strong_corr = 0.7           # |r| threshold for a "strong" correlation
        divergence_threshold = 0.3  # |pearson - spearman| flagged as divergent
        max_numeric_pairs = 50      # numeric pairs listed in full
        max_cat_num_pairs = 20      # categorical-numeric pairs listed in full
        max_cat_cat_detailed = 15   # categorical pairs with full tables
        alpha = 0.05

        try:
            from scipy.stats import chi2 as scipy_chi2
        except ImportError:
            scipy_chi2 = None

        def to_py(value, digits=4):
            """Convert a numeric scalar to a rounded float; NaN/None -> None."""
            if value is None or pd.isna(value):
                return None
            return round(float(value), digits)

        def desc_key(field):
            """Sort key: descending by field, None values last."""
            return lambda r: (r[field] is None, -(r[field] or 0))

        relationships = {
            "numeric_numeric": {
                "eligible_columns": [],
                "pairs_analyzed": 0,
                "top_positive": [],
                "top_negative": [],
                "strong_pairs": [],
                "divergent_pairs": [],
                "all_pairs": [],
            },
            "categorical_numeric": {
                "eligible_categorical": [],
                "eligible_numeric": [],
                "pairs_analyzed": 0,
                "pairs": [],
            },
            "categorical_categorical": {
                "eligible_columns": [],
                "pairs_analyzed": 0,
                "pairs": [],
                "other_pairs": [],
            },
            "skipped": {
                "numeric_columns": {},
                "categorical_columns": {},
                "numeric_pairs": {},
                "categorical_numeric_pairs": {},
                "categorical_pairs": {},
            },
            "summary": {},
            "notes": [],
        }
        notes = relationships["notes"]
        skipped = relationships["skipped"]

        if df.empty:
            notes.append("Dataset is empty; no relationships analyzed.")
            results["analyze_relationships"] = relationships
            return {
                "current_step": "analyze_relationships",
                "results": results,
            }

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        num_df = df[numeric_cols].replace([float("inf"), float("-inf")], float("nan"))

        if not numeric_cols:
            notes.append("No numeric columns found.")
        if not categorical_cols:
            notes.append("No categorical columns found.")

        # A. Numerical-numerical relationships
        eligible_num = []
        for col in numeric_cols:
            s = num_df[col]
            if int(s.count()) < min_observations:
                skipped["numeric_columns"][str(col)] = (
                    f"fewer than {min_observations} non-missing values"
                )
            elif s.nunique(dropna=True) < 2:
                skipped["numeric_columns"][str(col)] = "constant column"
            else:
                eligible_num.append(col)
        relationships["numeric_numeric"]["eligible_columns"] = [
            str(c) for c in eligible_num
        ]

        if len(eligible_num) < 2:
            if numeric_cols:
                notes.append(
                    "Fewer than 2 eligible numeric columns; "
                    "no numeric-numeric correlations computed."
                )
        else:
            num_pairs = []
            for i, col_a in enumerate(eligible_num):
                for col_b in eligible_num[i + 1:]:
                    pair_name = f"{col_a} | {col_b}"
                    pair = num_df[[col_a, col_b]].dropna()
                    if len(pair) < min_observations:
                        skipped["numeric_pairs"][pair_name] = (
                            f"fewer than {min_observations} paired observations"
                        )
                        continue
                    if pair[col_a].nunique() < 2 or pair[col_b].nunique() < 2:
                        skipped["numeric_pairs"][pair_name] = (
                            "constant in the paired observations"
                        )
                        continue
                    num_pairs.append({
                        "variable_1": str(col_a),
                        "variable_2": str(col_b),
                        "n": int(len(pair)),
                        "pearson": to_py(pair[col_a].corr(pair[col_b])),
                        "spearman": to_py(
                            pair[col_a].corr(pair[col_b], method="spearman")
                        ),
                    })

            section = relationships["numeric_numeric"]
            section["pairs_analyzed"] = len(num_pairs)
            valid = [p for p in num_pairs if p["pearson"] is not None]
            section["top_positive"] = sorted(
                [p for p in valid if p["pearson"] > 0],
                key=lambda p: -p["pearson"],
            )[:top_n]
            section["top_negative"] = sorted(
                [p for p in valid if p["pearson"] < 0],
                key=lambda p: p["pearson"],
            )[:top_n]
            section["strong_pairs"] = [
                p for p in num_pairs
                if any(
                    p[m] is not None and abs(p[m]) >= strong_corr
                    for m in ("pearson", "spearman")
                )
            ]
            section["divergent_pairs"] = [
                p for p in valid
                if p["spearman"] is not None
                and abs(p["pearson"] - p["spearman"]) >= divergence_threshold
            ]
            ranked = sorted(
                num_pairs,
                key=lambda p: (
                    p["pearson"] is None,
                    -abs(p["pearson"] or 0),
                ),
            )
            section["all_pairs"] = ranked[:max_numeric_pairs]
            if len(ranked) > max_numeric_pairs:
                notes.append(
                    f"all_pairs truncated to the {max_numeric_pairs} strongest "
                    f"of {len(ranked)} numeric pairs."
                )
            if not num_pairs:
                notes.append("No numeric pair had enough valid observations.")

        # Eligibility of categorical columns (shared by B and C)
        eligible_cat = []
        for col in categorical_cols:
            n_unique = int(df[col].nunique(dropna=True))
            if n_unique < 2:
                skipped["categorical_columns"][str(col)] = "fewer than 2 categories"
            elif n_unique > max_categories:
                skipped["categorical_columns"][str(col)] = (
                    f"high cardinality ({n_unique} categories > {max_categories})"
                )
            else:
                eligible_cat.append(col)

        # B. Categorical-numerical relationships
        usable_num = [c for c in numeric_cols if int(num_df[c].count()) > 0]
        relationships["categorical_numeric"]["eligible_categorical"] = [
            str(c) for c in eligible_cat
        ]
        relationships["categorical_numeric"]["eligible_numeric"] = [
            str(c) for c in usable_num
        ]

        if numeric_cols and categorical_cols:
            cn_pairs = []
            for cat in eligible_cat:
                for num in usable_num:
                    pair_name = f"{cat} | {num}"
                    sub = pd.DataFrame({
                        "cat": df[cat], "num": num_df[num]
                    }).dropna()
                    if sub.empty or sub["cat"].nunique() < 2:
                        skipped["categorical_numeric_pairs"][pair_name] = (
                            "fewer than 2 categories with numeric values"
                        )
                        continue

                    grouped = sub.groupby("cat", observed=True)["num"]
                    stats = grouped.agg(["count", "mean", "median", "min", "max"])
                    overall_mean = sub["num"].mean()
                    total_ss = float(((sub["num"] - overall_mean) ** 2).sum())
                    between_ss = float(
                        (stats["count"] * (stats["mean"] - overall_mean) ** 2).sum()
                    )
                    eta_sq = (
                        to_py(between_ss / total_ss) if total_ss > 0 else None
                    )

                    groups = {}
                    for category in stats.index:
                        row = stats.loc[category]
                        groups[str(category)] = {
                            "count": int(row["count"]),
                            "mean": to_py(row["mean"]),
                            "median": to_py(row["median"]),
                            "min": to_py(row["min"]),
                            "max": to_py(row["max"]),
                            "small_group": bool(row["count"] < min_group_size),
                        }
                    cn_pairs.append({
                        "categorical": str(cat),
                        "numeric": str(num),
                        "n": int(len(sub)),
                        "eta_squared": eta_sq,
                        "groups": groups,
                    })

            cn_pairs.sort(key=desc_key("eta_squared"))
            relationships["categorical_numeric"]["pairs_analyzed"] = len(cn_pairs)
            relationships["categorical_numeric"]["pairs"] = cn_pairs[:max_cat_num_pairs]
            if len(cn_pairs) > max_cat_num_pairs:
                notes.append(
                    f"categorical_numeric pairs truncated to the "
                    f"{max_cat_num_pairs} highest eta-squared of {len(cn_pairs)}."
                )
            if not cn_pairs:
                notes.append(
                    "No categorical column suitable for grouping numeric columns."
                )
        elif numeric_cols or categorical_cols:
            notes.append(
                "Categorical-numeric analysis needs both numeric and "
                "categorical columns."
            )

        # C. Categorical-categorical relationships
        relationships["categorical_categorical"]["eligible_columns"] = [
            str(c) for c in eligible_cat
        ]
        if len(eligible_cat) < 2:
            if categorical_cols:
                notes.append(
                    "Fewer than 2 eligible categorical columns; "
                    "no categorical-categorical analysis."
                )
        else:
            cc_pairs = []
            for i, col_a in enumerate(eligible_cat):
                for col_b in eligible_cat[i + 1:]:
                    pair_name = f"{col_a} | {col_b}"
                    sub = df[[col_a, col_b]].dropna()
                    if len(sub) < min_observations:
                        skipped["categorical_pairs"][pair_name] = (
                            f"fewer than {min_observations} paired observations"
                        )
                        continue
                    table = pd.crosstab(sub[col_a], sub[col_b])
                    n_rows, n_cols = table.shape
                    if n_rows < 2 or n_cols < 2:
                        skipped["categorical_pairs"][pair_name] = (
                            "fewer than 2 categories in the paired observations"
                        )
                        continue

                    n = int(table.to_numpy().sum())
                    row_tot = table.sum(axis=1)
                    col_tot = table.sum(axis=0)
                    expected = pd.DataFrame(
                        row_tot.to_numpy()[:, None] * col_tot.to_numpy()[None, :] / n,
                        index=table.index,
                        columns=table.columns,
                    )
                    statistic = float(
                        (((table - expected) ** 2) / expected).to_numpy().sum()
                    )
                    dof = (n_rows - 1) * (n_cols - 1)
                    min_expected = float(expected.to_numpy().min())
                    pct_ge_5 = float((expected >= 5).to_numpy().mean() * 100)
                    is_valid = min_expected >= 1 and pct_ge_5 >= 80
                    cramers_v = to_py(
                        (statistic / (n * (min(n_rows, n_cols) - 1))) ** 0.5
                    )

                    chi_square = {
                        "statistic": to_py(statistic),
                        "degrees_of_freedom": int(dof),
                        "p_value": None,
                        "significant_at_0.05": None,
                        "valid": bool(is_valid),
                        "min_expected": to_py(min_expected),
                        "pct_expected_ge_5": round(pct_ge_5, 2),
                    }
                    if not is_valid:
                        chi_square["reason"] = (
                            "expected counts too small (need all >= 1 and "
                            "at least 80% >= 5)"
                        )
                    elif scipy_chi2 is not None:
                        p_value = float(scipy_chi2.sf(statistic, dof))
                        chi_square["p_value"] = float(f"{p_value:.6g}")
                        chi_square["significant_at_0.05"] = bool(p_value < alpha)

                    proportions = table.div(row_tot, axis=0)
                    cc_pairs.append({
                        "variable_1": str(col_a),
                        "variable_2": str(col_b),
                        "n": n,
                        "contingency_table": {
                            str(r): {str(c): int(v) for c, v in table.loc[r].items()}
                            for r in table.index
                        },
                        "row_proportions": {
                            str(r): {
                                str(c): round(float(v), 4)
                                for c, v in proportions.loc[r].items()
                            }
                            for r in proportions.index
                        },
                        "chi_square": chi_square,
                        "cramers_v": cramers_v,
                    })

            cc_pairs.sort(key=desc_key("cramers_v"))
            section = relationships["categorical_categorical"]
            section["pairs_analyzed"] = len(cc_pairs)
            section["pairs"] = cc_pairs[:max_cat_cat_detailed]
            section["other_pairs"] = [
                {
                    "variable_1": p["variable_1"],
                    "variable_2": p["variable_2"],
                    "n": p["n"],
                    "cramers_v": p["cramers_v"],
                }
                for p in cc_pairs[max_cat_cat_detailed:]
            ]
            if scipy_chi2 is None and cc_pairs:
                notes.append(
                    "SciPy is unavailable; chi-square p-values are None "
                    "(statistic, degrees of freedom and Cramer's V are still reported)."
                )
            if not cc_pairs:
                notes.append("No categorical pair had enough valid observations.")

        relationships["summary"] = {
            "numeric_columns_eligible": len(eligible_num),
            "numeric_columns_skipped": len(skipped["numeric_columns"]),
            "categorical_columns_eligible": len(eligible_cat),
            "categorical_columns_skipped": len(skipped["categorical_columns"]),
            "numeric_numeric_pairs": relationships["numeric_numeric"]["pairs_analyzed"],
            "categorical_numeric_pairs": relationships["categorical_numeric"]["pairs_analyzed"],
            "categorical_categorical_pairs": relationships["categorical_categorical"]["pairs_analyzed"],
        }
        logger.info("Relationship analysis summary: %s", relationships["summary"])

        results["analyze_relationships"] = relationships

        return {
            "current_step": "analyze_relationships",
            "results": results,
        }


    def extract_observations_node(state: EDAState):
        """Extract observations from the latest analysis results using LLM."""
        logger.info("Extracting observations")
        
        current_step = state.get("current_step", "")
        results = state.get("results", {})
        observations = state.get("observations", {})
        
        if model is None or not current_step or current_step not in results:
            return {"observations": observations}
        
        step_results = results.get(current_step, {})
        
        class ObservationOutput(BaseModel):
            observations: list[str] = Field(description="1-2 concise, actionable observations")
        
        observation_prompt = ChatPromptTemplate.from_messages([
            ("system", load_prompt("extract_observations_system.txt")),
            ("human", load_prompt("extract_observations_human.txt")),
        ])
        
        chain = observation_prompt | model.with_structured_output(ObservationOutput)
        response = chain.invoke({
            "step_name": current_step.replace("_", " ").title(),
            "results": str(step_results)
        })
        
        observations[current_step] = response.observations
        
        return {
            "observations": observations,
        }
    
    def synthesize_findings_node(state: EDAState):
        """Synthesize accumulated findings into summary and recommendations."""
        logger.info("Synthesizing findings")
        
        observations = state.get("observations", {})
        
        if model is None:
            return {
                "summary": "No LLM provided for synthesis",
                "recommendations": [],
            }
        
        class SynthesisOutput(BaseModel):
            summary: str = Field(description="A concise 2-3 sentence summary of key findings")
            recommendations: list[str] = Field(description="3-5 actionable recommendations")
        
        all_observations = []
        for step_name, step_obs in observations.items():
            all_observations.append(f"\n{step_name.replace('_', ' ').title()}:")
            for obs in step_obs:
                all_observations.append(f"  - {obs}")
        
        observations_text = "\n".join(all_observations)
        
        synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", load_prompt("synthesize_findings_system.txt")),
            ("human", load_prompt("synthesize_findings_human.txt")),
        ])
        
        chain = synthesis_prompt | model.with_structured_output(SynthesisOutput)
        response = chain.invoke({"observations": observations_text})
        
        return {
            "summary": response.summary,
            "recommendations": response.recommendations,
        }
    
    workflow = StateGraph(EDAState)
    
    workflow.add_node("profile_dataset", profile_dataset_node)
    workflow.add_node("extract_observations_1", extract_observations_node)
    workflow.add_node("analyze_missingness", analyze_missingness_node)
    workflow.add_node("extract_observations_2", extract_observations_node)
    workflow.add_node("compute_aggregates", compute_aggregates_node)
    workflow.add_node("extract_observations_3", extract_observations_node)
    workflow.add_node("analyze_relationships", analyze_relationships_node)
    workflow.add_node("extract_observations_4", extract_observations_node)
    workflow.add_node("synthesize_findings", synthesize_findings_node)
    
    workflow.set_entry_point("profile_dataset")
    
    workflow.add_edge("profile_dataset", "extract_observations_1")
    workflow.add_edge("extract_observations_1", "analyze_missingness")
    workflow.add_edge("analyze_missingness", "extract_observations_2")
    workflow.add_edge("extract_observations_2", "compute_aggregates")
    workflow.add_edge("compute_aggregates", "extract_observations_3")
    workflow.add_edge("extract_observations_3", "analyze_relationships")
    workflow.add_edge("analyze_relationships", "extract_observations_4")
    workflow.add_edge("extract_observations_4", "synthesize_findings")
    workflow.add_edge("synthesize_findings", END)
    
    app = workflow.compile(checkpointer=checkpointer, name=WORKFLOW_NAME)
    
    return app

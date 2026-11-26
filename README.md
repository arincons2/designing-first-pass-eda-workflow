# EDA Workflow

An AI-powered exploratory data analysis workflow that performs consistent, first-pass analysis of datasets using LangChain and LangGraph. The workflow runs a fixed set of analysis tools, uses an LLM to extract observations after each step, and synthesizes findings into a summary with actionable recommendations.

## How It Works

The workflow follows a sequential process:
1. **Analyze**: Runs a fixed set of four predefined analysis tools on the dataset (dataset profiling, missingness analysis, aggregate statistics, and relationship analysis)
2. **Observe**: After each tool, the LLM extracts concise observations from the results
3. **Synthesize**: Once all tools have run, the LLM summarizes findings and provides actionable recommendations

This approach combines deterministic pandas-based analysis with LLM-powered interpretation.

The purpose of the workflow is a **first-pass Exploratory Data Analysis (EDA)**: quickly understanding a dataset's structure and quality, and identifying potentially interesting characteristics and relationships that merit deeper investigation.

## Analysis Tools

The workflow runs four tools in a fixed order. After each tool, an LLM step extracts observations from that tool's results; after the last one, the findings are synthesized.

```
profile_dataset → analyze_missingness → compute_aggregates → analyze_relationships → synthesize_findings
```

| # | Tool (graph node) | Results key | Purpose |
|---|-------------------|-------------|---------|
| 1 | `profile_dataset_node` | `results["profile_dataset"]` | Dataset structure: shape, columns, data types, summary statistics |
| 2 | `analyze_missingness_node` | `results["analyze_missingness"]` | Data quality: how much data is missing, and where |
| 3 | `compute_aggregates_node` | `results["compute_aggregates"]` | Descriptive summaries, overall and per category |
| 4 | `analyze_relationships_node` | `results["analyze_relationships"]` | Associations between pairs of variables |

The tools are complementary:

- **Profiling** describes *what the dataset looks like*.
- **Missingness analysis** describes *how complete it is*.
- **Aggregates** describe *how values are distributed and how they differ across groups*.
- **Relationship analysis** describes *how variables move together or are associated*.

In all tools, numeric columns are those with a numeric dtype, and categorical columns are those with an `object` or `category` dtype.

### 1. `profile_dataset_node`

Reports the number of rows and columns, column names and dtypes, the lists of numeric and categorical columns, `describe()` statistics for numeric columns, and the 10 most frequent values of each categorical column.

### 2. `analyze_missingness_node`

Reports the missing count and percentage per column, the columns with more than 20% missing values, and the number and percentage of complete rows.

### 3. `compute_aggregates_node`

Computes aggregate summaries and stores them in `results["compute_aggregates"]`.

| Result | Applies to | Contents |
|--------|-----------|----------|
| `numeric_stats` | Each numeric column | count, mean, median, std, min, first quartile (`q1`), third quartile (`q3`), max |
| `categorical_stats` | Each categorical column | unique count, missing count and %, most frequent category (name, count, %), frequencies and percentages of up to 10 categories, count of remaining values (`other_count`) |
| `grouped_stats` | Categorical columns with 2–10 unique values | For each category: row count, plus mean, median, std, min and max of every numeric column |

Details:

- Missing values are reported separately and are never counted as a category.
- Categorical columns with fewer than 2 or more than 10 categories are not used for grouping; they are listed in `skipped_grouping_columns` with the reason, and the columns actually used are listed in `grouping_columns`.
- `notes` records situations such as an empty dataset or a missing numeric/categorical column type.

**Why aggregates?** Profiling gives a general overview, but aggregation adds richer summaries: spread and quartiles for numeric columns, and category shares for categorical ones. Grouped aggregation goes further by showing how numeric measures differ between groups (for example, how the average of a numeric column differs across the categories of a categorical column), which a single dataset-wide summary can hide.

### 4. `analyze_relationships_node`

Investigates relationships between pairs of variables and stores them in `results["analyze_relationships"]`. Missing values are never imputed or treated as a category; each pair uses only its complete observations.

#### Numerical–numerical

- Computes **Pearson** (linear) and **Spearman** (rank-based, monotonic) correlation coefficients for each pair of eligible numeric columns.
- A positive coefficient means the variables tend to increase together; a negative one means one tends to decrease as the other increases.
- Reports the top 5 positive and top 5 negative pairs, `strong_pairs` (|coefficient| ≥ 0.7 for either method), and `divergent_pairs` (Pearson and Spearman differ by ≥ 0.3, which can point to non-linear relationships or outliers).
- A pair needs at least 10 paired observations; constant columns are skipped. `all_pairs` lists up to 50 pairs, ranked by absolute Pearson correlation.

#### Categorical–numerical

- For each eligible categorical column (2–10 categories) and numeric column, reports per-category **count, mean, median, min and max**, with a `small_group` flag for categories with fewer than 5 observations.
- Adds **eta-squared** (share of the numeric variable's variance explained by the category grouping, from 0 to 1) as an effect-size measure.
- Up to 20 pairs are kept, ranked by eta-squared.

#### Categorical–categorical

- For each pair of eligible categorical columns, builds a **contingency table** and **row proportions**.
- Runs a **chi-square test of independence**, reporting the statistic, **degrees of freedom**, **p-value** and whether it is significant at 0.05.
- The test is flagged `valid` only if expected counts are adequate (all ≥ 1 and at least 80% ≥ 5). When it is not valid, no p-value is reported and a reason is given.
- Reports **Cramér's V** (0 = no association, 1 = perfect association) as the strength of association.
- The top 15 pairs by Cramér's V get full detail; remaining pairs appear in `other_pairs` with only their variables, sample size and Cramér's V.

Columns and pairs that cannot be analyzed are listed in `skipped` with reasons, and `summary` counts the columns and pairs analyzed.

> **Interpretation note:** Correlation and statistical association describe patterns in the data; they do **not** establish causation. These results are exploratory and are meant to suggest what to investigate further.

**Why relationship analysis?** Examining variables one at a time cannot reveal how they relate to each other. Covering all three variable-type combinations gives a consistent view of associations and dependencies across the whole dataset.

## Why Aggregates and Relationship Analysis

Profiling and missingness analysis show what the data looks like and how complete it is, but they treat each column in isolation. `compute_aggregates_node` and `analyze_relationships_node` were added to go further:

- **Aggregate analysis** provides additional descriptive summaries and helps reveal patterns across groups.
- **Relationship analysis** investigates associations and dependencies between variables.
- The two are **complementary**: aggregates summarize *levels and differences* (what the values are, per group), while relationships quantify *association* (how strongly variables are linked, with effect sizes and, for categorical pairs, a significance test).

Together, they enrich the first-pass EDA beyond dataset profiling and missingness analysis, and give the synthesis step more material for observations and recommendations.

## Output Structure

Each tool reads the dataset from the workflow state, adds its output to the shared `results` dictionary under its own key, and sets `current_step`. The LLM observation step then reads the results for that step. All results are available after a run through `workflow.get_results()`.

**`results["compute_aggregates"]`**

```
numeric_stats              {column: {count, mean, median, std, min, q1, q3, max}}
categorical_stats          {column: {unique_count, missing_count, missing_pct, top_category,
                                     top_count, top_pct, frequencies, percentages, other_count}}
grouped_stats              {categorical_column: {category: {count, <numeric_column>: {mean, median, std, min, max}}}}
grouping_columns           [columns used for grouping]
skipped_grouping_columns   {column: reason}
notes                      [messages]
```

**`results["analyze_relationships"]`**

```
numeric_numeric          {eligible_columns, pairs_analyzed, top_positive, top_negative,
                          strong_pairs, divergent_pairs, all_pairs}     # pairs: variable_1, variable_2, n, pearson, spearman
categorical_numeric      {eligible_categorical, eligible_numeric, pairs_analyzed,
                          pairs}                                        # pairs: categorical, numeric, n, eta_squared, groups
categorical_categorical  {eligible_columns, pairs_analyzed, pairs, other_pairs}
                                                                        # pairs: contingency_table, row_proportions,
                                                                        #        chi_square, cramers_v
skipped                  {numeric_columns, categorical_columns, numeric_pairs,
                          categorical_numeric_pairs, categorical_pairs}
summary                  {counts of eligible/skipped columns and analyzed pairs}
notes                    [messages]
```

## Testing

The EDA workflow, including all four analysis tools, was successfully tested end to end using `data/cafe_sales.csv` as the source of raw data.

## Setup

### Prerequisites

- **Python 3.10 or 3.11**
- **Poetry** (dependency manager)
- **OpenAI API Key**

### Installation Steps

1. **Install Poetry** (if not already installed):
   
   **Windows (PowerShell)**:
   ```powershell
   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
   ```
   
   **macOS/Linux**:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
   
   After installation, restart your terminal. If `poetry` command is not found:
   - **Windows**: Add `%APPDATA%\Python\Scripts` to your system PATH
   - **macOS/Linux**: Add `export PATH="$HOME/.local/bin:$PATH"` to your `~/.bashrc` or `~/.zshrc`

2. **Install dependencies**:
   ```bash
   poetry install
   ```
   
   This will install all dependencies with the exact versions specified in `poetry.lock`, ensuring consistency across all environments.

3. **Set up your OpenAI API key**:
   
   **Windows**:
   ```powershell
   copy .env.example .env
   ```
   
   **macOS/Linux**:
   ```bash
   cp .env.example .env
   ```
   
   Then edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-key-here
   ```

### Multiple Python Versions?

If you have multiple Python versions installed and want to use a specific one:

```bash
# Tell Poetry which Python to use
poetry env use python3.11  # or python3.10

# Then install dependencies
poetry install
```

Poetry will create a virtual environment with your chosen Python version.

## Usage

### Python API

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from eda_workflow.eda_workflow import EDAWorkflow

load_dotenv()

# Initialize the workflow with an LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
workflow = EDAWorkflow(model=llm)

# Run analysis on a dataset
workflow.invoke_workflow("data/cafe_sales.csv")

# Retrieve results
summary = workflow.get_summary()              # str
recommendations = workflow.get_recommendations()  # list[str]
observations = workflow.get_observations()    # dict[str, list[str]]
results = workflow.get_results()              # dict
```

### Running the Example

```bash
poetry run python example_usage.py
```

This runs a full analysis on the sample dataset and prints the results for each step.

## Project Structure

```
eda-agent/
├── data/
│   └── cafe_sales.csv             # Sample dataset
├── eda_workflow/
│   ├── __init__.py
│   ├── eda_workflow.py             # Main workflow class and graph
│   └── prompts/                   # LLM prompt templates
│       ├── extract_observations_system.txt
│       ├── extract_observations_human.txt
│       ├── synthesize_findings_system.txt
│       └── synthesize_findings_human.txt
├── .env.example                   # Environment variable template
├── example_usage.py               # Example script
├── pyproject.toml                 # Dependencies configuration
├── poetry.lock                    # Locked dependency versions
└── README.md
```

**Important**: The `poetry.lock` file is committed to ensure all users get identical, tested dependency versions.
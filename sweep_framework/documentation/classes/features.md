
# Purpose

`features.yml` defines the preprocessing schema for the plan data.  It separates dynamic features (recurring workflow activity) from static features (plan-level attributes) and specifies the time window for aggregation.

This config file ensures that feature engineering is discoverable, reproducible, and extensible without editing Python code.

# Structure

The `features.yml` file acts as a blueprint for how plan data is transformed into model-ready inputs.  Instead of hard-coding feature lists or aggregate rules inside Python code, this configuration file externalizes those decisions so they can be easily changed, versioned, and documented.

At the top of the file, you'll find the `time_window` setting.  This determines the granularity of the dynamic data, i.e. whether the preprocessor should treat each row as a daily snapshot, a weekly rollup, a monthly aggregation, or a quarterly summary.  By adjusting this single paramater, you can experiment with different temporal resolutions without rewriting code.

The next section, called `feature_schema`, lists all the dynamic features that vary over time.  These are the operational metrics captured for each plan period: counts of requests and completions, average turnaround days, exception handling, and SLA breach rates.  Each entry in this list corresponds to a column in the raw input data, and together they form the sequence of vectors that feed into the recurrent model.  Expanding or contracting this list changes the dimensionality of the dynamic input, making it easy to prototype lean schemas or richer ones as experiments evolve.

Following that, the `static_schema` defines the plan-level attributes that remain constant across months.  These include rules, such as whether loans or hardships are allowed, demographics like participant counts, median balances and median ages, and other plan-level information.  These features are extracted once per plan to form a static vector that complements the dynamic sequence.  

By separating dynamic and static schemas, the framework ensures clarity in how features are used and makes it straightforward to add new plan descriptors without interfering with time-series logic.

Together, these three sections, `time window`, `feature_schema`, and `static_schema`, provide a declarative way to control preprocessing. Analysts can version the file (`features_v1.yml`, `features_v2.yml`) to track the evolution of experiments, and developers can load it directly into the `PlanPreprocessor` class to guarantee reproducibility. 

In short, `features.yml` is the contract between raw plan data and the model pipeline, making feature engineering transparent, flexible, and discoverable.

```yaml
# Time window for aggregation
time_window: monthly   # options: daily, weekly, monthly, quarterly

# Dynamic features (sequence-level, vary by period)
feature_schema:
  - termination_req
  - termination_comp
  - termination_days
  - inserv_req
  - inserv_comp
  - inserv_days
  - rollover_req
  - rollover_comp
  - rollover_days
  - hardship_req
  - hardship_comp
  - hardship_days
  - loan_req
  - loan_comp
  - loan_days
  - exceptions_opened
  - exceptions_resolved
  - sla_breach_rate

# Static features (plan-level, constant across months)
static_schema:
  - hardship_allowed
  - loan_allowed
  - inserv_allowed
  - participants
  - median_balance
  - median_age
  - fee_distribution
  - fee_loan_origination
  - vendor_type
```
# Documenting `features.yml` and Raw Input Data

The `features.yml` file is the contract between the raw plan data and the preprocessing pipeline.  It defines which features are expected and how they are organized.  To ensure reproducibility and avoid errors, two rules must always be followed.

## Raw Input Data Must Match the Time Window

The `time_window` setting in `features.yml`, e.g. `daily`, `weekly`, `monthly`, `quarterly`, determines the granularity of the time series.

* If you select `monthly`, the raw input file must contain one row per plan per month.
* If you select `weekly`, the raw input file must contain one row per plan per week.

The preprocessor does not automatically resample; its expects the raw data to already be aligned with the chosen time window.  This ensures that the dynamic sequence tensor has the correct number of timesteps and avoids mismatches between configuration and data.

## Schema and Raw Input File Must Be Consistent

The headers in the raw input file must align exactly with the fields listed in `features.yml`.  

* The `feature_schema` section defines the dynamic features, i.e.  the columns that vary by time period, and
* The `static_schema` section defines the static features, i.e. columns that remain constant across all periods for a plan.

The raw input file should be a comma-delimited tabular text file (CSV) with column names that match those in these schema entries.

For example:

```yaml
feature_schema:
  - termination_req
  - termination_comp
  - termination_days
  - loan_req
  - loan_comp
  - loan_days

static_schema:
  - hardship_allowed
  - loan_allowed
  - participants
  - median_balance
  - median_age
```
Corresponding CSV headers:

```csv
plan_id,month,termination_req,termination_comp,termination_days,loan_req,loan_comp,loan_days,hardship_allowed,loan_allowed,participants,median_balance,median_age,label
```

If a header is missing or misnamed, preprocessing will fail or silently produce incorrect tensors. Keeping the YAML and CSV in sync guarantees that experiments are reproducible and interpretable.

## Best Practice

* Version both files together: When you update features.yml, also update the raw input file headers.
* Document schema changes: Maintain a changelog explaining why features were added or removed.
* Validate alignment: Add a unit test that checks whether all YAML fields exist in the CSV headers before preprocessing begins.

# `label` Column

The raw input file must contain a `label` column.  This column provides the target classification for each plan, which the model uses during training and evaluation.

## Purpose

The label represents the efficiency outcome or class you are trying to predict.  For example, in retirement plan modeling, a label of `1` might indicate an "efficient" plan meeting SLA and operational benchmarks, while `0` indicates an "inefficient" plan.

## Format

The label should be a numeric value, commonly `1` or `0` for binary classification, where `1` represents the positive class and `0` represents the negative class.

Each row in the raw input file must include a label, even though the value is constant across all months for a given plan.

During preprocessing, the label is extracted once per plan and paired with the dynamic and static feature tensors.

## Consistency Requirement

Just like the dynamic and static features, the `label` column name must match exactly between the raw input file and the configuration and expectations of the preprocessor.  If the column is missing or misnamed, preprocessing will fail.

## Features Versus Target

The `label` column in the raw input file is not included in `features.yml`.  This is intentional.

* `features.yml` defines the input schema, the dynamic and static fields that are fed into the model.
* The `label` column represents the output target, e.g. efficient versus inefficient plan, and is handled separately during training and evaluation.

The raw input file must still include a `label` column, even though it is not listed in the YAML. Each row should carry the label value, which is constant across all periods for a given plan.

## Best Practice

* Treat the label column as the ground truth for supervised learning.
* Document the meaning of each label value (0 = inefficient, 1 = efficient, etc.) in your sweep notes.
* If you expand to multi‑class classification, update the documentation to explain each class code.

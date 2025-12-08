
# Preparing Input Data

The raw input file is the foundation of the sweep framework.  It must be structured consistently with the configuration defined in `features.yml` and prepared carefully to ensure the RNN can learn effectively.  The following requirements apply.

## File Format

* The raw input file must be a comma-delimited tabular text file (CSV).
* Column headers must align exactly with the fields listed in `features.yml`.
* A `label` column must also be present, even though it is not part of `features.yml`.  This column represents the supervised learning target.

## Time Window Alignment

The granularity of rowa must match the `time_window` specified in `features.yml`.

* `monthly` provide one row per plan per month.
* `weekly` provide one row per plan per week.
* `quarterly` provide one row per plan per quarter.

The preprocessor does not resample automatically; the raw file must already be aligned with the chosen time window.

## Data Types

Dynamic features should be numeric (integer for counts or float for rates or averages).

Static features may contain categorical values, e.g. vendor type or plan type.  These must be encoded consistently via one-hot encoding or categorical index mapping.

Label must be numeric (commonly `1` or `0` for binary classification).

## Categorical Variables

* Nominal categories (unordered) should be one-hot encoded or mapped to embeddings.
* Ordinal categories (ordered) may be encoded as integers if the order is meaningful.

Avoid encoding nominal categories as raw integers, since the model will misinterpret them as continuous values.

## Normalization

Continuous features, e.g. balances, ages, turnaround dates, rates, should be normalized before training using either Z-score standardizaton (center at zero, unit variance) or min-max scaling, scaled to [0,1].

Binary flags (0/1) should remain untouched.

Normalization parameters, e.g. mean, standard deviation, minimum, maximum, must be computed on the training set only and applied consistently to validation and test sets to avoid data leakage.

Apply normalization per feature across all timesteps, not per timestep.  There are two possible ways of scaling your data.

### Normalizing Per Timestep

Imagine you have monthly values for `termination_days`.  If you normalize each month independently, e.g. substract that month's mean and divide by that month's standard deviation, you will destroy the temporal signal.  Every month would look centered, and the RNN would lose the ability to detect trends or seasonality across time.

### Normalizing Per Feature Across All Timesteps

Instead, you compute normalization parameters once per feature across the entire training dataset.  For example, compute the mean and standard deviation of `termination_days` across all plans and all months in the training set.  Apply that same transformation to every timestep for that feature.  This preserves the relative differences between months while still putting features on comparable scales.

### Example

Suppose `termination_days` across all plans and months has a mean of `3.2` and standard deviation of `0.5`.  Normalization is calculated as: 
$$
x^{\prime} = \frac{x - 3.2}{0.5}
$$
and applied to every month's `termination_days` value.  So January = 3.5 is 0.6, February = 3.0 is -0.4, etc.  

The sequence still shows ups and downs over time, but now it's scaled consistently.

### Best Practice

* Compute normalization parameters per feature, using the training set only.
* Apply the same parameters to all timesteps (and to validation/test sets).
* This ensures the RNN sees consistent, comparable values across time and avoids data leakage.
* Never normalize each timestep independently; that erases the temporal structure you want the RNN to learn.

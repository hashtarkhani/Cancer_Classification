# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 13:26:41 2025

@author: Soheil Hashtarkhani
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
# --------------------------
# Settings
# --------------------------
FREE_TEXT_FLAG = True   # Set to True to evaluate the FreeText subset instead
N_BOOT = 2000            # Number of bootstrap iterations
ALPHA = 0.05             # For 95% CIs

# --------------------------
# Load data
# --------------------------
print("Current Working Directory:", os.getcwd())
data0 = pd.read_csv("FinalResults2.csv") # CSV file showing preicted disgnosis by each model and real diagnosis
data0.info()

# Stratify: ICD-coded (False) vs FreeText (True)
data = data0[data0['FreeText'] == FREE_TEXT_FLAG].copy()

# Ground truth and weights
true_labels_full = data['Diagnosis']
freq_full = data['Freq']

# Models to evaluate (predictions are in these columns)
models = ['GPT3.5_Clean', 'GPT4_Clean', 'Gemini_Clean', 'LLama_Clean', 'BioBert_Clean']

# --------------------------
# Metric helpers
# --------------------------
def calculate_accuracy(y_true, y_pred):
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    return (y_true == y_pred).mean()

def calculate_weighted_accuracy(y_true, y_pred, weights):
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    weights = pd.Series(weights).reset_index(drop=True)
    correct = (y_true == y_pred)
    return (correct * weights).sum() / weights.sum()

def calculate_macro_f1_weighted(y_true, y_pred, weights):
    # Macro F1 with sample weights (your frequency weighting)
    return f1_score(y_true, y_pred, average="macro",
                    sample_weight=weights, zero_division=0)

def bootstrap_ci(y_true, y_pred, weights, metric_func, n_boot=1000, alpha=0.05, random_state=42):
    """
    Generic bootstrap CI for metrics.
    metric_func should accept (y_true, y_pred, weights) where weights may be ignored internally.
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    weights = np.asarray(weights) if weights is not None else None

    n = len(y_true)
    stats = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)  # sample rows with replacement
        if weights is None:
            stats[b] = metric_func(y_true[idx], y_pred[idx], None)
        else:
            stats[b] = metric_func(y_true[idx], y_pred[idx], weights[idx])

    lower = np.percentile(stats, 100 * (alpha / 2.0))
    upper = np.percentile(stats, 100 * (1 - alpha / 2.0))
    return float(lower), float(upper)

# --------------------------
# Misclassification helper
# --------------------------
def get_misclassifications(df, model_name):
    y_true = df['Diagnosis']
    y_pred = df[model_name]
    misclassified = df[y_true != y_pred]
    mis_counts = (
        misclassified
        .groupby(['Diagnosis', model_name])
        .size()
        .reset_index(name='Count')
        .sort_values(by='Count', ascending=False)
    )
    return mis_counts

# --------------------------
# Evaluation loop
# --------------------------
evaluation_results = {}

for model in models:
    # Drop rows with missing predictions for this model to ensure clean alignment
    cur = data[['Diagnosis', 'Freq', model]].dropna(subset=[model]).copy()
    y_true = cur['Diagnosis']
    y_pred = cur[model]
    w = cur['Freq']

    # Point estimates
    acc = calculate_accuracy(y_true, y_pred)
    w_acc = calculate_weighted_accuracy(y_true, y_pred, w)
    w_macro_f1 = calculate_macro_f1_weighted(y_true, y_pred, w)

    # Bootstrap CIs
    acc_lo, acc_hi = bootstrap_ci(y_true, y_pred, None,
                                  metric_func=lambda yt, yp, _:
                                      calculate_accuracy(yt, yp),
                                  n_boot=N_BOOT, alpha=ALPHA, random_state=2025)

    wmf1_lo, wmf1_hi = bootstrap_ci(y_true, y_pred, w,
                                    metric_func=calculate_macro_f1_weighted,
                                    n_boot=N_BOOT, alpha=ALPHA, random_state=2025)

    # Store results
    evaluation_results[model] = {
        'Accuracy': acc,
        'Accuracy_CI_low': acc_lo,
        'Accuracy_CI_high': acc_hi,
        'Weighted Accuracy': w_acc,
        'Weighted Macro F1': w_macro_f1,
        'Weighted Macro F1_CI_low': wmf1_lo,
        'Weighted Macro F1_CI_high': wmf1_hi,
        'N (evaluated)': len(cur)
    }

    # Misclassification table (top 10)
    top_mis = get_misclassifications(cur, model)
    print(f"\nTop Misclassifications for {model} (FreeText={FREE_TEXT_FLAG}):")
    print(top_mis.head(10))
    print("\n" + "="*60 + "\n")

# --------------------------
# Summary table
# --------------------------
summary_df = (
    pd.DataFrame(evaluation_results)
    .T
    .loc[:, [
        'N (evaluated)',
        'Accuracy', 'Accuracy_CI_low', 'Accuracy_CI_high',
        'Weighted Accuracy',
        'Weighted Macro F1', 'Weighted Macro F1_CI_low', 'Weighted Macro F1_CI_high'
    ]]
    .sort_values(by=['Weighted Macro F1', 'Accuracy'], ascending=False)
)

# Nicely formatted view
with pd.option_context('display.float_format', '{:.4f}'.format):
    print(summary_df)

# If you also want a single string column like "0.912 (0.889–0.932)" for quick tables:
def fmt_ci(point, lo, hi):
    return f"{point:.3f} ({lo:.3f}–{hi:.3f})"

summary_pretty = summary_df.copy()
summary_pretty['Accuracy (95% CI)'] = [
    fmt_ci(r['Accuracy'], r['Accuracy_CI_low'], r['Accuracy_CI_high'])
    for _, r in summary_df.iterrows()
]
summary_pretty['Weighted Macro F1 (95% CI)'] = [
    fmt_ci(r['Weighted Macro F1'], r['Weighted Macro F1_CI_low'], r['Weighted Macro F1_CI_high'])
    for _, r in summary_df.iterrows()
]
print("\nPretty summary:")
print(summary_pretty[['N (evaluated)', 'Accuracy (95% CI)', 'Weighted Accuracy',
                      'Weighted Macro F1 (95% CI)']])


# Create a clean DataFrame for export
export_df = pd.DataFrame({
    'Model': summary_df.index,
    'Accuracy': summary_df['Accuracy'],
    'Accuracy_CI_low': summary_df['Accuracy_CI_low'],
    'Accuracy_CI_high': summary_df['Accuracy_CI_high'],
    'Weighted_Macro_F1': summary_df['Weighted Macro F1'],
    'Weighted_Macro_F1_CI_low': summary_df['Weighted Macro F1_CI_low'],
    'Weighted_Macro_F1_CI_high': summary_df['Weighted Macro F1_CI_high']
}).reset_index(drop=True)

# Optional: round for readability
export_df = export_df.round(4)

# Save to CSV
export_filename = f"evaluation_FreeText.csv"

export_df.to_csv(export_filename, index=False)


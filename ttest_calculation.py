#this code was written by chatgpt
import numpy as np
from scipy import stats

# Example performance scores from k-fold cross-validation
baseline_scores = np.array([0.80, 0.82, 0.79, 0.81, 0.83])
new_model_scores = np.array([0.85, 0.86, 0.84, 0.87, 0.88])

# Calculate performance differences
differences = new_model_scores - baseline_scores

# Perform paired t-test
t_statistic, p_value = stats.ttest_rel(new_model_scores, baseline_scores)

print(f"T-statistic: {t_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpret the result
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis. The new model performs significantly better than the baseline model.")
else:
    print("Fail to reject the null hypothesis. No significant difference in performance.")

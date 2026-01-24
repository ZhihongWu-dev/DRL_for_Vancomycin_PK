# BCQ Policy Evaluation and Analysis Documentation

## 📚 Table of Contents

1.  [Evaluation Rationale](#evaluation-rationale)
2.  [Policy Value Assessment](#policy-value-assessment)
3.  [Clinical Safety Analysis](#clinical-safety-analysis)
4.  [Evaluation Flow](#evaluation-flow)
5.  [Key Metrics and Interpretation](#key-metrics-and-interpretation)

---

## 1. Evaluation Rationale

### Why Traditional Metrics Fail in Offline RL
In Offline Reinforcement Learning (RL), standard machine learning metrics like **accuracy** or **Mean Absolute Error (MAE)** are insufficient for policy evaluation [1].

*   **Goal Discrepancy**: RL aims to maximize cumulative reward, not to predict the "correct" action.
*   **No Ground Truth**: The historical actions in the dataset are not guaranteed to be optimal.
*   **Reward Focus**: A policy is superior if it yields a higher expected cumulative reward, even if its actions differ significantly from the historical data.

### The Standard: Policy Value
The core metric for evaluating an RL policy $\pi$ is its **Policy Value** ($V^\pi$), defined as the expected discounted cumulative reward:

$$V^\pi = E[\sum_{t=0}^{\infty} \gamma^t r_t | \pi]$$

---

## 2. Policy Value Assessment

### Fitted Q Evaluation (FQE)
Since online interaction is prohibited, the policy value is estimated using **Fitted Q Evaluation (FQE)**, which leverages the learned Q-function [2].

For the continuous action space in BCQ, the policy value is estimated via Monte Carlo sampling:

$$V^\pi(s) \approx \frac{1}{N} \sum_{i=1}^{N} Q(s, a_i), \quad \text{where } a_i \sim \pi(\cdot|s)$$

The BCQ policy $\pi$ is defined by the VAE and Perturbation Actor, selecting the action $\tilde{a}^*$ that maximizes the Q-value among $K$ candidates.

### Policy Ranking Consistency
A crucial step is to compare the learned BCQ policy against baseline policies using their estimated Q-values:

| Policy | Description | Expected Q-Value Ranking |
| :--- | :--- | :--- |
| **Behavior Policy** | Actions taken directly from the dataset. | Lowest (Represents the data's inherent value) |
| **BCQ Policy** | Constrained, improved actions from the BCQ agent. | Intermediate (Improved but conservative) |
| **Greedy Policy** | Unconstrained $\arg\max Q(s, a)$ (Theoretical upper bound). | Highest (Prone to overestimation/extrapolation) |

The expected outcome is that $Q_{\text{Behavior}} < Q_{\text{BCQ}} < Q_{\text{Greedy}}$, confirming that BCQ successfully improves upon the behavior policy while remaining conservative.

### Bootstrapped Confidence Intervals (CIs)
To ensure statistical rigor, **Bootstrapped Confidence Intervals** are computed for the mean Q-values of each policy. This quantifies the uncertainty in the mean estimate and confirms that the observed policy improvement is statistically significant.

---

## 3. Clinical Safety Analysis

Given the high-stakes nature of drug dosing, evaluation must include metrics that quantify safety and clinical relevance.

### Multi-Feature Policy Comparison
The policy's response to key clinical features is analyzed by plotting the recommended dose against critical patient variables:

*   **Renal Function**: Dose vs. Creatinine, BUN.
*   **Infection Severity**: Dose vs. WBC, Temperature.
*   **Drug Concentration**: Dose vs. Vancomycin Level.

This analysis demonstrates that the BCQ policy adapts rationally to patient state, confirming its clinical utility.

### High-Risk Subgroup Analysis
The policy is analyzed on clinically defined high-risk subgroups (e.g., patients with high creatinine or a specific warning flag). Key metrics compared between the Behavior and BCQ policies include:

*   **Mean Dose**
*   **Maximum Dose**
*   **Zero-Dose Frequency**

**Expected Outcome**: BCQ should recommend a more conservative (lower mean, lower max) dose in high-risk states, demonstrating learned risk-awareness.

### Extreme Dose Reduction Metrics
Safety is quantified by measuring the frequency of extreme dose recommendations:

| Metric | Calculation | Rationale |
| :--- | :--- | :--- |
| **Zero-Dose Frequency** | Percentage of recommended doses = 0 mg. | Measures policy's tendency to withhold medication. |
| **High-Dose Frequency** | Percentage of recommended doses $\ge 1500$ mg. | Measures reduction in potentially toxic dosing. |
| **95th Percentile Dose** | $P_{95}$ of the recommended dose distribution. | Quantifies the upper bound of the policy's typical recommendation. |

---

## 4. Evaluation Flow

The evaluation process is executed fully offline using the final trained BCQ model and the complete dataset.

1.  **Load Data**: Use the full dataset (no train/validation split) to ensure comprehensive coverage.
2.  **Load Model**: Load the final BCQ agent (VAE, Actor, Q-Networks) from the saved checkpoint.
3.  **Policy Value Estimation**: Calculate $V^\pi$ for the BCQ policy using FQE (Monte Carlo sampling).
4.  **Baseline Comparison**: Calculate the actual return for the Behavior Policy from the dataset.
5.  **Safety Analysis**: Execute the multi-feature, subgroup, and extreme dose analyses.
6.  **Visualization**: Generate publication-ready figures (e.g., policy comparison charts, dose distribution histograms).

---

## 5. Key Metrics and Interpretation

| Metric | Interpretation | Desired Result |
| :--- | :--- | :--- |
| **Policy Value ($V^\pi$)** | Expected long-term clinical outcome. | $V^{\text{BCQ}} > V^{\text{Behavior}}$ |
| **Relative Improvement** | Percentage gain over the historical policy. | Positive (e.g., $> 0\%$) |
| **Zero-Dose Frequency** | Policy's decision to withhold medication. | Optimized (Lower than Behavior if unnecessary) |
| **High-Dose Frequency** | Frequency of potentially toxic doses. | Significantly lower than Behavior |
| **Q-Value Ranking** | Statistical confirmation of conservatism. | $Q_{\text{Behavior}} < Q_{\text{BCQ}} < Q_{\text{Greedy}}$ |

---

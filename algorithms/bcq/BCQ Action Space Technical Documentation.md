# Batch-Constrained Q-Learning (BCQ) - Continuous Action Space Technical Documentation

## 📚 Table of Contents

1.  [Project Overview](#project-overview)
2.  [Algorithm Principle](#algorithm-principle)
3.  [System Architecture](#system-architecture)
4.  [Model Implementation](#model-implementation)
5.  [Training Flow](#training-flow)
6.  [Technical Details](#technical-details)

---

## 1. Project Overview

### Application Context
This project focuses on optimizing vancomycin dosing in the Intensive Care Unit (ICU) by leveraging **Offline Reinforcement Learning (RL)** to derive an optimal, safe, and effective dosing strategy from historical patient data [1].

### Core Characteristics
*   **Continuous Action Space**: The drug dose (in mg) is treated as a continuous variable, allowing for fine-grained, personalized recommendations.
*   **Trajectory Structure**: Data is structured into episodes corresponding to individual patient ICU stays (`stay_id`), with a 4-hour time step resolution.
*   **Offline Learning**: The model is trained exclusively on historical data, ensuring safety by avoiding online interaction with the real environment.
*   **Constrained Policy**: **BCQ** explicitly constrains the learned policy to the support of the observed data, which is critical for safety in high-stakes medical applications.

### Technology Stack
*   **Algorithm**: Batch-Constrained Q-Learning (BCQ)
*   **Framework**: PyTorch
*   **Policy Type**: Perturbation-based constrained policy derived from an Action VAE.
*   **Q-Network**: Twin Q-Networks (Double Q-Learning) to prevent overestimation.

---

## 2. Algorithm Principle

### The Extrapolation Error Problem
Standard off-policy RL algorithms often suffer from **extrapolation error** when applied to offline data. The Q-function may assign arbitrarily high values to actions not present in the dataset, leading to a learned policy that suggests unsafe, out-of-distribution (OOD) actions [2].

### BCQ Core Solution
BCQ addresses this by **constraining the policy** to only select actions that are close to those observed in the batch. This is achieved through a three-part architecture:

1.  **Action Variational Autoencoder (ActionVAE)**: Models the distribution of actions given a state, $\pi_B(a|s)$, effectively learning the boundaries of the behavior policy.
2.  **Perturbation Actor ($\phi$)**: Learns a small, bounded adjustment ($\Delta a$) to the VAE-sampled action, allowing for policy refinement without leaving the data manifold.
3.  **Double Q-Networks ($Q_1, Q_2$)**: Used for stable and conservative value estimation.

### ActionVAE Training (Behavior Modeling)
The VAE is trained to reconstruct the batch action $a$ given the state $s$. The loss function combines two terms:

*   **Reconstruction Loss**: $MSE(a, \hat{a})$, where $\hat{a}$ is the reconstructed action from the decoder.
*   **KL Divergence Loss**: $D_{KL}(q(z|s, a) \parallel \mathcal{N}(0, I))$, which regularizes the latent space $z$ to a standard Gaussian distribution.

The total VAE loss is calculated as: $L_{VAE} = \text{Reconstruction Loss} + \beta \cdot \text{KL Loss}$, where $\beta$ is a weighting factor (e.g., 0.5).

### Critic Training (Double Q-Learning)
The Q-networks are trained using a modified Bellman target that incorporates the constraint:

1.  **Sample $K$ candidate actions** from the VAE for the next state $s'$, $a_i \sim VAE(s')$.
2.  **Perturb each candidate** using the perturbation actor: $\tilde{a}_i = a_i + \phi(s', a_i)$.
3.  **Clamp actions** to the valid range (e.g., $[-1, 1]$).
4.  **Select the best constrained action** using the target Q-networks: $\tilde{a}^* = \arg\max_i \min(Q_1^t(s', \tilde{a}_i), Q_2^t(s', \tilde{a}_i))$.
5.  **TD Target**: $y = r + \gamma \min(Q_1^t(s', \tilde{a}^*), Q_2^t(s', \tilde{a}^*))$.

The critic loss is the Mean Squared Error (MSE) between the predicted Q-values and the TD target $y$.

### Perturbation Actor Training (Policy Improvement)
The actor is trained to maximize the Q-value of the perturbed action. In implementation, this is achieved by minimizing the negative Q-value:

$$L_{\pi} = -E_{s, a \sim VAE(s)} \left[ \min(Q_1(s, a + \phi(s, a)), Q_2(s, a + \phi(s, a))) \right]$$

The input action $a$ for the actor loss is sampled from the VAE at the current state $s$.

---

## 3. System Architecture

### Network Components

| Component | Function | Input | Output |
| :--- | :--- | :--- | :--- |
| **ActionVAE** | Models behavior policy $\pi_B(a|s)$ | State $s$, Action $a$ | Latent $\mu, \log\sigma$, Reconstructed $\hat{a}$ |
| **Perturbation Actor ($\phi$)** | Learns bounded adjustment $\Delta a$ | State $s$, Action $a$ | Adjustment $\Delta a \in [-\xi, \xi]$ |
| **Q-Network ($Q_1, Q_2$)** | Estimates action-value function | State $s$, Action $a$ | Q-value $Q(s, a) \in \mathbb{R}$ |
| **Target Q-Network ($Q_1^t, Q_2^t$)** | Stabilizes training (soft update) | State $s$, Action $a$ | Target Q-value |

### Data Flow and Preprocessing
The data flow follows standard offline RL practices:

1.  **Data Source**: `ready_data.xlsx` (converted to transitions).
2.  **Preprocessing**:
    *   State features are normalized using a `StandardScaler` (fit on training data).
    *   Actions (`totalamount_mg`) are normalized to the $[-1, 1]$ range.
    *   Transitions $(s, a, r, s', done)$ are constructed based on `stay_id` and 4-hour time steps.
3.  **Training**: Batches are sampled from the replay buffer to train the VAE, Critics, and Actor sequentially.

---

## 4. Model Implementation

### ActionVAE Structure
The VAE consists of an Encoder and a Decoder, implemented as Multi-Layer Perceptrons (MLPs) with ReLU activations.

*   **Encoder**: Maps $(s, a)$ to $(\mu, \log\sigma)$.
*   **Decoder**: Maps $(s, z)$ to $\hat{a}$.
*   **Latent Space**: $z$ is sampled using the reparameterization trick: $z = \mu + \exp(\log\sigma) \cdot \epsilon$.

### Perturbation Actor
The actor is an MLP that takes the state $s$ and a VAE-sampled action $a$ as input and outputs a bounded adjustment $\Delta a$. The output is scaled by a factor $\xi$ (e.g., 0.05) using a $\tanh$ activation to ensure the perturbation is within the defined limit.

### Q-Networks
The Q-networks are standard MLPs that take the concatenated state and action $(s, a)$ as input and output a single Q-value.

---

## 5. Training Flow

The BCQ training loop involves four distinct optimization steps per batch:

| Step | Component | Loss Function | Goal |
| :--- | :--- | :--- | :--- |
| **1** | **ActionVAE** | Reconstruction (MSE) + KL Loss | Model the behavior action distribution $\pi_B(a|s)$. |
| **2** | **Q-Networks ($Q_1, Q_2$)** | MSE to Constrained TD Target | Conservatively estimate values for data-supported actions. |
| **3** | **Perturbation Actor ($\phi$)** | Negative Q-Value Loss | Refine the policy towards higher-value actions within the constraint. |
| **4** | **Target Networks** | Soft Update: $\theta^t \leftarrow \tau \theta + (1-\tau) \theta^t$ | Stabilize the training process. |

### Training Configuration (Example)
| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Batch Size** | 256 | Number of transitions sampled per step. |
| **Learning Rate (LR)** | $1 \times 10^{-4}$ | Adam optimizer learning rate. |
| **Discount Factor ($\gamma$)** | 0.99 | Balances immediate vs. future rewards. |
| **Target Update ($\tau$)** | 0.005 | Soft update coefficient for target networks. |
| **Perturbation Limit ($\xi$)** | 0.05 | Maximum allowed deviation from VAE-sampled action. |
| **VAE Samples ($K$)** | 10 | Number of candidate actions sampled for Q-target calculation. |

---

## 6. Technical Details

### Action Normalization
The raw dose action (mg) is normalized to the $[-1, 1]$ range using the minimum and maximum observed doses in the training set. This is crucial for stability, especially when the dose range is large. The final recommended dose is denormalized back to mg during inference.

### Safety and Conservatism
The BCQ architecture inherently promotes safety by:
1.  **VAE Constraint**: Only generating actions close to what was observed.
2.  **Bounded Perturbation**: Limiting the degree of policy improvement.
3.  **Double Q-Learning**: Using $\min(Q_1, Q_2)$ for conservative value estimation.

---

## References

[1] BCQ Implementation. *Manus AI Internal Documentation*.
[2] Fujimoto, S., Meger, D., & Precup, D. (2019). Off-Policy Deep Reinforcement Learning without Catastrophic Forgetting. *International Conference on Machine Learning (ICML)*.

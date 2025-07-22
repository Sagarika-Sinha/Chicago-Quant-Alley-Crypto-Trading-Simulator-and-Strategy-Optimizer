import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import math
from algorithms import *
np.random.seed(42)
def create_bandit_env(K, distribution='bernoulli'):
    if distribution == 'bernoulli':
        return np.random.rand(K)
    elif distribution == 'gaussian':
        return np.random.normal(0.5, 0.1, K)
def pull_arm(probs, arm, distribution='bernoulli'):
    if distribution == 'bernoulli':
        return np.random.binomial(1, probs[arm])
    elif distribution == 'gaussian':
        return np.random.normal(probs[arm], 1.0)
def generate_contextual_data(K, T, d):
    theta_star = np.random.rand(d)
    X = np.random.rand(K, T, d) 
    Y = np.zeros((K, T))
    for k in range(K):
        for t in range(T):
            noise = np.random.normal(0, 0.01)
            Y[k, t] = X[k, t].dot(theta_star) + noise
    return X, Y
def linucb(contexts, rewards, alpha=0.1):
    K, T, d = contexts.shape
    A = [np.identity(d) for _ in range(K)]
    b = [np.zeros(d) for _ in range(K)]
    regrets = []
    cum_regret = 0
    for t in range(T):
        x_t = contexts[:, t, :]
        p = np.zeros(K)
        for a in range(K):
            A_inv = np.linalg.inv(A[a])
            theta = A_inv @ b[a]
            p[a] = theta @ x_t[a] + alpha * np.sqrt(x_t[a] @ A_inv @ x_t[a])
        arm = np.argmax(p)
        reward = rewards[arm, t]
        A[arm] += np.outer(x_t[arm], x_t[arm])
        b[arm] += reward * x_t[arm]
        best_arm = np.argmax([rewards[a, t] for a in range(K)])
        regret = rewards[best_arm, t] - reward
        cum_regret += regret
        regrets.append(cum_regret)
    return regrets
def best_arm_selection_frequency(probs, selections, T):
    best_arm = np.argmax(probs)
    return sum([1 for arm in selections if arm == best_arm]) / T
def plot_confidence_bounds(values, counts, label):
    timesteps = np.arange(len(values[0]))
    plt.figure(figsize=(10, 6))
    for arm in range(len(values)):
        means = np.array(values[arm])
        conf = np.sqrt(2 * np.log(timesteps + 2) / (np.array(counts[arm]) + 1e-6))
        plt.plot(timesteps, means, label=f"Arm {arm} mean")
        plt.fill_between(timesteps, means - conf, means + conf, alpha=0.2)
    plt.title(f"Confidence Bounds vs Observed Rewards: {label}")
    plt.xlabel("Time Step")
    plt.ylabel("Estimated Mean Reward")
    plt.legend()
    plt.grid(True)
    plt.show()
def run_all_algorithms(K=10, T=10000, distribution='bernoulli'):
    probs = create_bandit_env(K, distribution)
    results = {
        "1. Epsilon-Greedy": epsilon_greedy(probs, T, epsilon=0.1, distribution=distribution),
        "2. Explore-Then-Commit": explore_then_commit(probs, T, exploration_rounds=100, distribution=distribution),
        "3. UCB1": ucb1(probs, T, distribution=distribution),
        "4. KL-UCB": kl_ucb(probs, T, distribution=distribution),
        "5. Thompson Sampling": thompson_sampling(probs, T, distribution=distribution),
        "6. Weighted Majority": weighted_majority(probs, T),
        "7. EXP3": exp3(probs, T, gamma=0.1),
        "9. Halving": halving(probs, T),
        "10. LUCB": lucb(probs, T),
        "11. KL-LUCB": kl_lucb(probs, T),
        "12. lil'UCB": lil_ucb(probs, T)
    }
    return results, probs
def plot_results(results):
    plt.figure(figsize=(12, 8))
    for name, regrets in results.items():
        plt.plot(regrets, label=name)
    plt.xlabel("Time Step")
    plt.ylabel("Cumulative Regret")
    plt.title("Bandit Algorithms - Cumulative Regret")
    plt.legend()
    plt.grid(True)
    plt.savefig("Bandit Algorithms.png")
if __name__ == '__main__':
    results, probs = run_all_algorithms()
    plot_results(results)
    K, T, d = 10, 1000, 5
    X, Y = generate_contextual_data(K, T, d)
    linucb_regret = linucb(X, Y, alpha=0.1)
    plt.plot(linucb_regret, label="8. LinUCB")
    plt.xlabel("Time Step")
    plt.ylabel("Cumulative Regret")
    plt.title("LinUCB - Contextual Bandit Regret")
    plt.grid(True)
    plt.legend()
    plt.savefig("LinUCB.png")

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
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
def epsilon_greedy(probs, T, epsilon=0.1, distribution='bernoulli'):
    K = len(probs)
    counts = np.zeros(K)
    values = np.zeros(K)
    regrets = []
    best_arm = np.argmax(probs)
    cum_regret = 0
    for t in range(T):
        if np.random.rand() < epsilon:
            arm = np.random.randint(K)
        else:
            arm = np.argmax(values)
        reward = pull_arm(probs, arm, distribution)
        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]
        regret = probs[best_arm] - probs[arm]
        cum_regret += regret
        regrets.append(cum_regret)
    return regrets
def explore_then_commit(probs, T, exploration_rounds=100, distribution='bernoulli'):
    K = len(probs)
    counts = np.zeros(K)
    values = np.zeros(K)
    regrets = []
    best_arm = np.argmax(probs)
    cum_regret = 0
    for t in range(T):
        if t < K * exploration_rounds:
            arm = t % K
        else:
            arm = np.argmax(values)
        reward = pull_arm(probs, arm, distribution)
        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]
        regret = probs[best_arm] - probs[arm]
        cum_regret += regret
        regrets.append(cum_regret)
    return regrets
def ucb1(probs, T, distribution='bernoulli'):
    K = len(probs)
    counts = np.zeros(K)
    values = np.zeros(K)
    regrets = []
    best_arm = np.argmax(probs)
    cum_regret = 0
    for t in range(T):
        if t < K:
            arm = t
        else:
            ucb_values = values + np.sqrt(2 * np.log(t + 1) / counts)
            arm = np.argmax(ucb_values)
        reward = pull_arm(probs, arm, distribution)
        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]
        regret = probs[best_arm] - probs[arm]
        cum_regret += regret
        regrets.append(cum_regret)
    return regrets
def kl_divergence(p, q):
    if p == 0: return (1 - p) * np.log((1 - p) / (1 - q))
    if p == 1: return p * np.log(p / q)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
def kl_ucb(probs, T, distribution='bernoulli', c=3):
    K = len(probs)
    counts = np.zeros(K)
    values = np.zeros(K)
    regrets = []
    best_arm = np.argmax(probs)
    cum_regret = 0
    for t in range(T):
        if t < K:
            arm = t
        else:
            ucb_values = []
            for i in range(K):
                mu = values[i]
                def f(q): return kl_divergence(mu, q) - np.log(t + 1) / counts[i]
                upper = minimize_scalar(lambda q: -f(q), bounds=(mu, 1), method='bounded')
                ucb_values.append(upper.x if upper.success else mu)
            arm = np.argmax(ucb_values)
        reward = pull_arm(probs, arm, distribution)
        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]
        regret = probs[best_arm] - probs[arm]
        cum_regret += regret
        regrets.append(cum_regret)
    return regrets
def thompson_sampling(probs, T, distribution='bernoulli'):
    K = len(probs)
    alpha = np.ones(K)
    beta_ = np.ones(K)
    regrets = []
    best_arm = np.argmax(probs)
    cum_regret = 0
    for t in range(T):
        samples = np.random.beta(alpha, beta_)
        arm = np.argmax(samples)
        reward = pull_arm(probs, arm, distribution)
        alpha[arm] += reward
        beta_[arm] += 1 - reward
        regret = probs[best_arm] - probs[arm]
        cum_regret += regret
        regrets.append(cum_regret)
    return regrets
def weighted_majority(probs, T, eta=0.5):
    K = len(probs)
    weights = np.ones(K)
    cum_regret = 0
    regrets = []
    best_arm = np.argmax(probs)
    for t in range(T):
        prob_dist = weights / weights.sum()
        arm = np.random.choice(K, p=prob_dist)
        reward = pull_arm(probs, arm)
        estimated_loss = (1 - reward) / prob_dist[arm]
        weights[arm] *= np.exp(-eta * estimated_loss)
        regret = probs[best_arm] - probs[arm]
        cum_regret += regret
        regrets.append(cum_regret)
    return regrets
def exp3(probs, T, gamma=0.1):
    K = len(probs)
    weights = np.ones(K)
    cum_regret = 0
    regrets = []
    best_arm = np.argmax(probs)
    for t in range(T):
        prob_dist = (1 - gamma) * (weights / weights.sum()) + gamma / K
        arm = np.random.choice(K, p=prob_dist)
        reward = pull_arm(probs, arm)
        estimated_reward = reward / prob_dist[arm]
        weights[arm] *= np.exp(gamma * estimated_reward / K)
        regret = probs[best_arm] - probs[arm]
        cum_regret += regret
        regrets.append(cum_regret)
    return regrets
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
def halving(probs, T, delta=0.1):
    K = len(probs)
    arms = list(range(K))
    regrets = []
    best_arm = np.argmax(probs)
    cum_regret = 0
    rounds = int(np.ceil(np.log2(K)))
    n = T // (rounds * len(arms))
    for _ in range(rounds):
        means = np.zeros(len(arms))
        for i, arm in enumerate(arms):
            rewards = [pull_arm(probs, arm) for _ in range(n)]
            means[i] = np.mean(rewards)
        half = len(arms) // 2
        arms = [arms[i] for i in np.argsort(means)[-half:]]
        for arm in arms:
            regret = probs[best_arm] - probs[arm]
            cum_regret += regret * n
        regrets.append(cum_regret)
    return regrets + [cum_regret] * (T - len(regrets))
def lucb(probs, T):
    K = len(probs)
    counts = np.ones(K)
    values = np.array([pull_arm(probs, i) for i in range(K)])
    regrets = []
    cum_regret = 0
    best_arm = np.argmax(probs)
    for t in range(K, T):
        ucb = values + np.sqrt(np.log(t + 1) / (2 * counts))
        lcb = values - np.sqrt(np.log(t + 1) / (2 * counts))
        a = np.argmax(values)
        b = np.argmax([ucb[i] if i != a else -np.inf for i in range(K)])
        arm = b if lcb[a] > ucb[b] else a
        reward = pull_arm(probs, arm)
        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]
        regret = probs[best_arm] - probs[arm]
        cum_regret += regret
        regrets.append(cum_regret)
    return regrets
def kl_lucb(probs, T):
    K = len(probs)
    counts = np.ones(K)
    values = np.array([pull_arm(probs, i) for i in range(K)])
    regrets = []
    cum_regret = 0
    best_arm = np.argmax(probs)
    for t in range(K, T):
        kl_ucb_values = values.copy()
        for i in range(K):
            def f(q): return kl_divergence(values[i], q) - np.log(t + 1) / counts[i]
            upper = minimize_scalar(lambda q: -f(q), bounds=(values[i], 1), method='bounded')
            kl_ucb_values[i] = upper.x if upper.success else values[i]
        a = np.argmax(values)
        b = np.argmax([kl_ucb_values[i] if i != a else -np.inf for i in range(K)])
        arm = b if values[a] - values[b] < 0.1 else a
        reward = pull_arm(probs, arm)
        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]
        regret = probs[best_arm] - probs[arm]
        cum_regret += regret
        regrets.append(cum_regret)
    return regrets
def lil_ucb(probs, T):
    K = len(probs)
    counts = np.ones(K)
    values = np.array([pull_arm(probs, i) for i in range(K)])
    regrets = []
    cum_regret = 0
    best_arm = np.argmax(probs)
    def beta(n):
        return (1 + np.sqrt(1 + 2 * np.log(n))) ** 2
    for t in range(K, T):
        bonuses = np.sqrt((2 * beta(t) * np.log(np.log(t + 1))) / counts)
        ucb_values = values + bonuses
        arm = np.argmax(ucb_values)
        reward = pull_arm(probs, arm)
        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]
        regret = probs[best_arm] - probs[arm]
        cum_regret += regret
        regrets.append(cum_regret)
    return regrets
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
    plt.savefig("plot_assign_2")
if __name__ == '__main__':
    results, probs = run_all_algorithms()
    plot_results(results)

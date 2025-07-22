# Chicago Quant Alley Crypto Trading Simulator and Strategy Optimizer
Did this project as part of Seasons of Code 2025.  
Developed a Mid-Frequency Trading simulator with backtesting and slippage tool on Delta Exchange Options in Python  
Reduced strategy simulation latency by 3.6% using a vectorized execution engine and optimizing API calls using ccxt  
Improved Sharpe ratio by 396% on simulated strategies by optimizing stop-loss & leverage on $7.90 PnL  
# Bandit Algorithms-Assignment 3

## Overview
This project implements and evaluates 12 multi-armed bandit (MAB) algorithms.

## Algorithms Implemented

### Basic & Classical Bandits
1. Epsilon-Greedy
2. Explore-Then-Commit (ETC)
3. UCB1
4. KL-UCB
5. Thompson Sampling

### Adversarial Bandits
6. Weighted Majority
7. Exp3

### Contextual Bandits
8. LinUCB (with synthetic context vectors)

### Pure Exploration / Best Arm Identification
9. Halving
10. LUCB
11. KL-LUCB
12. lil’UCB

## Experimental Setup
- Number of arms (K): 10
- Horizon (T): 10000 steps
- Reward distributions: Bernoulli (default)
- Contextual features: 5-dimensional synthetic vectors

## Evaluation Metrics
- Cumulative regret over time
- Frequency of best arm selection
- Confidence bounds vs. observed rewards

## Usage

### Requirements
- Python 3.7+
- numpy
- matplotlib
- scipy

Install dependencies using:
```bash
pip install numpy matplotlib scipy
```

### Running the Code
Simply execute the script:
```bash
python "algorithms.py"
```
```bash
python "part2.py"
```

It will output:
- Cumulative regret plots for all algorithms
- LinUCB contextual regret curve
- Optionally: confidence plots and best arm analysis

## Deliverables
- `algorithms.py`: Core script with all algorithms
- `part2.py`
- `Bandit_Assignment_Report.pdf`: Full report with analysis
- `README.md`: This file

from matplotlib import pyplot as plt
import os
import sys

sys.path.insert(0, 'src')
from ucb import ucb_normal
from ucb import asymp_ucb_normal
from ucb import moss_normal

# test with only 10 simulations
mu1 = 0
mu2s = [0.01 * d for d in range(0, 101)]
n = 1000
numsim = 10

def ucb_norm_plot(mu1, mu2s, n, numsim):
    regrets_ucb_normal = []
    regrets_asymp_ucb_normal = []
    regrets_moss_normal = []

    variance_ucb_normal = []
    variance_asymp_ucb_normal = []
    variance_moss_normal = []
    for mu in mu2s:
        ucb = ucb_normal(0, mu, n, numsim)
        asymp_ucb = asymp_ucb_normal(0, mu, n, numsim)
        moss = moss_normal(0, mu, n, numsim)
        
        regrets_ucb_normal.append(ucb[0])
        regrets_asymp_ucb_normal.append(asymp_ucb[0])
        regrets_moss_normal.append(moss[0])
        
        variance_ucb_normal.append(ucb[1])
        variance_asymp_ucb_normal.append(asymp_ucb[1])
        variance_moss_normal.append(moss[1])

    current_dir = os.getcwd()
    parent = os.path.join(current_dir, os.pardir)
    script_dir = os.path.dirname(parent)
    results_dir = os.path.join(script_dir, 'Results/')
    file_name = "ucb_normal_algs"

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    plt.style.use('seaborn')
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols = 2, figsize=(16,6))

    ax1.plot(mu2s, regrets_ucb_normal, label = r'UCB($\delta$)', color = 'red')
    ax1.plot(mu2s, regrets_asymp_ucb_normal, label = 'Asymptotically Optimal UCB', color = 'blue')
    ax1.plot(mu2s, regrets_moss_normal, label = 'MOSS', color = 'black')
    ax1.set_xlabel((r'$\mu_1 - \mu_2$'))
    ax1.set_ylabel('regret')
    ax1.set_title(r'UCB($\delta$) vs Asymptotically Optimal UCB vs MOSS bernoulli setting regrets (normal)')
    ax1.legend()

    ax2.plot(mu2s, variance_ucb_normal, label = r'UCB($\delta$)', color = 'red')
    ax2.plot(mu2s, variance_asymp_ucb_normal, label = 'Asymptotically Optimal UCB', color = 'blue')
    ax2.plot(mu2s, variance_moss_normal, label = 'MOSS', color = 'black')
    ax2.set_xlabel((r'$\mu_1 - \mu_2$'))
    ax2.set_ylabel('variance')
    ax2.set_title(r'UCB($\delta$) vs Asymptotically Optimal UCB vs MOSS bernoulli setting variance of regrets (normal)')
    ax2.legend()
    plt.savefig(results_dir + file_name)

if __name__ == "__main__":
    ucb_norm_plot(mu1, mu2s, n, numsim)
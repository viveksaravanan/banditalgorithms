from matplotlib import pyplot as plt
import os
import sys

sys.path.insert(0, 'src')
from ucb import ucb_bernoulli
from ucb import moss_bernoulli
from ucb import klucb_bernoulli

# test with only 20 simulations
mu1 = .5
mu2s = [0.01 * d for d in range(20, 81)]
n = 1000
numsim = 20

def ucb_bern_plot(mu1, mu2s, n, numsim):
    regrets_ucb_bernoulli = []
    regrets_moss_bernoulli = []
    #regrets_klucb_bernoulli = []

    variance_ucb_bernoulli = []
    variance_moss_bernoulli = []
    #variance_klucb_bernoulli = []

    for mu in mu2s:
        ucb = ucb_bernoulli(0.5, mu, n, numsim)
        moss = moss_bernoulli(0.5, mu, n, numsim)
        #klucb = klucb_bernoulli(0.5, mu, n, numsim)
        
        regrets_ucb_bernoulli.append(ucb[0])
        regrets_moss_bernoulli.append(moss[0])
        #regrets_klucb_bernoulli.append(klucb[0])
        
        variance_ucb_bernoulli.append(ucb[1])
        variance_moss_bernoulli.append(moss[1])
        #variance_klucb_bernoulli.append(klucb[1])

    current_dir = os.getcwd()
    parent = os.path.join(current_dir, os.pardir)
    script_dir = os.path.dirname(parent)
    results_dir = os.path.join(script_dir, 'Results/')
    file_name = "ucb_bern_algs"

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    
    plt.style.use('seaborn')
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols = 2, figsize=(16,6))

    ax1.plot(mu2s, regrets_ucb_bernoulli, label = r'UCB($\delta$)', color = 'red')
    #ax1.plot(mu2s, regrets_klucb_bernoulli, label = 'KL-UCB', color = 'blue')
    ax1.plot(mu2s, regrets_moss_bernoulli, label = 'MOSS', color = 'black')
    ax1.set_xlabel((r'$\mu_1 - \mu_2$'))
    ax1.set_ylabel('regret')
    ax1.legend()

    ax2.plot(mu2s, variance_ucb_bernoulli, label = r'UCB($\delta$)', color = 'red')
    #ax2.plot(mu2s, variance_klucb_bernoulli, label = 'KL-UCB', color = 'blue')
    ax2.plot(mu2s, variance_moss_bernoulli, label = 'MOSS', color = 'black')
    ax2.set_xlabel((r'$\mu_1 - \mu_2$'))
    ax2.set_ylabel('variance')
    ax2.legend()
    plt.savefig(results_dir + file_name)

if __name__ == "__main__":
    ucb_bern_plot(mu1, mu2s, n, numsim)


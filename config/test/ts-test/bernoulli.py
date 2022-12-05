from matplotlib import pyplot as plt
import os
import sys

sys.path.insert(0, 'src')
from ts import thompson_bernoulli

# test with only 20 simulations and one prior
mu1 = 0.5
mu2s = [0.01 * d for d in range(0,101)]
n = 1000
numsim = 20
priors_lst = [[(1, 1), (1, 1)]]#, [(1, 1), (1, 3)]]#, [(10, 10), (10, 10)], [(10, 10), (10, 30)]]

def ts_norm_plot(mu1, mu2s, priors, n, numsim):
    regrets = []
    variance = []

    for mu2 in mu2s:
        true_mus = [mu1, mu2]
        result = thompson_bernoulli(true_mus, priors, n, numsim)

        regrets.append(result[0])
        variance.append(result[1])

    
    current_dir = os.getcwd()
    parent = os.path.join(current_dir, os.pardir)
    script_dir = os.path.dirname(parent)
    results_dir = os.path.join(script_dir, 'Results/')
    file_name = "ts_bernoulli_" + str(priors)

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    
    plt.style.use('seaborn')
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols = 2, figsize=(16,6)) 
    prior1_label = str(priors[0])
    prior2_label = str(priors[1])
    ax1.plot(mu2s, regrets, label = 'prior: Beta' + prior1_label + ', Beta' + prior2_label)
    ax1.set_xlabel((r'$\mu_1 - \mu_2$'))
    ax1.set_ylabel('regret')
    ax1.legend()

    ax2.plot(mu2s, variance, label = r'prior: Beta' + prior1_label + ', Beta' + prior2_label)
    ax2.set_xlabel((r'$\mu_1 - \mu_2$'))
    ax2.set_ylabel('variance')
    ax2.legend()

    plt.savefig(results_dir + file_name)

def experiment():
    for priors in priors_lst:
        ts_norm_plot(mu1, mu2s, priors, n, numsim)

if __name__ == "__main__":
    experiment()
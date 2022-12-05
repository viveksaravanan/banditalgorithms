from matplotlib import pyplot as plt
import os
import sys

sys.path.insert(0, 'src')
from etc import etc_bernoulli

mu1 = 0.5
mu2s = [0.01 * d for d in range(20,81)]
ms = [25]#, 50, 75, 100, 'optimal']
n = 1000
numsim = 5

def etc_bern_plot(mu1, mu2s, m, n, numsim):
    regrets = []
    for mu2 in mu2s:
        regrets.append(etc_bernoulli(mu1, mu2, m, n, numsim))

    if m == 'optimal':
        lbl = 'optimal'
    else:
        lbl = str(m)

    current_dir = os.getcwd()
    parent = os.path.join(current_dir, os.pardir)
    script_dir = os.path.dirname(parent)
    results_dir = os.path.join(script_dir, 'Results/')
    file_name = "etc_bernoulli_" + lbl

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    plt.style.use('seaborn')
    plt.plot(mu2s, regrets, label = lbl)
    plt.xlabel
    plt.xlabel((r'$\mu_1 - \mu_2$'))
    plt.ylabel('regret')
    plt.title('explore-then-commit bernoulli setting')
    plt.legend()
    plt.savefig(results_dir + file_name)

def experiment():
    for m in ms:
        etc_bern_plot(mu1, mu2s, m, n, numsim)

if __name__ == "__main__":
    experiment()
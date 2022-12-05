from matplotlib import pyplot as plt
import os
import sys

sys.path.insert(0, 'src')
from lints import lin_ts

# test with only 10 simulations and one arm vector
vs = [0.01 * d for d in range(-20,21)]
a_s = [[0.1, -0.1]]#, [0.1, -0.2], [0.1, 0.2]]
n = 1000
numsim = 10

def lin_ts_plot(vs, a, n, numsim):
    regrets = []
    variance = []

    for v in vs:
        result = lin_ts(v, a, n, numsim)

        regrets.append(result[0])
        variance.append(result[1])

    current_dir = os.getcwd()
    parent = os.path.join(current_dir, os.pardir)
    script_dir = os.path.dirname(parent)
    results_dir = os.path.join(script_dir, 'Results/')
    file_name = "lints_" + str(a).replace(".", "_")

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    plt.style.use('seaborn')
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols = 2, figsize=(16,6)) 

    ax1.plot(vs, regrets, label = 'arm 1: ' + str(a[0]) + ', arm 2: ' + str(a[1]))
    ax1.set_xlabel((r'$\mu_1 - \mu_2$'))
    ax1.set_ylabel('regret')
    ax1.legend()

    ax2.plot(vs, variance, label = 'arm 1: ' + str(a[0]) + ', arm 2: ' + str(a[1]))
    ax2.set_xlabel((r'$\mu_1 - \mu_2$'))
    ax2.set_ylabel('variance')
    ax2.legend()

    plt.savefig(results_dir + file_name)

def experiment():
    for a in a_s:
        lin_ts_plot(vs, a, n, numsim)


if __name__ == "__main__":
    experiment()


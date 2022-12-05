import numpy as np
from matplotlib import pyplot as plt
import os


def etc_normal(mu1, mu2, m, optimal, n, numsim):
    """
    Implementation of the explore-then-commit algorithm with two arms that both follow a normal distribution. 
    
    Args:
        mu1 (int): the true mean of the first arm.
        mu2 (int): the true mean of the second arm.
        optimal (boolean): whether to calculate and use the optimal exploration rate.
        n (int): horizon 
    
    Returns:
        The average regret after 1000 simulations. 
    """

    # list used to store the regrets across simulations
    regrets = []
    for i in range(numsim):
        
        # calculate optimal exploration rate
        if optimal == True:
            if mu2 == 0:
                m = 1
            else:
                m = max(1, int((4 / mu2) * np.log((n * mu2) / 4)))
        
        # find true reward
        true_reward = max(mu1, mu2) * n
        
        # find exploration reward
        mu1_reward = mu1 * m
        mu2_reward = mu2 * m
        explore_reward = mu1_reward + mu2_reward
        
        # find empirical mean for each arm after exploring
        mu1_empirical = [sum([np.random.normal(mu1, 1) for i in range(m)]) / m]
        mu2_empirical = [sum([np.random.normal(mu2, 1) for i in range(m)]) / m]
        
        # choose arm with greater empirical mean
        if mu1_empirical > mu2_empirical:
            chosen_arm = mu1
        else:
            chosen_arm = mu2
        
        # calculate total reward after exploitation phase 
        exploit_reward = chosen_arm * (n - 2 * m)
        total_reward = explore_reward + exploit_reward
        
        # calculate regret
        regret = true_reward - total_reward
        regrets.append(regret)
    
    # find and return average regret across simulations
    expected_regret = sum(regrets) / len(regrets)
    return expected_regret

def etc_plot(mu1, mu2s, m, optimal, n, numsim):
    regrets = []
    for mu2 in mu2s:
        regrets.append(etc_normal(mu1, mu2, m, optimal, n, numsim))
    plt.style.use('seaborn')
    if optimal == True:
        lbl = 'optimal'
    else:
        lbl = str(m)
    plt.plot(mu2s, regrets, label = lbl)
    plt.xlabel
    plt.xlabel((r'$\mu_1 - \mu_2$'))
    plt.ylabel('regret')
    plt.title('explore-then-commit normal setting')
    plt.legend()
    plt.show()
    
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'Results/')
    file_name = "etc_normal_" + str(m)

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
        
    plt.savefig(results_dir + file_name)
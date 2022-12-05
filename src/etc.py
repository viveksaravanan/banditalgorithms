import numpy as np

def etc_normal(mu1, mu2, m, n, numsim):
    """
    Implementation of the explore-then-commit algorithm with two arms that both follow a normal distribution. 
    
    Args:
        mu1 (int): the true mean of the first arm.
        mu2 (int): the true mean of the second arm.
        m (int): exploration rate.
        n (int): horizon. 
        numsim (int) : number of simulations.
    
    Returns:
        The average regret after simulations. 
    """

    # list used to store the regrets across simulations
    regrets = []
    for i in range(numsim):
        
        # calculate optimal exploration rate
        if m == 'optimal':
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

def etc_bernoulli(mu1, mu2, m, n, numsim):
    """
    Implementation of the explore-then-commit algorithm with two arms that both follow a bernoulli distribution. 
    
    Args:
        mu1 (int): the true mean of the first arm.
        mu2 (int): the true mean of the second arm.
        m (int): exploration rate.
        n (int): horizon.
        numsim (int): number of simulations.
    
    Returns:
        The average regret after simulations. 
    """

    # list used to store the regrets across simulations
    regrets = []
    for i in range(numsim):
        
        # calculate optimal exploration rate
        if m == 'optimal':
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
        mu1_empirical = [sum([np.random.binomial(n = 1, p = mu1) for i in range(m)]) / m]
        mu2_empirical = [sum([np.random.binomial(n = 1, p = mu2) for i in range(m)]) / m]

        # choose arm with greater empirical mean
        if mu1_empirical > mu2_empirical:
            chosen_arm = mu1
        else:
            chosen_arm = mu2
    
        # calculate total reward after exploitation phase 
        exploit_reward = chosen_arm * (n - 2*m)
        total_reward = exploit_reward + explore_reward

        # calculate regret
        regret = true_reward - total_reward
        regrets.append(regret)
    
    # find and return average regret across simulations
    expected_regret = sum(regrets) / len(regrets)
    return expected_regret
import numpy as np
import math

def ucb_normal(mu1, mu2, n, numsim):
    """
    Implementation of the upper confidence bound algorithm with two arms that both follow a normal distribution. 
    
    Args:
        mu1 (int): the true mean of the first arm.
        mu2 (int): the true mean of the second arm.
        n (int): horizon.
        numsim (int): number of simulations.
    
    Returns:
        The average regret and regret variance after simulations. 
    """
    
    delta = 1 / n**2
    
    # list used to store the regrets across simulations
    regrets = []
    for i in range(numsim):
        
        # find true reward 
        true_reward = max(mu1, mu2) * n
        
        # initalize upper bounds both arms
        mu1_upper_bound = 0
        mu2_upper_bound = 0
        
        # pull both arms once and observe reward
        mu1_reward = np.random.normal(mu1, 1)
        mu2_reward = np.random.normal(mu2, 1)
        
        # increment arm selection by 1 for both arms
        mu1_selections, mu2_selections = 1, 1
        
        # find empirical means for both arms after pulling both
        mu1_empirical = mu1_reward / mu1_selections
        mu2_empirical = mu2_reward / mu2_selections
        
        # store total reward after pulling both arms once each
        total_reward = mu1_reward + mu2_reward
        
        for j in range(2, n):
            
            # calculate upper confidence bound for both arms
            mu1_upper_bound = mu1_empirical + math.sqrt((2 * math.log(1 / delta)) / mu1_selections)
            mu2_upper_bound = mu2_empirical + math.sqrt((2 * math.log(1 / delta)) / mu2_selections)
            
            # choose arm with higher upper bound
            if mu1_upper_bound > mu2_upper_bound:
                mu1_selections += 1
                new_sample = np.random.normal(mu1, 1)
                mu1_reward += new_sample
                total_reward += mu1
                mu1_empirical = mu1_reward / mu1_selections
            else:
                mu2_selections += 1
                new_sample = np.random.normal(mu2, 1)
                mu2_reward += new_sample
                total_reward += mu2
                mu2_empirical = mu2_reward / mu2_selections
        
        # calculate regret
        
        regret = true_reward - total_reward
        regrets.append(regret)
    
    # find and return average regret across simulations
    expected_regret = sum(regrets) / len(regrets)
    variance = np.var(regrets)
    return [expected_regret, variance]

def ucb_bernoulli(mu1, mu2, n, numsim):
    """
    Implementation of the upper confidence bound algorithm with two arms that both follow a bernoulli distribution. 
    
    Args:
        mu1 (int): the true mean of the first arm.
        mu2 (int): the true mean of the second arm.
        n (int): horizon.
        numsim (int): number of simulations.
    
    Returns:
        The average regret and regret variance after simulations. 
    """
    
    delta = 1 / n**2
    
    # list used to store the regrets across simulations
    regrets = []
    for i in range(numsim):
        
        # find true reward 
        true_reward = max(mu1, mu2) * n
        
        # initalize upper bounds both arms
        mu1_upper_bound = 0
        mu2_upper_bound = 0
        
        # pull both arms once and observe reward
        mu1_reward = np.random.normal(mu1, 1)
        mu2_reward = np.random.normal(mu2, 1)
        
        # increment arm selection by 1 for both arms
        mu1_selections, mu2_selections = 1, 1
        
        # find empirical means for both arms after pulling both
        mu1_empirical = mu1_reward / mu1_selections
        mu2_empirical = mu2_reward / mu2_selections
        
        # store total reward after pulling both arms once each
        total_reward = mu1_reward + mu2_reward
        
        for j in range(2, n):
            
            # calculate upper confidence bound for both arms
            mu1_upper_bound = mu1_empirical + math.sqrt((2 * math.log(1 / delta)) / mu1_selections)
            mu2_upper_bound = mu2_empirical + math.sqrt((2 * math.log(1 / delta)) / mu2_selections)
            
            # choose arm with higher upper bound
            if mu1_upper_bound > mu2_upper_bound:
                mu1_selections += 1
                new_sample = np.random.binomial(n = 1, p = mu1)
                mu1_reward += new_sample
                total_reward += mu1
                mu1_empirical = mu1_reward / mu1_selections
            else:
                mu2_selections += 1
                new_sample = np.random.binomial(n = 1, p = mu2)
                mu2_reward += new_sample
                total_reward += mu2
                mu2_empirical = mu2_reward / mu2_selections
        
        # calculate regret
        regret = true_reward - total_reward
        regrets.append(regret)
    
    # find and return average regret across simulations
    expected_regret = sum(regrets) / len(regrets)
    variance = np.var(regrets)
    return [expected_regret, variance]

def asymp_ucb_normal(mu1, mu2, n, numsim):
    """
    Implementation of the asymptotically optimal upper confidence bound algorithm with two arms that both follow a 
    normal distribution. 
    
    Args:
        mu1 (int): the true mean of the first arm.
        mu2 (int): the true mean of the second arm.
        n (int): horizon.
        numsim (int): number of simulations.
    
    Returns:
        The average regret and regret variance after simulations. 
    """
    
    # list used to store the regrets across simulations
    regrets = []
    for i in range(numsim):
        
        # find true reward 
        true_reward = max(mu1, mu2) * n
        
        # initalize upper bounds both arms
        mu1_upper_bound = 0
        mu2_upper_bound = 0
        
        # pull both arms once and observe reward
        mu1_reward = np.random.normal(mu1, 1)
        mu2_reward = np.random.normal(mu2, 1)
        
        # increment arm selection by 1 for both arms
        mu1_selections, mu2_selections = 1, 1
        
        # find empirical means for both arms after pulling both
        mu1_empirical = mu1_reward / mu1_selections
        mu2_empirical = mu2_reward / mu2_selections
        
        # store total reward after pulling both arms once each
        total_reward = mu1_reward + mu2_reward
        
        for j in range(2, n):
            
            # calculate upper confidence bound for both arms
            mu1_upper_bound = mu1_empirical + np.sqrt((2 * np.log(1 + j * np.log(j)**2)) / mu1_selections)
            mu2_upper_bound = mu2_empirical + np.sqrt((2 * np.log(1 + j * np.log(j)**2)) / mu2_selections)
            
            # choose arm with higher upper bound
            if mu1_upper_bound > mu2_upper_bound:
                mu1_selections += 1
                new_sample = np.random.normal(mu1, 1)
                mu1_reward += new_sample
                total_reward += mu1
                mu1_empirical = mu1_reward / mu1_selections
            else:
                mu2_selections += 1
                new_sample = np.random.normal(mu2, 1)
                mu2_reward += new_sample
                total_reward += mu2
                mu2_empirical = mu2_reward / mu2_selections
        
        # calculate regret
        regret = true_reward - total_reward
        regrets.append(regret)
    
    # find and return average regret across simulations
    expected_regret = sum(regrets) / len(regrets)
    variance = np.var(regrets)
    return [expected_regret, variance]

def moss_normal(mu1, mu2, n, numsim):
    """
    Implementation of the MOSS algorithm with two arms that both follow a normal distribution. 
    
    Args:
        mu1 (int): the true mean of the first arm.
        mu2 (int): the true mean of the second arm.
        n (int): horizon.
        numsim (int): number of simulations.
    
    Returns:
        The average regret and regret variance after simulations. 
    """
    
    # list used to store the regrets across simulations
    regrets = []
    for i in range(numsim):
        
        # find true reward 
        true_reward = max(mu1, mu2) * n
        
        # initalize upper bounds both arms
        mu1_upper_bound = 0
        mu2_upper_bound = 0
        
        # pull both arms once and observe reward
        mu1_reward = np.random.normal(mu1, 1)
        mu2_reward = np.random.normal(mu2, 1)
        
        # increment arm selection by 1 for both arms
        mu1_selections, mu2_selections = 1, 1
        
        # find empirical means for both arms after pulling both
        mu1_empirical = mu1_reward / mu1_selections
        mu2_empirical = mu2_reward / mu2_selections
        
        # store total reward after pulling both arms once each
        total_reward = mu1_reward + mu2_reward
        
        for j in range(2, n):
            
            # calculate upper confidence bound for both arms
            mu1_upper_bound = mu1_empirical + np.sqrt(((4 / mu1_selections) * np.log(max(1, n / (2 * mu1_selections)))))
            mu2_upper_bound = mu2_empirical + np.sqrt(((4 / mu2_selections) * np.log(max(1, n / (2 * mu2_selections)))))
            
            # choose arm with higher upper bound
            if mu1_upper_bound > mu2_upper_bound:
                mu1_selections += 1
                new_sample = np.random.normal(mu1, 1)
                mu1_reward += new_sample
                total_reward += mu1
                mu1_empirical = mu1_reward / mu1_selections
            else:
                mu2_selections += 1
                new_sample = np.random.normal(mu2, 1)
                mu2_reward += new_sample
                total_reward += mu2
                mu2_empirical = mu2_reward / mu2_selections
        
        # calculate regret
        regret = true_reward - total_reward
        regrets.append(regret)
    
    # find and return average regret across simulations
    expected_regret = sum(regrets) / len(regrets)
    variance = np.var(regrets)
    return [expected_regret, variance]

def moss_bernoulli(mu1, mu2, n, numsim):
    """
    Implementation of the MOSS algorithm with two arms that both follow a bernoulli distribution. 
    
    Args:
        mu1 (int): the true mean of the first arm.
        mu2 (int): the true mean of the second arm.
        n (int): horizon 
        numsim(int): number of simulations
    
    Returns:
        The average regret and regret variance after simulations. 
    """
    
    # list used to store the regrets across simulations
    regrets = []
    for i in range(numsim):
        
        # find true reward 
        true_reward = max(mu1, mu2) * n
        
        # initalize upper bounds both arms
        mu1_upper_bound = 0
        mu2_upper_bound = 0
        
        # pull both arms once and observe reward
        mu1_reward = np.random.normal(mu1, 1)
        mu2_reward = np.random.normal(mu2, 1)
        
        # increment arm selection by 1 for both arms
        mu1_selections, mu2_selections = 1, 1
        
        # find empirical means for both arms after pulling both
        mu1_empirical = mu1_reward / mu1_selections
        mu2_empirical = mu2_reward / mu2_selections
        
        # store total reward after pulling both arms once each
        total_reward = mu1_reward + mu2_reward
        
        for j in range(2, n):
            
            # calculate upper confidence bound for both arms
            mu1_upper_bound = mu1_empirical + np.sqrt(((4 / mu1_selections) * np.log(max(1, n / (2 * mu1_selections)))))
            mu2_upper_bound = mu2_empirical + np.sqrt(((4 / mu2_selections) * np.log(max(1, n / (2 * mu2_selections)))))
            
            # choose arm with higher upper bound
            if mu1_upper_bound > mu2_upper_bound:
                mu1_selections += 1
                new_sample = np.random.binomial(n = 1, p = mu1)
                mu1_reward += new_sample
                total_reward += mu1
                mu1_empirical = mu1_reward / mu1_selections
            else:
                mu2_selections += 1
                new_sample = np.random.binomial(n = 1, p = mu2)
                mu2_reward += new_sample
                total_reward += mu2
                mu2_empirical = mu2_reward / mu2_selections
        
        # calculate regret
        regret = true_reward - total_reward
        regrets.append(regret)
    
    # find and return average regret across simulations
    expected_regret = sum(regrets) / len(regrets)
    variance = np.var(regrets)
    return [expected_regret, variance]

def klucb_bernoulli(mu1, mu2, n, numsim):
    """
    Implementation of the KL-UCB algorithm with two arms that both follow a bernoulli distribution. 
    
    Args:
        mu1 (int): the true mean of the first arm.
        mu2 (int): the true mean of the second arm.
        n (int): horizon 
        numsim(int): number of simulations
    
    Returns:
        The average regret and regret variance after simulations. 
    """
    
    # list used to store regrets across simulations
    regrets = []
    for _ in range(100):
        
        # initalize upper bounds for both arms
        mu1_upper_bound, mu2_upper_bound = 0, 0
        
        # pull both arms and observe rewards
        mu1_reward = np.random.binomial(n=1,p=mu1)
        mu2_reward = np.random.binomial(n=1,p=mu2)
        
        # increment arm selection for both arms
        mu1_selections, mu2_selections = 1, 1

        # find empirical means for both arms after pulling both
        mu1_empirical = mu1_reward / mu1_selections
        mu2_empirical = mu2_reward / mu2_selections
        
        # store total reward after pulling both arms once each
        total_reward = mu1_reward + mu2_reward
        
        # find true reward
        true_reward = max(mu1, mu2) * n

        for t in range(2, n):
            mu1_bound = np.log(1 + t * np.log(np.log(t))) / mu1_selections
            mu2_bound = np.log(1 + t * np.log(np.log(t))) / mu2_selections
            bounds = np.array([mu1_bound, mu2_bound])

            # use binary search to find and compute kl-divergence < bound
            p = np.array([mu1_empirical, mu2_empirical])
            qmin = p
            qmax = np.ones(p.size)
            for _ in range(16):  # Error bounded by 2^-16.
                q = (qmax + qmin) / 2
                ndx = (np.where(p > 0, p * np.log(p / q), 0) + np.where(p < 1, (1 - p) * np.log((1 - p) / (1 - q)), 0)) < bounds
                qmin[ndx] = q[ndx]
                qmax[~ndx] = q[~ndx]

            chosen = np.argmax(q)
            
            # choose arm with higher bound
            if chosen == 0:
                mu1_selections += 1
                new_sample = np.random.binomial(n = 1, p = mu1)
                mu1_reward += new_sample
                total_reward += mu1
                mu1_empirical = mu1_reward / mu1_selections
            else:
                mu2_selections += 1
                new_sample = np.random.binomial(n = 1, p = mu2)
                mu2_reward += new_sample
                total_reward += mu2
                mu2_empirical = mu2_reward / mu2_selections
                
        # calculate regret
        regret = true_reward - total_reward
        regrets.append(regret)
        
    # find and return average regret across simulations
    expected_regret = sum(regrets) / len(regrets)
    variance = np.var(regrets)
    return [expected_regret, variance]
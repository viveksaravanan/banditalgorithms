import numpy as np

def lin_ts(v, a, n, numsim):
    """
    Implementation of Thompson Sampling applied to linear bandits.
    
    Args:
        v(float): unknown theta.
        a: list of feature vectors for both arms.
        n(int): horizon
        numsim(int): number of simulations
        
    Returns:
        The average regret and regret variance after simulations. 
    """
    
    # list used to store regrets across simulations
    regrets = []
    for _ in range(numsim):
        
        # set theta, sigma to prior
        theta, sigma = 0, 1
        
        # find true reward
        arm1 = a[0] * v
        arm2 = a[1] * v
        maxarm = max(arm1, arm2)
        true_reward = maxarm * n
        
        # initialize vectors
        total_reward = 0.0
        selections = [0,0]

        #pull first arm before starting algorithm
        a1_reward = a[0] * v
        selections[0] += 1
        theta = 1/(1/sigma + a[0] * a[0]) * (((1/sigma) * theta) + a1_reward*a[0])
        sigma = 1/(1/sigma + a[0] * a[0])

        #pull second arm
        a2_reward = a[1] * v 
        selections[1] += 1
        theta = 1/(1/sigma + a[1] * a[1]) * (((1/sigma) * theta) + a2_reward*a[1])
        sigma = 1/(1/sigma + a[1] * a[1])

        for i in range(2, n):
            
            # sample from distribution
            x = np.random.normal(theta, sigma)
            
            # choose greater arm
            arms = [np.inner(x, a[0]), np.inner(x, a[1])]
            pulled = np.argmax(arms)
            reward = a[pulled] * v 
            total_reward += reward
            selections[pulled] += 1
            theta = 1/(1/sigma + a[pulled] * a[pulled]) * (((1/sigma) * theta) + reward*a[pulled])
            sigma = 1/(1/sigma + a[pulled] * a[pulled])
        
        # calculate regret
        regret = true_reward - total_reward
        regrets.append(regret)
        
    # find and return average regret and variance across simulations
    expected_regret = sum(regrets) / len(regrets)
    variance = np.var(regrets)
    return [expected_regret, variance]
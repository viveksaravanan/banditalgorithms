import numpy as np

def lin_ucb(v, a, n, numsim):
    """
    Implementation of the UCB algorithm applied to linear bandits.
    
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
        
        # set delta and lambda values
        delta = 1/n
        lmbda = 0.1
        d = 1

        # initialize vectors
        b = 0
        theta = 0
        V = lmbda
        total_reward = 0
        
        # find true reward
        arm1_exp_reward = a[0] * v
        arm2_exp_reward = a[1] * v
        maxarm = max(arm1_exp_reward, arm2_exp_reward)
        true_reward = maxarm * n

        for t in range(1, n + 1):
            arms = [0, 0]
            beta = np.sqrt(lmbda) + np.sqrt(2 * np.log(1 / delta) + d * np.log(1 + (t - 1) / (delta * d)))

            # choose arm with higher bound
            for i in range(len(arms)):
                arms[i] = a[i] * theta + beta * np.sqrt(a[i] * a[i] * (1/V))
            if arms[0] == arms[1]:
                pulled = np.random.choice(len(arms)) # arbritarily choose arms if equal
            else:
                pulled = np.argmax(arms)

            output = a[pulled] * v + np.random.normal(0, 1) # equation to find reward
            total_reward += output

            # updates 
            V = V + a[pulled] * a[pulled]
            b = b + output * a[pulled]
            theta = (1/V) * b
            
        # calculate regret
        regret = true_reward - total_reward
        regrets.append(regret)
    
    # find and return average regret and variance across simulations
    expected_regret = sum(regrets) / len(regrets)
    variance = np.var(regrets)
    return [expected_regret, variance]
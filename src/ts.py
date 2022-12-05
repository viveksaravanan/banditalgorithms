import numpy as np
import math

def posterior_normal(mu, prior_var, sig_var, reward):
    """
    Helper function for normal thompson sampling to update the find the posterior distribution.
    
    Args:
        mu (float): prior mean.
        prior_var(int): prior variance of given mu.
        sign_var(int): signal variance of given mu.
        reward(float): reward from previous round.
    
    Returns:
        posterior mean and variance.
    """
    
    # from equation to find posterior distribution
    mu1_prior = (mu / prior_var + reward / sig_var) / (1 / prior_var + 1 / sig_var)
    mu1_prior_var = (1 / sig_var + 1 / prior_var)**-1
    return mu1_prior, mu1_prior_var

def thompson_normal(true_mus, priors, n, numsim):
    """
    Implementation of the Thompson Sampling algorithm with two arms that both follow a normal distribution. 
    
    Args:
        true_mus: list with two tuples that includes both means and their respective variances.
        priors: list with two tuples that includes both prior means and their respective variances.
        n(int): horizon.
        numsim(int): number of simulations.
        
    Returns:
        The average regret and regret variance after simulations. 
    """
    
    # list used to store regrets across simulations
    regrets = []
    for _ in range(numsim):
        
        # store true means and variances
        mu1, mu1_sig_var = true_mus[0]
        mu2, mu2_sig_var = true_mus[1]
        
        # store prior means and variances
        mu1_prior, mu1_prior_var = priors[0]
        mu2_prior, mu2_prior_var = priors[1]
        
        # intialize and set individual arm reward and total reward to 0
        mu1_reward, mu2_reward, total_reward = 0, 0, 0
        
        # initialize and set individual arm selections to 0
        mu1_selections, mu2_selections = 0, 0
        
        # find true reward
        true_reward = max(mu1, mu2) * n 
        for t in range(n):
            
            # sample from prior distributions for both arms
            mu1_sample = np.random.normal(mu1_prior, mu1_prior_var)
            mu2_sample = np.random.normal(mu2_prior, mu2_prior_var)
            
            # choose which arm to pull
            if mu1_sample > mu2_sample:
                mu1_reward = np.random.normal(mu1, mu1_sig_var)
                total_reward += mu1
                mu1_prior, mu1_prior_var = posterior_normal(mu1_prior, mu1_prior_var, mu1_sig_var, mu1_reward)
                mu1_selections += 1
            else:
                mu2_reward = np.random.normal(mu2, mu2_sig_var)
                total_reward += mu2
                mu2_prior, mu2_prior_var = posterior_normal(mu2_prior, mu2_prior_var, mu2_sig_var, mu2_reward)
                mu2_selections += 1
                
        # calculate regret
        regret = true_reward - total_reward
        regrets.append(regret)
    
    # find and return average regret and variance across simulations
    expected_regret = sum(regrets) / len(regrets)
    variance = np.var(regrets)
    return [expected_regret, variance]

def posterior_bernoulli(alpha, beta, reward):
    """
    Helper function for bernoulli thompson sampling to update the find the posterior distribution.
    
    Args:
        alpha: prior alpha value.
        beta: prior beta value.
        reward(float): reward from previous round.
    
    Returns:
        posterior alpha and beta.
    """
    
    # from equation to find posterior distribution
    alpha = alpha + reward
    beta = beta + 1 - reward
    return alpha, beta

def thompson_bernoulli(true_mus, priors, n, numsim):
    """
    Implementation of the Thompson Sampling algorithm with two arms that both follow a bernoulli distribution. 
    
    Args:
        true_mus: list that includes the means of both arms.
        priors: list with two tuples that includes both prior alpha and beta values.
        n(int): horizon.
        numsim(int): number of simulations.
        
    Returns:
        The average regret and regret variance after simulations. 
    """
    
    # list used to store regrets across simulations
    regrets = []
    for _ in range(numsim):
        
        # store true means
        mu1, mu2 = true_mus
        
        # store prior alpha and beta values
        alpha1, beta1 = priors[0]
        alpha2, beta2 = priors[1]
        
        # intialize and set individual arm reward and total reward to 0
        mu1_reward, mu2_reward, total_reward = 0, 0, 0
        
        # initialize and set individual arm selections to 0
        mu1_selections, mu2_selections = 0, 0
        
        # find true reward
        true_reward = max(mu1, mu2) * n
        for t in range(n):
            
            # sample from prior distributions for both arms
            mu1_sample = np.random.beta(alpha1, beta1)
            mu2_sample = np.random.beta(alpha2, beta2)
            
            # choose which arm to pull
            if mu1_sample > mu2_sample:
                mu1_reward = np.random.binomial(n = 1, p = mu1)
                total_reward += mu1
                alpha1, beta1 = posterior_bernoulli(alpha1, beta1, mu1_reward)
                mu1_selections += 1
            else:
                mu2_reward = np.random.binomial(n = 1, p = mu2)
                total_reward += mu2
                alpha2, beta2 = posterior_bernoulli(alpha2, beta2, mu2_reward)
                mu2_selections += 1
                
        # calculate regret
        regret = true_reward - total_reward
        regrets.append(regret)
    
    # find and return average regret and variance across simulations
    expected_regret = sum(regrets) / len(regrets)
    variance = np.var(regrets)
    return [expected_regret, variance]
import numpy as np


def entropyAction(learner, state):
    """
    Select an action return the entropy, over a single sample or multiple
    """
    policy = learner.samplePolicy(state)
    entropy = -np.dot(policy, np.log(policy).T)
    ## When cast to np.float64 this can cause errors in np multinomial
    ## Normalize if this is larger than 1 due to rounding
    policy.astype(np.float64)
    pol_sum = np.sum(policy)
    if pol_sum > 1.0:
        policy = policy / pol_sum
    try:        
        action = np.random.multinomial(1, policy[0])
        action = np.argmax(action)
    except:
        action = np.argmax(policy[0])

    return action, entropy

def QBC_KL(learner, state):
    """
    Uses multiple passes through a dropout NN as an approximation for
    multiple hypotheses sampled from a Gaussian
    
    The policy should be the probabilities of selecting from discrete actions, 
    calculates the KL divergence between the committee members
    """
    #TODO clean this up so there is no concern with shape casting, make it so that
    # for arbitrary number of committee members and state space this works
    # import ipdb; ipdb.set_trace()
    sample = learner.samplePolicy(state, batch=32, apply_dropout=True)

    # need the average probability of each action
    # The consensus probability, average probability of each action
    p_consensus = np.sum(sample, axis=0)/sample.shape[0]
    log_diff = np.log(sample) - np.log(p_consensus)
    
    # Compute the KL divergence of each committee member to the consensus probability
    kl_div = np.sum(log_diff * sample, axis=1)
    # Average over the committee members to get the average divergence
    avg_kl = np.sum(kl_div) / sample.shape[0]
    
    policy_sum = np.sum(p_consensus)
    if policy_sum > 1.0:
        p_consensus = p_consensus / policy_sum

    action = np.argmax(p_consensus)
    # try:
    #     action = np.random.multinomial(1, p_consensus)
    #     action = np.argmax(action)
    # except:
    #     action = np.argmax(policy)
        
    # Return the average divergence along with the sampled action from the consensus        
    return action, avg_kl

def varianceAction(learner, state):
    """
    Select an action based on multiple forward passes through the network
    Return the variance over the selected action
    """
    batch = 32
    action, action_var = learner.uncertainAction(state, batch=batch)
    return action, action_var  
    

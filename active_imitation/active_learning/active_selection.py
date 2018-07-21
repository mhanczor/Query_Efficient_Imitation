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
        
    # Return the average divergence along with the sampled action from the consensus        
    return action, avg_kl

def QBC_JSD(learner, state):
    """
    Uses multiple passes through a dropout NN as an approximation for
    multiple hypotheses sampled from a Gaussian
    
    Use Jensen-Shanon Divergence as a metric for determining which samples to select for learning
    When using log_2, JS Divergence is guaranteed 0 <= JSD <= 1
    """
    
    policy = learner.samplePolicy(state, batch=32, apply_dropout=True)
    n = policy.shape[0]
    
    try:
        policy_entropy = -np.sum(policy*np.log2(policy + 1e-8), axis=1, keepdims=True) # Added for stability
        
        weighted_policy = (1./n) * np.sum(policy, axis=0, keepdims=True)
        weighted_entropy = -np.sum(weighted_policy*np.log2(weighted_policy + 1e-8), axis=1, keepdims=True)
        
        JSD =  weighted_entropy - (1./n) * np.sum(policy_entropy, axis=0, keepdims=True)
        JSD = JSD.squeeze()
    except:
        JSD = 0.
        
    if JSD < 0:
        JSD = -1.
    
    policy_avg = np.mean(policy, axis=0, keepdims=True)
    policy_sum = np.sum(policy_avg)
    if policy_sum > 1.0:
        policy_avg = policy_avg / policy_sum

    action = np.argmax(policy_avg, axis=1) # Greedily take actions according to the current policy

    # Return the average divergence along with the sampled action from the consensus        
    return action, JSD
    
def varianceAction(learner, state):
    """
    Select an action based on multiple forward passes through the network
    Return the variance over the selected action
    """
    assert not learner.concrete
    
    batch = 32
    action, per_action_var = learner.uncertainAction(state, batch=batch)
    # Assume independence between actions in the action space, sum variances
    action_var = np.sum(per_action_var)
    return action, action_var  
    
def concreteUncertainty(learner, state):
    """
    Using concrete dropout, calculate the epistemic uncertainty of the model
    """
    assert learner.concrete # Need to be in concrete mode
    batch = 32
    action, per_action_var = learner.uncertainAction(state, batch=32)
    # variance norm #TODO, decite if this is the way to go?
    action_var = np.linalg.norm(per_action_var)
    
    return action, action_var
    
    
    
    
    
    


from active_imitation.utils import AggBuffer
from active_imitation.learners.active import Hindsight_DAgger, Efficient_DAgger
from active_imitation.active_learning import entropyAction, QBC_KL, varianceAction
from active_imitation.learners import DAgger

MODES  = {'pool':   Hindsight_DAgger,
          'stream': Efficient_DAgger,
          'none':   DAgger}
          # 'random_stream': Random_DAgger}
          # 'random_pool'
     
DEFAULT_PARAMS = {'random_sample': False,
                  'mixing': 1.0}
                  
POOL_DEFAULT = {'action_selection':varianceAction}

STREAM_DEFAULT = {}

#TODO Don't have env_dims and envs as arguments, gotta be a better way
def configure_robot(env, env_dims, agent, expert, mode, param_mods=None):
    
    agg_buffer = AggBuffer(env_dims, continuous=True)
    params = DEFAULT_PARAMS
    
    if mode in MODES:
        trainer = MODES[mode]
    else:
        raise ValueError('No learning mode of type {} found'.format(mode))
    
    if mode == 'pool':
        params.update(POOL_DEFAULT)
    elif mode == 'stream':
        params.update(STREAM_DEFAULT)
    elif mode == 'random':
        params
    
    # Modify any of the default parameters
    if param_mods != None:
        for key in param_mods.keys():
            params[key] = param_mods[key]
    
    learning_mode = trainer(env,
                            agent,
                            expert,
                            agg_buffer=agg_buffer,
                            continuous=True,
                            **params)
    return learning_mode
    
    # learner = Efficient_DAgger(env, 
    #                           agent, 
    #                           expert, 
    #                           mixing=1.0, 
    #                           certainty_thresh=var_thresh)
    
    
    # learner = DAgger(env, agent, expert, mixing=0.0)

    # learner = Entropy_DAgger(env, 
    #                         agent, 
    #                         expert, 
    #                         mixing=0.0, 
    #                         certainty_thresh=var_thresh)
    # learner = Random_DAgger(env,
    #                         agent,
    #                         expert,
    #                         mixing=1.0,
    #                         certainty_thresh=var_thresh)
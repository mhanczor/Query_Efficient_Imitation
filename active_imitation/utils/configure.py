
from active_imitation.utils import AggBuffer
# from active_imitation.learners.active import Hindsight_DAgger, Efficient_DAgger
from active_imitation.active_learning import entropyAction, QBC_KL, QBC_JSD, varianceAction, concreteUncertainty, QBC_JSD_Single
# from active_imitation.learners import DAgger
from active_imitation.learners import DAgger, Hindsight_DAgger, Efficient_DAgger

MODES  = {'pool':   Hindsight_DAgger,
          'stream': Efficient_DAgger, 
          'classic': DAgger}
          # 'random_stream': Random_DAgger}
          # 'random_pool'
     
DEFAULT_PARAMS = {'mixing': 1.0}
                  
POOL_DEFAULT = {'random_sample': False,
                'action_selection':varianceAction}

STREAM_DEFAULT = {'random_sample': False}

#TODO Don't have env_dims and envs as arguments, gotta be a better way
def configure_robot(env, env_dims, agent, expert, mode, continuous, concrete, param_mods=None):
    
    agg_buffer = AggBuffer(env_dims, continuous=True)
    params = DEFAULT_PARAMS
    
    if mode in MODES:
        trainer = MODES[mode]
    else:
        raise ValueError('No learning mode of type {} found'.format(mode))
    
    if mode == 'pool':
        params.update(POOL_DEFAULT)
        if not continuous:
            if 'isSpace' in param_mods:
                if param_mods['isSpace']:
                    params['action_selection'] = QBC_JSD
            else:
                params['action_selection'] = QBC_JSD_Single#QBC_KL
        elif concrete:
            params['action_selection'] = concreteUncertainty
    elif mode == 'stream':
        params.update(STREAM_DEFAULT)
    
    # Modify any of the default parameters
    if param_mods != None:
        for key in param_mods.keys():
            params[key] = param_mods[key]
    
    learning_mode = trainer(env,
                            agent,
                            expert,
                            agg_buffer=agg_buffer,
                            continuous=continuous,
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
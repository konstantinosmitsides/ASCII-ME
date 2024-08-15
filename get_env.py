
from qdax import environments, environments_v1

def get_env(env_name):
    if env_name == "hopper_uni":
        episode_length = 1000
        
        env = environments_v1.create(env_name, episode_length=episode_length)
    elif env_name == "halfcheetah_uni":
        episode_length = 1000

        env = environments_v1.create(env_name, episode_length=episode_length)
        
    elif env_name == "walker2d_uni":
        episode_length = 1000

        env = environments_v1.create(env_name, episode_length=episode_length)	
    elif env_name == "ant_uni":
        episode_length = 1000

        env = environments_v1.create(env_name, episode_length=episode_length, use_contact_forces=False, exclude_current_positions_from_observation=True)
    elif env_name == "humanoid_uni":
        episode_length = 1000

        env = environments_v1.create(env_name, episode_length=episode_length, exclude_current_positions_from_observation=True)	
    '''
    elif env_name == "ant_omni":
        episode_length = 250
        max_bd = 30.

        env = environments.create(env_name, episode_length=episode_length, use_contact_forces=False, exclude_current_positions_from_observation=False)	
    elif env_name == "humanoid_uni":
        episode_length = 1000
        max_bd = 1.

        env = environments.create(env_name, episode_length=episode_length)	
    else:
        ValueError(f"Environment {env_name} not supported.")
    '''
    return env
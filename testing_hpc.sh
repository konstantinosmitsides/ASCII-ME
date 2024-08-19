python main.py algo=ppoish_me seed=$RANDOM env=ant_uni env.episode_length=1000 num_iterations=51 algo.NORMALIZE_ENV=False algo.init_lecun=False algo.sample_trajectory=False algo.buffer_sample_batch_size=128 algo.grad_steps=16
python main.py algo=ppoish_me seed=$RANDOM env=ant_omni env.episode_length=1000 num_iterations=51 algo.NORMALIZE_ENV=False algo.init_lecun=False algo.sample_trajectory=False algo.buffer_sample_batch_size=128 algo.grad_steps=16

#python main.py algo=ppoish_me seed=$RANDOM env=ant_uni env.episode_length=1000 num_iterations=501 algo.NORMALIZE_ENV=False algo.init_lecun=True
#python main.py algo=ppoish_me seed=$RANDOM env=ant_uni env.episode_length=1000 num_iterations=21 algo.NORMALIZE_ENV=True

#python main.py algo=ppoish_me seed=$RANDOM env=walker2d_uni env.episode_length=1000 num_iterations=501 algo.NORMALIZE_ENV=False algo.init_lecun=False
#python main.py algo=ppoish_me seed=$RANDOM env=walker2d_uni env.episode_length=1000 num_iterations=501 algo.NORMALIZE_ENV=False algo.init_lecun=True
#python main.py algo=ppoish_me seed=$RANDOM env=walker2d_uni env.episode_length=1000 num_iterations=21 algo.NORMALIZE_ENV=True

#python main.py algo=ppoish_me seed=$RANDOM env=ant_omni env.episode_length=1000 num_iterations=501 algo.NORMALIZE_ENV=False algo.init_lecun=True
#python main.py algo=me seed=$RANDOM env=ant_omni env.episode_length=1000 num_iterations=700 policy_hidden_layer_sizes=[64,64]
#python main.py algo=ppoish_me seed=$RANDOM env=ant_omni env.episode_length=250 num_iterations=501 algo.NORMALIZE_ENV=True

#python main.py algo=ppoish_me seed=$RANDOM env=anttrap_omni env.episode_length=250 num_iterations=501 algo.NORMALIZE_ENV=False algo.init_lecun=True
#python main.py algo=ppoish_me seed=$RANDOM env=anttrap_omni env.episode_length=250 num_iterations=501 algo.NORMALIZE_ENV=True

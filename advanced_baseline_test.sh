#python main_me_mcpg_advanced_baseline.py seed=$RANDOM env=walker2d_uni env.episode_length=1000 num_iterations=2000 buffer_sample_batch_size=2
#python main_me_mcpg_advanced_baseline.py seed=$RANDOM env=walker2d_uni env.episode_length=1000 num_iterations=2000 buffer_sample_batch_size=2

python main_me_mcpg_advanced_baseline.py seed=$RANDOM env=walker2d_uni env.episode_length=1000 num_iterations=2000 buffer_sample_batch_size=8
python main_me_mcpg_advanced_baseline.py seed=$RANDOM env=walker2d_uni env.episode_length=1000 num_iterations=2000 buffer_sample_batch_size=8

python main_me_mcpg_advanced_baseline.py seed=$RANDOM env=ant_uni env.episode_length=1000 num_iterations=2000 buffer_sample_batch_size=2
python main_me_mcpg_advanced_baseline.py seed=$RANDOM env=ant_uni env.episode_length=1000 num_iterations=2000 buffer_sample_batch_size=2

python main_me_mcpg_advanced_baseline.py seed=$RANDOM env=ant_uni env.episode_length=1000 num_iterations=2000 buffer_sample_batch_size=8
python main_me_mcpg_advanced_baseline.py seed=$RANDOM env=ant_uni env.episode_length=1000 num_iterations=2000 buffer_sample_batch_size=8




# FIGURE 2 & FIGURE 4 & FIGURE 5

# ASCII-ME
python main.py algo=ascii_me env=walker2d_uni batch_size=4096 num_iterations=5000 seed=$RANDOM
python main.py algo=ascii_me env=ant_uni batch_size=4096 num_iterations=5000 seed=$RANDOM
python main.py algo=ascii_me env=hopper_uni batch_size=4096 num_iterations=5000 seed=$RANDOM
python main.py algo=ascii_me env=ant_omni batch_size=4096 num_iterations=5000 seed=$RANDOM
python main.py algo=ascii_me env=anttrap_omni batch_size=4096 num_iterations=5000 seed=$RANDOM

# PGA-ME
python main.py algo=pga_me env=walker2d_uni batch_size=1024 algo.replay_buffer_size=4_000_000 num_iterations=1000 seed=$RANDOM
python main.py algo=pga_me env=ant_uni batch_size=1024 algo.replay_buffer_size=4_000_000 num_iterations=1000 seed=$RANDOM
python main.py algo=pga_me env=hopper_uni batch_size=1024 algo.replay_buffer_size=4_000_000 num_iterations=1000 seed=$RANDOM
python main.py algo=pga_me env=ant_omni batch_size=1024 algo.replay_buffer_size=4_000_000 num_iterations=1000 seed=$RANDOM
python main.py algo=pga_me env=anttrap_omni batch_size=1024 algo.replay_buffer_size=4_000_000 num_iterations=1000 seed=$RANDOM

# DCRL-ME
python main.py algo=dcrl_me env=walker2d_uni batch_size=2048 algo.replay_buffer_size=8_000_000 algo.ga_batch_size=1024 algo.qpg_batch_size=512 algo.ai_batch_size=512 num_iterations=2000 seed=$RANDOM
python main.py algo=dcrl_me env=ant_uni batch_size=2048 algo.replay_buffer_size=8_000_000 algo.ga_batch_size=1024 algo.qpg_batch_size=512 algo.ai_batch_size=512 num_iterations=2000 seed=$RANDOM
python main.py algo=dcrl_me env=hopper_uni batch_size=2048 algo.replay_buffer_size=8_000_000 algo.ga_batch_size=1024 algo.qpg_batch_size=512 algo.ai_batch_size=512 num_iterations=2000 seed=$RANDOM
python main.py algo=dcrl_me env=ant_omni batch_size=2048 algo.replay_buffer_size=8_000_000 algo.ga_batch_size=1024 algo.qpg_batch_size=512 algo.ai_batch_size=512 num_iterations=2000 seed=$RANDOM
python main.py algo=dcrl_me env=anttrap_omni batch_size=2048 algo.replay_buffer_size=8_000_000 algo.ga_batch_size=1024 algo.qpg_batch_size=512 algo.ai_batch_size=512 num_iterations=2000 seed=$RANDOM

# ME
python main.py algo=me env=walker2d_uni batch_size=8192 num_iterations=7500 seed=$RANDOM
python main.py algo=me env=ant_uni batch_size=8192 num_iterations=7500 seed=$RANDOM
python main.py algo=me env=hopper_uni batch_size=8192 num_iterations=7500 seed=$RANDOM
python main.py algo=me env=ant_omni batch_size=8192 num_iterations=7500 seed=$RANDOM
python main.py algo=me env=anttrap_omni batch_size=8192 num_iterations=7500 seed=$RANDOM

# MEMES
python main.py algo=memes env=anttrap_omni batch_size=16 algo.sample_number=512 algo.num_in_optimizer_steps=1 algo.scan_novelty=8192 num_iterations=500 seed=$RANDOM
python main.py algo=memes env=ant_omni batch_size=16 algo.sample_number=512 algo.num_in_optimizer_steps=1 algo.scan_novelty=8192 num_iterations=500 seed=$RANDOM
python main.py algo=memes env=ant_uni batch_size=16 algo.sample_number=512 algo.num_in_optimizer_steps=1 algo.scan_novelty=8192 num_iterations=500 seed=$RANDOM
python main.py algo=memes env=walker2d_uni batch_size=16 algo.sample_number=512 algo.num_in_optimizer_steps=1 algo.scan_novelty=8192 num_iterations=500 seed=$RANDOM
python main.py algo=memes env=hopper_uni batch_size=16 algo.sample_number=512 algo.num_in_optimizer_steps=1 algo.scan_novelty=8192 num_iterations=500 seed=$RANDOM

# PPGA
python main.py algo=ppga env=ant_uni algo.env_batch_size=6000 algo.rollout_length=64 algo.total_iterations=300 seed=$RANDOM
python main.py algo=ppga env=hopper_uni algo.env_batch_size=6000 algo.rollout_length=64 algo.total_iterations=300 seed=$RANDOM
python main.py algo=ppga env=walker2d_uni algo.env_batch_size=6000 algo.rollout_length=64 algo.total_iterations=300 seed=$RANDOM
python main.py algo=ppga env=ant_omni algo.env_batch_size=6000 algo.rollout_length=64 algo.total_iterations=300 seed=$RANDOM
python main.py algo=ppga env=anttrap_omni algo.env_batch_size=6000 algo.rollout_length=64 algo.total_iterations=300 seed=$RANDOM





# FIGURE 3

# ASCII-ME
python main.py algo=ascii_me env=anttrap_omni batch_size=256 num_iterations=4000 seed=$RANDOM
python main.py algo=ascii_me env=anttrap_omni batch_size=512 num_iterations=2000 seed=$RANDOM
python main.py algo=ascii_me env=anttrap_omni batch_size=1024 num_iterations=1000 seed=$RANDOM
python main.py algo=ascii_me env=anttrap_omni batch_size=2048 num_iterations=500 seed=$RANDOM
python main.py algo=ascii_me env=anttrap_omni batch_size=4096 num_iterations=250 seed=$RANDOM
python main.py algo=ascii_me env=anttrap_omni batch_size=8192 num_iterations=125 seed=$RANDOM


python main.py algo=ascii_me env=hopper_uni batch_size=256 num_iterations=4000 seed=$RANDOM
python main.py algo=ascii_me env=hopper_uni batch_size=512 num_iterations=2000 seed=$RANDOM
python main.py algo=ascii_me env=hopper_uni batch_size=1024 num_iterations=1000 seed=$RANDOM
python main.py algo=ascii_me env=hopper_uni batch_size=2048 num_iterations=500 seed=$RANDOM
python main.py algo=ascii_me env=hopper_uni batch_size=4096 num_iterations=250 seed=$RANDOM
python main.py algo=ascii_me env=hopper_uni batch_size=8192 num_iterations=125 seed=$RANDOM

python main.py algo=ascii_me env=ant_uni batch_size=256 num_iterations=4000 seed=$RANDOM
python main.py algo=ascii_me env=ant_uni batch_size=512 num_iterations=2000 seed=$RANDOM
python main.py algo=ascii_me env=ant_uni batch_size=1024 num_iterations=1000 seed=$RANDOM
python main.py algo=ascii_me env=ant_uni batch_size=2048 num_iterations=500 seed=$RANDOM
python main.py algo=ascii_me env=ant_uni batch_size=4096 num_iterations=250 seed=$RANDOM
python main.py algo=ascii_me env=ant_uni batch_size=8192 num_iterations=125 seed=$RANDOM

python main.py algo=ascii_me env=walker2d_uni batch_size=256 num_iterations=4000 seed=$RANDOM
python main.py algo=ascii_me env=walker2d_uni batch_size=512 num_iterations=2000 seed=$RANDOM
python main.py algo=ascii_me env=walker2d_uni batch_size=1024 num_iterations=1000 seed=$RANDOM
python main.py algo=ascii_me env=walker2d_uni batch_size=2048 num_iterations=500 seed=$RANDOM
python main.py algo=ascii_me env=walker2d_uni batch_size=4096 num_iterations=250 seed=$RANDOM
python main.py algo=ascii_me env=walker2d_uni batch_size=8192 num_iterations=125 seed=$RANDOM

python main.py algo=ascii_me env=ant_omni batch_size=256 num_iterations=4000 seed=$RANDOM
python main.py algo=ascii_me env=ant_omni batch_size=512 num_iterations=2000 seed=$RANDOM
python main.py algo=ascii_me env=ant_omni batch_size=1024 num_iterations=1000 seed=$RANDOM
python main.py algo=ascii_me env=ant_omni batch_size=2048 num_iterations=500 seed=$RANDOM
python main.py algo=ascii_me env=ant_omni batch_size=4096 num_iterations=250 seed=$RANDOM
python main.py algo=ascii_me env=ant_omni batch_size=8192 num_iterations=125 seed=$RANDOM

# MEMES
python main.py algo=memes env=anttrap_omni batch_size=16 algo.sample_number=512 algo.num_in_optimizer_steps=1 algo.scan_novelty=8192 num_iterations=125 seed=$RANDOM
python main.py algo=memes env=ant_omni batch_size=16 algo.sample_number=512 algo.num_in_optimizer_steps=1 algo.scan_novelty=8192 num_iterations=125 seed=$RANDOM
python main.py algo=memes env=ant_uni batch_size=16 algo.sample_number=512 algo.num_in_optimizer_steps=1 algo.scan_novelty=8192 num_iterations=125 seed=$RANDOM
python main.py algo=memes env=walker2d_uni batch_size=16 algo.sample_number=512 algo.num_in_optimizer_steps=1 algo.scan_novelty=8192 num_iterations=125 seed=$RANDOM
python main.py algo=memes env=hopper_uni batch_size=16 algo.sample_number=512 algo.num_in_optimizer_steps=1 algo.scan_novelty=8192 num_iterations=125 seed=$RANDOM

python main.py algo=memes env=anttrap_omni batch_size=8 algo.sample_number=512 algo.num_in_optimizer_steps=1 algo.scan_novelty=8192 num_iterations=250 seed=$RANDOM
python main.py algo=memes env=ant_omni batch_size=8 algo.sample_number=512 algo.num_in_optimizer_steps=1 algo.scan_novelty=8192 num_iterations=250 seed=$RANDOM
python main.py algo=memes env=ant_uni batch_size=8 algo.sample_number=512 algo.num_in_optimizer_steps=1 algo.scan_novelty=8192 num_iterations=250 seed=$RANDOM
python main.py algo=memes env=walker2d_uni batch_size=8 algo.sample_number=512 algo.num_in_optimizer_steps=1 algo.scan_novelty=8192 num_iterations=250 seed=$RANDOM
python main.py algo=memes env=hopper_uni batch_size=8 algo.sample_number=512 algo.num_in_optimizer_steps=1 algo.scan_novelty=8192 num_iterations=250 seed=$RANDOM

python main.py algo=memes env=anttrap_omni batch_size=4 algo.sample_number=512 algo.num_in_optimizer_steps=1 algo.scan_novelty=8192 num_iterations=500 seed=$RANDOM
python main.py algo=memes env=ant_omni batch_size=4 algo.sample_number=512 algo.num_in_optimizer_steps=1 algo.scan_novelty=8192 num_iterations=500 seed=$RANDOM
python main.py algo=memes env=ant_uni batch_size=4 algo.sample_number=512 algo.num_in_optimizer_steps=1 algo.scan_novelty=8192 num_iterations=500 seed=$RANDOM
python main.py algo=memes env=walker2d_uni batch_size=4 algo.sample_number=512 algo.num_in_optimizer_steps=1 algo.scan_novelty=8192 num_iterations=500 seed=$RANDOM
python main.py algo=memes env=hopper_uni batch_size=4 algo.sample_number=512 algo.num_in_optimizer_steps=1 algo.scan_novelty=8192 num_iterations=500 seed=$RANDOM

python main.py algo=memes env=anttrap_omni batch_size=2 algo.sample_number=512 algo.num_in_optimizer_steps=1 algo.scan_novelty=8192 num_iterations=1000 seed=$RANDOM
python main.py algo=memes env=ant_omni batch_size=2 algo.sample_number=512 algo.num_in_optimizer_steps=1 algo.scan_novelty=8192 num_iterations=1000 seed=$RANDOM
python main.py algo=memes env=ant_uni batch_size=2 algo.sample_number=512 algo.num_in_optimizer_steps=1 algo.scan_novelty=8192 num_iterations=1000 seed=$RANDOM
python main.py algo=memes env=walker2d_uni batch_size=2 algo.sample_number=512 algo.num_in_optimizer_steps=1 algo.scan_novelty=8192 num_iterations=1000 seed=$RANDOM
python main.py algo=memes env=hopper_uni batch_size=2 algo.sample_number=512 algo.num_in_optimizer_steps=1 algo.scan_novelty=8192 num_iterations=1000 seed=$RANDOM

python main.py algo=memes env=anttrap_omni batch_size=1 algo.sample_number=512 algo.num_in_optimizer_steps=1 algo.scan_novelty=8192 num_iterations=2000 seed=$RANDOM
python main.py algo=memes env=ant_omni batch_size=1 algo.sample_number=512 algo.num_in_optimizer_steps=1 algo.scan_novelty=8192 num_iterations=2000 seed=$RANDOM
python main.py algo=memes env=ant_uni batch_size=1 algo.sample_number=512 algo.num_in_optimizer_steps=1 algo.scan_novelty=8192 num_iterations=2000 seed=$RANDOM
python main.py algo=memes env=walker2d_uni batch_size=1 algo.sample_number=512 algo.num_in_optimizer_steps=1 algo.scan_novelty=8192 num_iterations=2000 seed=$RANDOM
python main.py algo=memes env=hopper_uni batch_size=1 algo.sample_number=512 algo.num_in_optimizer_steps=1 algo.scan_novelty=8192 num_iterations=2000 seed=$RANDOM

# ME
python main.py algo=me env=anttrap_omni batch_size=256 num_iterations=4000 seed=$RANDOM
python main.py algo=me env=anttrap_omni batch_size=512 num_iterations=2000 seed=$RANDOM
python main.py algo=me env=anttrap_omni batch_size=1024 num_iterations=1000 seed=$RANDOM
python main.py algo=me env=anttrap_omni batch_size=2048 num_iterations=500 seed=$RANDOM
python main.py algo=me env=anttrap_omni batch_size=4096 num_iterations=250 seed=$RANDOM
python main.py algo=me env=anttrap_omni batch_size=8192 num_iterations=125 seed=$RANDOM


python main.py algo=me env=hopper_uni batch_size=256 num_iterations=4000 seed=$RANDOM
python main.py algo=me env=hopper_uni batch_size=512 num_iterations=2000 seed=$RANDOM
python main.py algo=me env=hopper_uni batch_size=1024 num_iterations=1000 seed=$RANDOM
python main.py algo=me env=hopper_uni batch_size=2048 num_iterations=500 seed=$RANDOM
python main.py algo=me env=hopper_uni batch_size=4096 num_iterations=250 seed=$RANDOM
python main.py algo=me env=hopper_uni batch_size=8192 num_iterations=125 seed=$RANDOM

python main.py algo=me env=ant_uni batch_size=256 num_iterations=4000 seed=$RANDOM
python main.py algo=me env=ant_uni batch_size=512 num_iterations=2000 seed=$RANDOM
python main.py algo=me env=ant_uni batch_size=1024 num_iterations=1000 seed=$RANDOM
python main.py algo=me env=ant_uni batch_size=2048 num_iterations=500 seed=$RANDOM
python main.py algo=me env=ant_uni batch_size=4096 num_iterations=250 seed=$RANDOM
python main.py algo=me env=ant_uni batch_size=8192 num_iterations=125 seed=$RANDOM

python main.py algo=me env=walker2d_uni batch_size=256 num_iterations=4000 seed=$RANDOM
python main.py algo=me env=walker2d_uni batch_size=512 num_iterations=2000 seed=$RANDOM
python main.py algo=me env=walker2d_uni batch_size=1024 num_iterations=1000 seed=$RANDOM
python main.py algo=me env=walker2d_uni batch_size=2048 num_iterations=500 seed=$RANDOM
python main.py algo=me env=walker2d_uni batch_size=4096 num_iterations=250 seed=$RANDOM
python main.py algo=me env=walker2d_uni batch_size=8192 num_iterations=125 seed=$RANDOM

python main.py algo=me env=ant_omni batch_size=256 num_iterations=4000 seed=$RANDOM
python main.py algo=me env=ant_omni batch_size=512 num_iterations=2000 seed=$RANDOM
python main.py algo=me env=ant_omni batch_size=1024 num_iterations=1000 seed=$RANDOM
python main.py algo=me env=ant_omni batch_size=2048 num_iterations=500 seed=$RANDOM
python main.py algo=me env=ant_omni batch_size=4096 num_iterations=250 seed=$RANDOM
python main.py algo=me env=ant_omni batch_size=8192 num_iterations=125 seed=$RANDOM

# DCRL-ME
python main.py algo=dcrl_me env=hopper_uni batch_size=256 algo.ga_batch_size=128 algo.qpg_batch_size=64 algo.ai_batch_size=64 algo.replay_buffer_size=1_000_000 num_iterations=4000 seed=$RANDOM
python main.py algo=dcrl_me env=hopper_uni batch_size=512 algo.ga_batch_size=256 algo.qpg_batch_size=128 algo.ai_batch_size=128 algo.replay_buffer_size=2_000_000 num_iterations=2000 seed=$RANDOM
python main.py algo=dcrl_me env=hopper_uni batch_size=1024 algo.ga_batch_size=512 algo.qpg_batch_size=256 algo.ai_batch_size=256 algo.replay_buffer_size=4_000_000 num_iterations=1000 seed=$RANDOM
python main.py algo=dcrl_me env=hopper_uni batch_size=2048 algo.ga_batch_size=1024 algo.qpg_batch_size=512 algo.ai_batch_size=512 algo.replay_buffer_size=8_000_000 num_iterations=500 seed=$RANDOM
python main.py algo=dcrl_me env=hopper_uni batch_size=8192 algo.ga_batch_size=4096 algo.qpg_batch_size=2048 algo.ai_batch_size=2048 algo.replay_buffer_size=32_000_000 num_iterations=125 seed=$RANDOM

python main.py algo=dcrl_me env=ant_uni batch_size=256 algo.ga_batch_size=128 algo.qpg_batch_size=64 algo.ai_batch_size=64 algo.replay_buffer_size=1_000_000 num_iterations=4000 seed=$RANDOM
python main.py algo=dcrl_me env=ant_uni batch_size=512 algo.ga_batch_size=256 algo.qpg_batch_size=128 algo.ai_batch_size=128 algo.replay_buffer_size=2_000_000 num_iterations=2000 seed=$RANDOM
python main.py algo=dcrl_me env=ant_uni batch_size=1024 algo.ga_batch_size=512 algo.qpg_batch_size=256 algo.ai_batch_size=256 algo.replay_buffer_size=4_000_000 num_iterations=1000 seed=$RANDOM
python main.py algo=dcrl_me env=ant_uni batch_size=2048 algo.ga_batch_size=1024 algo.qpg_batch_size=512 algo.ai_batch_size=512 algo.replay_buffer_size=8_000_000 num_iterations=500 seed=$RANDOM
python main.py algo=dcrl_me env=ant_uni batch_size=4096 algo.ga_batch_size=2048 algo.qpg_batch_size=1024 algo.ai_batch_size=1024 algo.replay_buffer_size=16_000_000 num_iterations=250 seed=$RANDOM
python main.py algo=dcrl_me env=ant_uni batch_size=8192 algo.ga_batch_size=4096 algo.qpg_batch_size=2048 algo.ai_batch_size=2048 algo.replay_buffer_size=32_000_000 num_iterations=125 seed=$RANDOM

python main.py algo=dcrl_me env=walker2d_uni batch_size=256 algo.ga_batch_size=128 algo.qpg_batch_size=64 algo.ai_batch_size=64 algo.replay_buffer_size=1_000_000 num_iterations=4000 seed=$RANDOM
python main.py algo=dcrl_me env=walker2d_uni batch_size=512 algo.ga_batch_size=256 algo.qpg_batch_size=128 algo.ai_batch_size=128 algo.replay_buffer_size=2_000_000 num_iterations=2000 seed=$RANDOM
python main.py algo=dcrl_me env=walker2d_uni batch_size=1024 algo.ga_batch_size=512 algo.qpg_batch_size=256 algo.ai_batch_size=256 algo.replay_buffer_size=4_000_000 num_iterations=1000 seed=$RANDOM
python main.py algo=dcrl_me env=walker2d_uni batch_size=2048 algo.ga_batch_size=1024 algo.qpg_batch_size=512 algo.ai_batch_size=512 algo.replay_buffer_size=8_000_000 num_iterations=500 seed=$RANDOM
python main.py algo=dcrl_me env=walker2d_uni batch_size=4096 algo.ga_batch_size=2048 algo.qpg_batch_size=1024 algo.ai_batch_size=1024 algo.replay_buffer_size=16_000_000 num_iterations=250 seed=$RANDOM
python main.py algo=dcrl_me env=walker2d_uni batch_size=8192 algo.ga_batch_size=4096 algo.qpg_batch_size=2048 algo.ai_batch_size=2048 algo.replay_buffer_size=32_000_000 num_iterations=125 seed=$RANDOM

python main.py algo=dcrl_me env=anttrap_omni batch_size=256 algo.ga_batch_size=128 algo.qpg_batch_size=64 algo.ai_batch_size=64 algo.replay_buffer_size=1_000_000 num_iterations=4000 seed=$RANDOM
python main.py algo=dcrl_me env=anttrap_omni batch_size=512 algo.ga_batch_size=256 algo.qpg_batch_size=128 algo.ai_batch_size=128 algo.replay_buffer_size=2_000_000 num_iterations=2000 seed=$RANDOM
python main.py algo=dcrl_me env=anttrap_omni batch_size=1024 algo.ga_batch_size=512 algo.qpg_batch_size=256 algo.ai_batch_size=256 algo.replay_buffer_size=4_000_000 num_iterations=1000 seed=$RANDOM
python main.py algo=dcrl_me env=anttrap_omni batch_size=2048 algo.ga_batch_size=1024 algo.qpg_batch_size=512 algo.ai_batch_size=512 algo.replay_buffer_size=8_000_000 num_iterations=500 seed=$RANDOM
python main.py algo=dcrl_me env=anttrap_omni batch_size=4096 algo.ga_batch_size=2048 algo.qpg_batch_size=1024 algo.ai_batch_size=1024 algo.replay_buffer_size=16_000_000 num_iterations=250 seed=$RANDOM
python main.py algo=dcrl_me env=anttrap_omni batch_size=8192 algo.ga_batch_size=4096 algo.qpg_batch_size=2048 algo.ai_batch_size=2048 algo.replay_buffer_size=32_000_000 num_iterations=125 seed=$RANDOM

python main.py algo=dcrl_me env=ant_omni batch_size=256 algo.ga_batch_size=128 algo.qpg_batch_size=64 algo.ai_batch_size=64 algo.replay_buffer_size=1_000_000 num_iterations=4000 seed=$RANDOM
python main.py algo=dcrl_me env=ant_omni batch_size=512 algo.ga_batch_size=256 algo.qpg_batch_size=128 algo.ai_batch_size=128 algo.replay_buffer_size=2_000_000 num_iterations=2000 seed=$RANDOM
python main.py algo=dcrl_me env=ant_omni batch_size=1024 algo.ga_batch_size=512 algo.qpg_batch_size=256 algo.ai_batch_size=256 algo.replay_buffer_size=4_000_000 num_iterations=1000 seed=$RANDOM
python main.py algo=dcrl_me env=ant_omni batch_size=2048 algo.ga_batch_size=1024 algo.qpg_batch_size=512 algo.ai_batch_size=512 algo.replay_buffer_size=8_000_000 num_iterations=500 seed=$RANDOM
python main.py algo=dcrl_me env=ant_omni batch_size=4096 algo.ga_batch_size=2048 algo.qpg_batch_size=1024 algo.ai_batch_size=1024 algo.replay_buffer_size=16_000_000 num_iterations=250 seed=$RANDOM
python main.py algo=dcrl_me env=ant_omni batch_size=8192 algo.ga_batch_size=4096 algo.qpg_batch_size=2048 algo.ai_batch_size=2048 algo.replay_buffer_size=32_000_000 num_iterations=125 seed=$RANDOM

# PGA-ME
python main.py algo=pga_me env=anttrap_omni batch_size=256 algo.replay_buffer_size=1_000_000 num_iterations=4000 seed=$RANDOM
python main.py algo=pga_me env=anttrap_omni batch_size=512 algo.replay_buffer_size=2_000_000 num_iterations=2000 seed=$RANDOM
python main.py algo=pga_me env=anttrap_omni batch_size=1024 algo.replay_buffer_size=4_000_000 num_iterations=1000 seed=$RANDOM
python main.py algo=pga_me env=anttrap_omni batch_size=2048 algo.replay_buffer_size=8_000_000 num_iterations=500 seed=$RANDOM
python main.py algo=pga_me env=anttrap_omni batch_size=4096 algo.replay_buffer_size=16_000_000 num_iterations=250 seed=$RANDOM
python main.py algo=pga_me env=anttrap_omni batch_size=8192 algo.replay_buffer_size=32_000_000 num_iterations=125 seed=$RANDOM

python main.py algo=pga_me env=hopper_uni batch_size=256 algo.replay_buffer_size=1_000_000 num_iterations=4000 seed=$RANDOM
python main.py algo=pga_me env=hopper_uni batch_size=512 algo.replay_buffer_size=2_000_000 num_iterations=2000 seed=$RANDOM
python main.py algo=pga_me env=hopper_uni batch_size=1024 algo.replay_buffer_size=4_000_000 num_iterations=1000 seed=$RANDOM
python main.py algo=pga_me env=hopper_uni batch_size=2048 algo.replay_buffer_size=8_000_000 num_iterations=500 seed=$RANDOM
python main.py algo=pga_me env=hopper_uni batch_size=4096 algo.replay_buffer_size=16_000_000 num_iterations=250 seed=$RANDOM
python main.py algo=pga_me env=hopper_uni batch_size=8192 algo.replay_buffer_size=32_000_000 num_iterations=125 seed=$RANDOM

python main.py algo=pga_me env=ant_uni batch_size=256 algo.replay_buffer_size=1_000_000 num_iterations=4000 seed=$RANDOM
python main.py algo=pga_me env=ant_uni batch_size=512 algo.replay_buffer_size=2_000_000 num_iterations=2000 seed=$RANDOM
python main.py algo=pga_me env=ant_uni batch_size=1024 algo.replay_buffer_size=4_000_000 num_iterations=1000 seed=$RANDOM
python main.py algo=pga_me env=ant_uni batch_size=2048 algo.replay_buffer_size=8_000_000 num_iterations=500 seed=$RANDOM
python main.py algo=pga_me env=ant_uni batch_size=4096 algo.replay_buffer_size=16_000_000 num_iterations=250 seed=$RANDOM
python main.py algo=pga_me env=ant_uni batch_size=8192 algo.replay_buffer_size=32_000_000 num_iterations=125 seed=$RANDOM

python main.py algo=pga_me env=walker2d_uni batch_size=256 algo.replay_buffer_size=1_000_000 num_iterations=4000 seed=$RANDOM
python main.py algo=pga_me env=walker2d_uni batch_size=512 algo.replay_buffer_size=2_000_000 num_iterations=2000 seed=$RANDOM
python main.py algo=pga_me env=walker2d_uni batch_size=1024 algo.replay_buffer_size=4_000_000 num_iterations=1000 seed=$RANDOM
python main.py algo=pga_me env=walker2d_uni batch_size=2048 algo.replay_buffer_size=8_000_000 num_iterations=500 seed=$RANDOM
python main.py algo=pga_me env=walker2d_uni batch_size=4096 algo.replay_buffer_size=16_000_000 num_iterations=250 seed=$RANDOM
python main.py algo=pga_me env=walker2d_uni batch_size=8192 algo.replay_buffer_size=32_000_000 num_iterations=125 seed=$RANDOM

python main.py algo=pga_me env=ant_omni batch_size=256 algo.replay_buffer_size=1_000_000 num_iterations=4000 seed=$RANDOM
python main.py algo=pga_me env=ant_omni batch_size=512 algo.replay_buffer_size=2_000_000 num_iterations=2000 seed=$RANDOM
python main.py algo=pga_me env=ant_omni batch_size=1024 algo.replay_buffer_size=4_000_000 num_iterations=1000 seed=$RANDOM
python main.py algo=pga_me env=ant_omni batch_size=2048 algo.replay_buffer_size=8_000_000 num_iterations=500 seed=$RANDOM
python main.py algo=pga_me env=ant_omni batch_size=4096 algo.replay_buffer_size=16_000_000 num_iterations=250 seed=$RANDOM
python main.py algo=pga_me env=ant_omni batch_size=8192 algo.replay_buffer_size=32_000_000 num_iterations=125 seed=$RANDOM

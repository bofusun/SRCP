Xvfb :1 -screen 0 1024x768x24 &
cd /data/sjb/HILP-master/hilp_zsrl
conda activate SRCP
export DISPLAY=:1

CUDA_VISIBLE_DEVICES=0 python url_benchmark/train_offline.py run_group=EXP device=cuda agent=srcp agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False seed=10 task=walker_run expl_agent=rnd load_replay_buffer=/data/sjb/exorl-main/datasets/walker/rnd/replay_pixel64.pt replay_buffer_episodes=5000 obs_type=pixels agent.batch_size=512 num_grad_steps=500000
CUDA_VISIBLE_DEVICES=0 python url_benchmark/train_offline.py run_group=EXP device=cuda agent=srcp agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False seed=10 task=quadruped_run expl_agent=rnd load_replay_buffer=/data/sjb/exorl-main/datasets/quadruped/rnd/replay_pixel64.pt replay_buffer_episodes=5000 obs_type=pixels agent.batch_size=512 num_grad_steps=500000
CUDA_VISIBLE_DEVICES=0 python url_benchmark/train_offline.py run_group=EXP device=cuda agent=srcpc agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False seed=10 task=cheetah_run expl_agent=rnd load_replay_buffer=/data/sjb/exorl-main/datasets/cheetah/rnd/replay_pixel64.pt replay_buffer_episodes=5000 obs_type=pixels agent.batch_size=512 num_grad_steps=500000
CUDA_VISIBLE_DEVICES=0 python url_benchmark/train_offline.py run_group=EXP device=cuda agent=srcp agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False seed=10 task=jaco_reach_top_left expl_agent=rnd load_replay_buffer=/data/sjb/exorl-main/datasets/jaco/rnd/replay_pixel64.pt replay_buffer_episodes=5000 obs_type=pixels agent.batch_size=512 num_grad_steps=500000


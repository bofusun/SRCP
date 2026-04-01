# SRCP
Code for "Saliency-Guided Representation with Consistency Policy Learning for Visual Unsupervised Reinforcement Learning", CVPR 2026 paper.

---

# Introduction

we propose **S**aliency-Guided **R**epresentation with **C**onsistency **P**olicy Learning (SRCP), a novel framework that enhances zero-shot generalization of SR methods in visual URL. 
SRCP decouples representation learning from successor training by introducing a saliency-guided dynamics task to capture dynamics-relevant representations, thereby improving successor measure.
Moreover, it integrates a fast-sampling consistency policy 
with URL-specific classifier-free guidance and tailored training objectives to improve skill-conditioned policy modeling and controllability.
Extensive experiments demonstrate that SRCP achieves state-of-the-art zero-shot generalization in visual URL and and remains compatible with various SR methods.

<img width="5115" height="2598" alt="frame11 (1)_01" src="https://github.com/user-attachments/assets/6eb6413f-df20-45fe-b80b-3691169d8c4d" />

---
# Quick Start

1. Setting up repo
```
git clone https://github.com/bofusun/SRCP
```
2. Install Dependencies
```
conda create -n SRCP python=3.8
conda activate SRCP
cd SRCP
pip install -r requirements.txt
```
3. Train

(1) Pretrain SRCP on RND datatset
```
CUDA_VISIBLE_DEVICES=0 python url_benchmark/train_offline.py run_group=EXP device=cuda agent=srcp agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False seed=10 task=walker_run expl_agent=rnd load_replay_buffer=/datasets/walker/rnd/replay_pixel64.pt replay_buffer_episodes=5000 obs_type=pixels agent.batch_size=512 num_grad_steps=500000
CUDA_VISIBLE_DEVICES=0 python url_benchmark/train_offline.py run_group=EXP device=cuda agent=srcp agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False seed=10 task=quadruped_run expl_agent=rnd load_replay_buffer=/datasets/quadruped/rnd/replay_pixel64.pt replay_buffer_episodes=5000 obs_type=pixels agent.batch_size=512 num_grad_steps=500000
CUDA_VISIBLE_DEVICES=0 python url_benchmark/train_offline.py run_group=EXP device=cuda agent=srcpc agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False seed=10 task=cheetah_run expl_agent=rnd load_replay_buffer=/datasets/cheetah/rnd/replay_pixel64.pt replay_buffer_episodes=5000 obs_type=pixels agent.batch_size=512 num_grad_steps=500000
CUDA_VISIBLE_DEVICES=0 python url_benchmark/train_offline.py run_group=EXP device=cuda agent=srcp agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False seed=10 task=jaco_reach_top_left expl_agent=rnd load_replay_buffer=/datasets/jaco/rnd/replay_pixel64.pt replay_buffer_episodes=5000 obs_type=pixels agent.batch_size=512 num_grad_steps=500000
```

(2) Pretrain SRCPFB on RND datatset
```
CUDA_VISIBLE_DEVICES=0 python url_benchmark/train_offline.py run_group=EXP device=cuda agent=srcpfb agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False seed=10 task=walker_run expl_agent=rnd load_replay_buffer=/datasets/walker/rnd/replay_pixel64.pt replay_buffer_episodes=5000 obs_type=pixels agent.batch_size=512 num_grad_steps=500000
CUDA_VISIBLE_DEVICES=0 python url_benchmark/train_offline.py run_group=EXP device=cuda agent=srcpfb agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False seed=10 task=quadruped_run expl_agent=rnd load_replay_buffer=/datasets/quadruped/rnd/replay_pixel64.pt replay_buffer_episodes=5000 obs_type=pixels agent.batch_size=512 num_grad_steps=500000
CUDA_VISIBLE_DEVICES=0 python url_benchmark/train_offline.py run_group=EXP device=cuda agent=srcpfb agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False seed=10 task=cheetah_run expl_agent=rnd load_replay_buffer=/datasets/cheetah/rnd/replay_pixel64.pt replay_buffer_episodes=5000 obs_type=pixels agent.batch_size=512 num_grad_steps=500000
CUDA_VISIBLE_DEVICES=0 python url_benchmark/train_offline.py run_group=EXP device=cuda agent=srcpfb agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False seed=10 task=jaco_reach_top_left expl_agent=rnd load_replay_buffer=/datasets/jaco/rnd/replay_pixel64.pt replay_buffer_episodes=5000 obs_type=pixels agent.batch_size=512 num_grad_steps=500000

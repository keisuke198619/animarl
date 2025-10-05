## Data-driven simulator of multi-animal behavior using deep reinforcement learning 

### Requirements
* python 3.8+
* To install requirements (for Python 3.8):

```setup
pip install -r requirements.txt
```

* If you have a problem, see `pip_list_py3.8_all.txt` (not organized)

### configuration 
* The config files act as defaults for an algorithm or environment. 
* They are all located in `config`.
* `--config` refers to the config files in `config/algs`
* `--env-config` refers to the config files in `config/envs`

### Structure
* `components/` - Shared building blocks such as replay buffers, action selectors, and data transforms.
* `controllers/` - High-level logic for coordinating agents during rollouts.
* `envs/` - Multi-agent environment wrappers and the Animarl task definitions (`envs/animarl`).
* `learners/` - Training loops that implement the behaviour and Q-learning algorithms.
* `modules/` - Neural network modules, including MLP and RNN agent policies.
* `pretrain/` - Utilities and models used for supervised pretraining of agents.
* `runners/` - Episode runners that connect environments, learners, and controllers.
* `scripts/` - Shell scripts for running offline experiments, evaluations, and video generation.
* `utils/` - General-purpose utilities for logging, timing, and reinforcement-learning helpers.
* Top-level scripts such as `run.py`, `run.sh`, `learn_behavior.py`, and `learn_pretrainQ.py` provide entry points for different training and evaluation workflows.

### Main analysis
* see `run.sh`.
* dataset of animarl_human_2vs1 (artificial agents' chase-and-escape) are available.
* Further details are documented within the code.

### References 
- Multi-Agent Particle Environment: `https://github.com/openai/multiagent-particle-envs`
- Collaborative hunting: `https://github.com/TsutsuiKazushi/collaborative-hunting`
- MARL implementation: `https://github.com/lich14/CDS`

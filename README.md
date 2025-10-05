## Data-driven simulator of multi-animal behavior using deep reinforcement learning 

### Requirements
* python 3.8-
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

### Main analysis
* see `run.sh`.
* dataset of animarl_human_2vs1 (artificial agents' chase-and-escape) are available.
* Further details are documented within the code.

### References 
- Multi-Agent Particle Environment: `https://github.com/openai/multiagent-particle-envs`
- Collaborative hunting: `https://github.com/TsutsuiKazushi/collaborative-hunting`
- MARL implemenatation: `https://github.com/lich14/CDS`

from functools import partial
from .multiagentenv import MultiAgentEnv
from .animarl import Animarl_Agent_2vs1, Animarl_Silkmoth, Animarl_Fly_2vs1, Animarl_Newt_2vs1, Animarl_Bat, Animarl_Dragonfly
import sys
import os


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)
    
REGISTRY = {
    "animarl_agent_2vs1": partial(env_fn, env=Animarl_Agent_2vs1),    
    "animarl_silkmoth": partial(env_fn, env=Animarl_Silkmoth),  
    "animarl_fly_2vs1": partial(env_fn, env=Animarl_Fly_2vs1),
    "animarl_newt_2vs1": partial(env_fn, env=Animarl_Newt_2vs1),
    "animarl_bat": partial(env_fn, env=Animarl_Bat),
    "animarl_dragonfly_2vs1": partial(env_fn, env=Animarl_Dragonfly),
}


#if sys.platform == "linux":
#    os.environ.setdefault("SC2PATH",
#                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))

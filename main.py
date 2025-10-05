import numpy as np
import os, sys, argparse
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import torch as th
from utils.logging import get_logger
import yaml

from run import run
sys.setrecursionlimit(2000)  # default: 1000

numProcess = 4  
os.environ["OMP_NUM_THREADS"]=str(numProcess) 

# set to "no" if you want to see stdout/stderr in console
SETTINGS['CAPTURE_MODE'] = "fd"
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]
    
    # run the framework
    run(_run, config, _log)


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break
    if 'AAS' in config_name:
        config_name = config_name.replace('_AAS','')
        alg_from_demo = True
        alg_AAS = True
        alg_DIL = False
        alg_BC = False
    elif 'DIL' in config_name:
        config_name = config_name.replace('_DIL','')
        alg_from_demo = True
        alg_AAS = False
        alg_DIL = True
        alg_BC = False
    elif 'BC' in config_name:
        config_name = config_name.replace('_BC','')
        alg_from_demo = True
        alg_AAS = False
        alg_DIL = False
        alg_BC = True
    else: # RL only
        alg_from_demo = False
        alg_AAS = False
        alg_DIL = False
        alg_BC = False
    
    if 'RNN' in config_name or 'BC' in config_name:
        config_name = config_name.replace('_RNN','')
        config_name = config_name.replace('_BC','')
        alg_RNN = True
    else:
        alg_RNN = False

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)

        if arg_name == "--config":
            config_name = config_name.replace('DQN','')
            config_name = config_name.replace('BC','')
            variant_name = config_name
            return config_dict, alg_from_demo, alg_AAS, alg_DIL, alg_BC, alg_RNN, variant_name
        else:
            return config_dict

def _get_other_config(params, arg_name, arg_name2):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[0]
            config_val = _v.split("=")[1]
            if arg_name2 == 'test':
                config_val = True if config_val == 'True' else False
            elif arg_name2 == 'cond':
                config_val = config_val
            elif arg_name2 == 'cont':
                pretrain = False
                if config_val == 'True':
                    config_val = 'cont'
                elif config_val == 'False':
                    config_val = False 
                elif config_val == 'data':
                    config_val = False 
                    pretrain = True
                elif config_val == 'cont_data':
                    config_val = 'cont' 
                    pretrain = True
                elif 'cont_data_' in config_val:
                    pretrain = True
                    config_val = config_val[10:]
                elif 'cont_' in config_val:
                    pretrain = False
                    config_val = config_val[5:]
                # else: config_val = None
            del params[_i]
            break
    config_dict[arg_name2] = config_val
    if arg_name2 == 'test': 
        return config_dict
    elif arg_name2 == 'cond':
        return config_val
    elif arg_name2 == 'cont':
        return config_dict, pretrain 


    
def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


if __name__ == '__main__':
    params = deepcopy(sys.argv)
    th.set_num_threads(16)
    
    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
        
    alg_config, alg_from_demo, alg_AAS, alg_DIL, alg_BC, alg_RNN, variant_name = _get_config(params, "--config", "algs")
    env_config = _get_config(params, "--env-config", "envs")
    
    test_config = _get_other_config(params, "--test", "test")
    cont_config, pretrain = _get_other_config(params, "--cont", "cont")
    cond = _get_other_config(params, "--cond", "cond")

    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict= recursive_dict_update(config_dict, alg_config)
    if test_config:
        config_dict = recursive_dict_update(config_dict, test_config)
    config_dict['from_demo'] = alg_from_demo
    config_dict['alg_AAS'] = alg_AAS
    config_dict['pretrain'] = pretrain
    config_dict['alg_DIL'] = alg_DIL
    config_dict['alg_BC'] = alg_BC
    config_dict['alg_RNN'] = alg_RNN
    config_dict['variant'] = variant_name
    config_dict['cond'] = cond

    # now add all the config to sacred
    ex.add_config(config_dict)
    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")
    ex.observers.append(FileStorageObserver(file_obs_path)) # FileStorageObserver.create

    ex.run_commandline(params)


import gfootball.env as football_env
import time, pprint, json, os, importlib, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp 
from tensorboardX import SummaryWriter

from actor import *
from learner import *
from evaluator_with_hard_att_def import test
#from evaluator import evaluator
from datetime import datetime, timedelta


def save_args(arg_dict):
    os.makedirs(arg_dict["log_dir"])
    os.makedirs(arg_dict["log_dir_policy"])
    os.makedirs(arg_dict["log_dir_div"])
    os.makedirs(arg_dict["log_dir_dump"])
    os.makedirs(arg_dict["log_dir_dump_left"])
    os.makedirs(arg_dict["log_dir_dump_right"])

    for i in range(arg_dict["div_num"]):
        os.makedirs(arg_dict["log_dir_dump_left_skill_"+str(i)])

    args_info = json.dumps(arg_dict, indent=4)
    f = open(arg_dict["log_dir"]+"/args.json","w")
    f.write(args_info)
    f.close()

def copy_models(dir_src, dir_dst): # src: source, dst: destination
    # retireve list of models
    l_cands = [f for f in os.listdir(dir_src) if os.path.isfile(os.path.join(dir_src, f)) and 'model_' in f]
    l_cands = sorted(l_cands, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    print(f"models to be copied: {l_cands}")
    for m in l_cands:
        shutil.copyfile(os.path.join(dir_src, m), os.path.join(dir_dst, m))
    print(f"{len(l_cands)} models copied in the given directory")
    
def main(arg_dict):
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    cur_time = datetime.now()
    arg_dict["log_dir"] = "logs/" + cur_time.strftime("[%m-%d]%H.%M.%S") + "_Test_div"
    arg_dict["log_dir_policy"] = arg_dict["log_dir"] + '/policy'
    arg_dict["log_dir_div"] = arg_dict["log_dir"] + '/div'
    arg_dict["log_dir_dump"] = arg_dict["log_dir"] + '/dump'
    arg_dict["log_dir_dump_left"] = arg_dict["log_dir_dump"] + '/left'
    arg_dict["log_dir_dump_right"] = arg_dict["log_dir_dump"] + '/right'

    for i in range(arg_dict["div_num"]):
        arg_dict["log_dir_dump_left_skill_"+str(i)] = arg_dict["log_dir_dump_left"] + '/skill_'+str(i)

    save_args(arg_dict)

    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    pp = pprint.PrettyPrinter(indent=4)
    torch.set_num_threads(1)

    fe_att_def = importlib.import_module("encoders." + arg_dict["encoder"])
    fe_att_def = fe_att_def.FeatureEncoder()
    arg_dict["feature_dims"] = fe_att_def.get_feature_dims()

    model = importlib.import_module("models." + arg_dict["model"])
    div_model = importlib.import_module("models." + arg_dict["div_model"])
    cpu_device = torch.device('cpu')

    center_model= model.Model(arg_dict)
    
    if arg_dict["trained_model_path"]:
        checkpoint = torch.load(arg_dict["trained_model_path"], cpu_device) 
        center_model.load_state_dict(checkpoint['model_state_dict'])
    
    center_model.share_memory()

    processes = [] 

    for i in range(arg_dict["div_num"]):
        if "env_evaluation" in arg_dict:
            p = mp.Process(target=test, args=(center_model, i, arg_dict))
            p.start()
            processes.append(p)
        
    for p in processes:
        p.join()
    

if __name__ == '__main__':

    arg_dict = {
        "env": "11_vs_11_kaggle",    
        # "11_vs_11_selfplay" : environment used for self-play training
        # "11_vs_11_stochastic" : environment used for training against fixed opponent(rule-based AI)
        # "11_vs_11_kaggle" : environment used for training against fixed opponent(rule-based AI hard)
        "num_processes": 40,  # should be less than the number of cpu cores in your workstation.
        "batch_size": 32,   
        "buffer_size": 10,
        "rollout_len": 30,

        "lstm_size": 256,
        "k_epoch" : 3,
        "learning_rate" : 0.0001,
        "gamma" : 0.993,
        "lmbda" : 0.96,
        "entropy_coef" : 0.0001,
        "grad_clip" : 3.0,
        "eps_clip" : 0.1,

        "summary_game_window" : 29, 
        "model_save_min_interval" : 100000,  # number of gradient updates bewteen saving model
        "write_goal_dumps": True,
        "write_full_episode_dumps": False,


        "trained_model_path" : 'logs/[01-10]21.11.17_div/policy/model_15399360.tar', # use when you want to continue traning from given model.
        "latest_ratio" : 0.8, # works only for self_play training. 
        "latest_n_model" : 5, # works only for self_play training. 
        "print_mode" : False,

        "batch_num": 3,
        "div_num": 5,
        "div_batch_size": 128,
        "div_learning_rate" : 0.001,
        "div_lstm_size": 256,
        "div_model": "div_model4",
        "div_buffer_size": 1e6,
        "div_algorithm": "div_lr",
        "div_lr_step": 10000,
        "div_model_saved_interval": 60,

        "encoder" : "encoder_div",
        "rewarder" : "rewarder_att2",
        "model" : "conv1d_div2",#add left right closest
        "algorithm" : "ppo_with_lstm",

        "env_evaluation":'11_vs_11_stochastic'  # for evaluation of self-play trained agent (like validation set in Supervised Learning)
    }
    
    main(arg_dict)
from ast import arg
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
from evaluator import evaluator
#from evaluator import evaluator
from datetime import datetime, timedelta


def save_args(arg_dict):
    os.makedirs(arg_dict["log_dir"])
    os.makedirs(arg_dict["log_dir_policy"])
    os.makedirs(arg_dict["log_dir_div"])
    os.makedirs(arg_dict["log_dir_dump"])
    os.makedirs(arg_dict["log_dir_dump_left"])
    os.makedirs(arg_dict["log_dir_dump_right"])
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
    arg_dict["log_dir"] = "div_logs/" + cur_time.strftime("[%m-%d]%H.%M.%S") + "_div"
    arg_dict["log_dir_policy"] = arg_dict["log_dir"] + '/policy'
    arg_dict["log_dir_div"] = arg_dict["log_dir"] + '/div'
    arg_dict["log_dir_dump"] = arg_dict["log_dir"] + '/dump'
    arg_dict["log_dir_dump_left"] = arg_dict["log_dir_dump"] + '/left'
    arg_dict["log_dir_dump_right"] = arg_dict["log_dir_dump"] + '/right'
    for z in range(arg_dict["div_num"]):
        arg_dict["skill_"+str(z+1)] = arg_dict["log_dir_dump_left"] + '/skill' + str(z+1)
    
    save_args(arg_dict)
    if arg_dict["trained_model_path"]: 
        copy_models(os.path.dirname(arg_dict['trained_model_path']), arg_dict['log_dir_policy'])
    if arg_dict["trained_div_model_path"]: 
        copy_models(os.path.dirname(arg_dict['trained_div_model_path']), arg_dict['log_dir_div'])

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
    gpu_device = torch.device('cuda:0')

    center_model= model.Model(arg_dict)
    center_div_model = div_model.Model(arg_dict)

    if arg_dict["trained_model_path"]:
        checkpoint = torch.load(arg_dict["trained_model_path"], map_location=cpu_device)
        optimization_step = checkpoint['optimization_step']
        center_model.load_state_dict(checkpoint['model_state_dict'])
        center_model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        arg_dict["optimization_step"] = optimization_step
        print("Trained model", arg_dict["trained_model_path"] ,"suffessfully loaded") 
    else:
        optimization_step = 0

    if arg_dict["trained_div_model_path"]:
        div_checkpoint = torch.load(arg_dict["trained_div_model_path"], map_location=cpu_device)
        div_optimization_step = div_checkpoint['optimization_step']
        center_div_model.load_state_dict(div_checkpoint['model_state_dict'])
        center_div_model.optimizer.load_state_dict(div_checkpoint['optimizer_state_dict'])
        print("Trained div model", arg_dict["trained_div_model_path"] ,"suffessfully loaded") 
    else:
        div_optimization_step = 0

    model_dict = {
        'optimization_step': optimization_step,
        'model_state_dict': center_model.state_dict(),
        'optimizer_state_dict': center_model.optimizer.state_dict(),
    }
    div_model_dict = {
        'optimization_step': div_optimization_step,
        'model_state_dict': center_div_model.state_dict(),
        'optimizer_state_dict': center_div_model.optimizer.state_dict(),
    }
    
    path = arg_dict["log_dir_policy"]+f"/model_{optimization_step}.tar"
    torch.save(model_dict, path)
    div_path = arg_dict["log_dir_div"]+f"/div_model_{div_optimization_step}.tar"
    torch.save(div_model_dict, div_path)

        
    center_model.share_memory()
    center_div_model.share_memory()

    data_queue = mp.Queue()
    signal_queue = mp.Queue()

    summary_queue = mp.Queue()
    writer = SummaryWriter(logdir=arg_dict["log_dir"])
    
    processes = [] 

    p = mp.Process(target=learner, args=(center_div_model, center_model, data_queue, signal_queue, summary_queue, arg_dict, writer))
    p.start()
    processes.append(p)

    for rank in range(arg_dict["num_processes"]):
        if arg_dict["env"] == "11_vs_11_kaggle":
            p = mp.Process(target=actor_self, args=(rank, center_div_model, center_model, data_queue, signal_queue, summary_queue, arg_dict))
        else:
            p = mp.Process(target=actor, args=(rank, center_div_model, center_model, data_queue, signal_queue, summary_queue, arg_dict))
        p.start()
        processes.append(p)
    #for skill_num in range(arg_dict["div_num"]-1):
    for i in range(arg_dict["div_num"]*2):
        if "env_evaluation" in arg_dict:
            div_idx = i % arg_dict["div_num"]
            #p = mp.Process(target=evaluator, args=(skill_num+1, center_div_model, center_model, signal_queue, summary_queue, arg_dict))
            p = mp.Process(target=evaluator, args=(div_idx, center_div_model,center_model, signal_queue, summary_queue, arg_dict))
            p.start()
            processes.append(p)
        
    for p in processes:
        p.join()
    

if __name__ == '__main__':

    arg_dict = {
        "env": "11_vs_11_stochastic",    
        # "11_vs_11_selfplay" : environment used for self-play training
        # "11_vs_11_stochastic" : environment used for training against fixed opponent(rule-based AI)
        # "11_vs_11_kaggle" : environment used for training against fixed opponent(rule-based AI hard)
        "num_processes": 40,  # should be less than the number of cpu cores in your workstation.
        "batch_size": 32,   
        "buffer_size": 10,
        "rollout_len": 40,

        "lstm_size": 256,
        "k_epoch" : 3,
        "learning_rate" : 0.0001,
        "gamma" : 0.993,
        "lmbda" : 0.96,
        "entropy_coef" : 0.001,
        "grad_clip" : 3.0,
        "eps_clip" : 0.1,

        "summary_game_window" : 19, 
        "model_save_min_interval" : 300000,  # number of gradient updates bewteen saving model
        "saved_win_rate": -1,
        "saved_eval_win_rate": 0.9,

        "trained_model_path" : '', # use when you want to continue traning from given model.
        "trained_div_model_path" : '', # use when you want to continue traning from given model.
        "latest_ratio" : 0.5, # works only for self_play training. 
        "latest_n_model" : 10, # works only for self_play training. 
        "print_mode" : False,

        "batch_num": 3,
        "div_num": 5,
        "div_batch_size": 128,
        "div_learning_rate" : 0.001,
        "div_lstm_size": 64,
        "div_model": "ball_pos_div_model2",
        "div_buffer_size": 1e6,
        "div_algorithm": "div_lr",
        "div_lr_step": 1000,

        "encoder" : "encoder_div4",
        "rewarder" : "rewarder_att19",
        "model" : "conv1d_div2",#add left right closest
        "algorithm" : "ppo_with_lstm",

        "env_evaluation":'11_vs_11_competition'  # for evaluation of self-play trained agent (like validation set in Supervised Learning)
    }
    
    main(arg_dict)
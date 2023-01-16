import gfootball.env as football_env
import time, pprint, importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp 
from tensorboardX import SummaryWriter

def write_summary(writer, arg_dict, summary_queue, n_game, loss_lst, pi_loss_lst, v_loss_lst, \
                  entropy_lst, move_entropy_lst, optimization_step, self_play_board, win_evaluation, score_evaluation, div_loss_lst, div_entropy_lst, model, last_saved_step):
    win, score, tot_reward, game_len, div_acc = [], [], [], [], []
    loop_t, forward_t, wait_t = [], [], []

    for i in range(arg_dict["summary_game_window"]):
        game_data = summary_queue.get()
        a,b,c,d,opp_num,t1,t2,t3,z = game_data
        if arg_dict["env"] == "11_vs_11_kaggle":
            if opp_num in self_play_board:
                self_play_board[opp_num].append(a)
            else:
                self_play_board[opp_num] = [a]
        if 'env_evaluation' in arg_dict and opp_num==arg_dict['env_evaluation']:
            win_evaluation.append(a)
            score_evaluation.append(b)
        else:
            win.append(a)
            score.append(b)
            tot_reward.append(c)
            game_len.append(d)
            loop_t.append(t1)
            forward_t.append(t2)
            wait_t.append(t3)
            div_acc.append(z)

    writer.add_scalar('game/win_rate', float(np.mean(win)), n_game)
    writer.add_scalar('game/score', float(np.mean(score)), n_game)
    writer.add_scalar('game/reward', float(np.mean(tot_reward)), n_game)
    writer.add_scalar('game/div_acc', float(np.mean(div_acc)), n_game)
    writer.add_scalar('game/game_len', float(np.mean(game_len)), n_game)
    writer.add_scalar('train/step', float(optimization_step), n_game)
    writer.add_scalar('time/loop', float(np.mean(loop_t)), n_game)
    writer.add_scalar('time/forward', float(np.mean(forward_t)), n_game)
    writer.add_scalar('time/wait', float(np.mean(wait_t)), n_game)

    writer.add_scalar('train/loss', np.mean(loss_lst), n_game)
    writer.add_scalar('train/pi_loss', np.mean(pi_loss_lst), n_game)
    writer.add_scalar('train/v_loss', np.mean(v_loss_lst), n_game)
    writer.add_scalar('train/entropy', np.mean(entropy_lst), n_game)
    writer.add_scalar('train/move_entropy', np.mean(move_entropy_lst), n_game)

    if float(np.mean(win)) >= 0.8:
        if optimization_step >= last_saved_step + arg_dict["model_save_min_interval"]:
            model_dict = {
                'optimization_step': optimization_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
            }
            path = arg_dict["log_dir_policy"]+"/model_"+str(optimization_step)+".tar"
            torch.save(model_dict, path)
            print("Model saved :", path, "WinRate:", float(np.mean(win)))
            last_saved_step = optimization_step

    if len(div_loss_lst) > 0:
        writer.add_scalar('train/div_loss', np.mean(div_loss_lst), n_game)
        writer.add_scalar('train/div_entropy', np.mean(div_entropy_lst), n_game)

    mini_window = int(arg_dict['summary_game_window']//3)
    if len(win_evaluation)>=mini_window:
        writer.add_scalar('game/win_rate_evaluation', float(np.mean(win_evaluation)), n_game)
        writer.add_scalar('game/score_evaluation', float(np.mean(score_evaluation)), n_game)
        win_evaluation, score_evaluation = [], []

    for opp_num in self_play_board:
        if len(self_play_board[opp_num]) >= mini_window:
            label = 'self_play/'+opp_num
            writer.add_scalar(label, np.mean(self_play_board[opp_num][:mini_window]), n_game)
            self_play_board[opp_num] = self_play_board[opp_num][mini_window:]

    return win_evaluation, score_evaluation, last_saved_step


def save_div_model(div_model, arg_dict, div_step):
    if div_step % arg_dict["div_model_saved_interval"] == 0:
        div_model_dict = {
            'optimization_step': div_step*arg_dict["div_lr_step"],
            'model_state_dict': div_model.state_dict(),
            'optimizer_state_dict': div_model.optimizer.state_dict(),
        }

        path = arg_dict["log_dir_div"]+"/div_model_"+str(div_step*arg_dict["div_lr_step"])+".tar"

        torch.save(div_model_dict, path)
        print("Div_model saved :", path)
    else:
        pass

def seperate_save_model(i, model, arg_dict, optimization_step, last_saved_step):
    if optimization_step >= last_saved_step + arg_dict["model_save_interval"]:
        model_dict = {
            'optimization_step': optimization_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
        }
        if i == 0:
            path = arg_dict["log_dir_att"]+"/model_att_"+str(optimization_step)+".tar"
       
        else:
            path = arg_dict["log_dir_def"]+"/model_def_"+str(optimization_step)+".tar"
        torch.save(model_dict, path)
        print("Model saved :", path)
        return optimization_step
    else:
        return last_saved_step

def save_model(model, arg_dict, optimization_step, last_saved_step):
    if optimization_step >= last_saved_step + arg_dict["model_save_interval"]:
        model_dict = {
            'optimization_step': optimization_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
        }
        path = arg_dict["log_dir_policy"]+"/model_"+str(optimization_step)+".tar"
        torch.save(model_dict, path)
        print("Model saved :", path)
        return optimization_step
    else:
        return last_saved_step

def get_data(queue, arg_dict, model, tot_data):
    rl_data = []
    latest_data_size = arg_dict["rollout_len"]*arg_dict["batch_size"]

    if tot_data: 
        data_size = tot_data["player_div"].shape[1]
    else:
        data_size = latest_data_size

    for i in range(arg_dict["buffer_size"]):
        mini_batch_np = []
        for j in range(arg_dict["batch_size"]):
            rollout = queue.get()
            mini_batch_np.append(rollout)
        new_data = model.make_batch(mini_batch_np)
        rl_data.append(new_data)

        latest_data = new_data[4].copy()
        #latest_data_prime = new_data[4].copy()

        if tot_data and data_size < arg_dict["div_buffer_size"]:
            for key, _ in latest_data.items():
                if key == "player_div" or key == "ball" or key == "skill":
                    tot_data[key] = torch.cat([tot_data[key], latest_data[key].reshape(1, latest_data_size, -1)], dim=1)
                elif key == "hidden_div":
                    hidden1 = torch.cat([tot_data[key][0], latest_data[key][0].reshape(1, latest_data_size, -1)], dim=1)
                    hidden2 = torch.cat([tot_data[key][1], latest_data[key][1].reshape(1, latest_data_size, -1)], dim=1)
                    tot_data[key] = (hidden1, hidden2)

            #tot_data["player_div_prime"] = torch.cat([tot_data["player_div_prime"], latest_data_prime["player_div"].reshape(latest_data_size, -1)], dim=0)
            #tot_data["ball_prime"] = torch.cat([tot_data["ball_prime"], latest_data_prime["ball"].reshape(latest_data_size, -1)], dim=0)

        elif tot_data:
            for key, _ in latest_data.items():
                if key == "player_div" or key == "ball" or key == "opp_div" or key == "skill":
                    tot_data[key] = torch.cat([tot_data[key], latest_data[key].reshape(1, latest_data_size, -1)], dim=1)
                    tot_data[key] = tot_data[key][:, latest_data_size:, :]
                elif key == "hidden_div":
                    hidden1 = torch.cat([tot_data[key][0], latest_data[key][0].reshape(1, latest_data_size, -1)], dim=1)
                    hidden1 = hidden1[:, latest_data_size:, :]
                    hidden2 = torch.cat([tot_data[key][1], latest_data[key][1].reshape(1, latest_data_size, -1)], dim=1)
                    hidden2 = hidden2[:, latest_data_size:, :]
                    tot_data[key] = (hidden1, hidden2)
                
            #tot_data["player_div_prime"] = torch.cat([tot_data["player_div_prime"], latest_data_prime["player_div"].reshape(latest_data_size, -1)], dim=0)
            #tot_data["player_div_prime"] = tot_data["player_div_prime"][latest_data_size:, :]
            #tot_data["opp_div_prime"] = torch.cat([tot_data["opp_div_prime"], latest_data_prime["opp_div"].reshape(latest_data_size, -1)], dim=0)
            #tot_data["opp_div_prime"] = tot_data["opp_div_prime"][latest_data_size:, :]
            #tot_data["ball_prime"] = torch.cat([tot_data["ball_prime"], latest_data_prime["ball"].reshape(latest_data_size, -1)], dim=0)
            #tot_data["ball_prime"] = tot_data["ball_prime"][latest_data_size:, :]

        else:
            for key, _ in latest_data.items():
                if key == "player_div" or key == "ball" or key == "skill":
                    tot_data[key] = latest_data[key].reshape(1, latest_data_size, -1)
                elif key == "hidden_div":
                    hidden1 = latest_data[key][0].reshape(1, latest_data_size, -1)
                    hidden2 = latest_data[key][1].reshape(1, latest_data_size, -1)
                    tot_data[key] = (hidden1, hidden2)

            
            #tot_data["player_div_prime"] = latest_data_prime["player_div"].reshape(latest_data_size, -1)
            #tot_data["opp_div_prime"] = latest_data_prime["opp_div"].reshape(latest_data_size, -1)
            #tot_data["ball_prime"] = latest_data_prime["ball"].reshape(latest_data_size, -1)

    return rl_data, tot_data 

        
def sep_get_data(queue, arg_dict, model, tot_data, idx):
    rl_data = []
    for i in range(arg_dict["buffer_size"]):
        mini_batch_np = []
        for j in range(arg_dict["batch_size"]):
            rollout = queue.get()
            mini_batch_np.append(rollout)
        new_data = model.make_batch(mini_batch_np)
        rl_data.append(new_data)

        latest_data = new_data[0].copy()

        if tot_data and tot_data["player"].shape[1] < arg_dict["div_buffer_size"] and idx == 0:
            for key, _ in latest_data.items():
                if key != "hidden" and key != "hidden_div":
                    latest_data[key] = torch.cat([tot_data[key], latest_data[key]], dim=1)
                else:
                    hidden1 = torch.cat([tot_data[key][0], latest_data[key][0]], dim=1)
                    hidden2 = torch.cat([tot_data[key][1], latest_data[key][1]], dim=1)
                    latest_data[key] = (hidden1, hidden2)

        elif tot_data and idx == 0:
            for key, _ in latest_data.items():
                if key != "hidden" and key != "hidden_div":
                    latest_data[key] = torch.cat([tot_data[key], latest_data[key]], dim=1)
                    latest_data[key] = latest_data[key][:, arg_dict["batch_size"]:, :]
                else:
                    hidden1 = torch.cat([tot_data[key][0], latest_data[key][0]], dim=1)
                    hidden2 = torch.cat([tot_data[key][1], latest_data[key][1]], dim=1)
                    latest_data[key] = (hidden1, hidden2)


    return rl_data, latest_data 

def learner(center_div_model, center_model, queue, signal_queue, summary_queue, arg_dict, writer):
    print("Learner process started")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    imported_model = importlib.import_module("models." + arg_dict["model"])
    imported_div_model = importlib.import_module("models." + arg_dict["div_model"])
        
    div_model = imported_div_model.Model(arg_dict, device)
    div_model.load_state_dict(center_div_model.state_dict())
    div_model.optimizer.load_state_dict(center_div_model.optimizer.state_dict())
    div_model.to(device)

    imported_algo = importlib.import_module("algos." + arg_dict["algorithm"])
    imported_div_algo = importlib.import_module("algos." + arg_dict["div_algorithm"])
    model = imported_model.Model(arg_dict, device)
    model.load_state_dict(center_model.state_dict())
    model.optimizer.load_state_dict(center_model.optimizer.state_dict())
    
    algo = imported_algo.Algo(arg_dict)
    div_algo = imported_div_algo.Algo(arg_dict)
    
    for state in model.optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    model.to(device)
    
    optimization_step = 0
    if "optimization_step" in arg_dict:
        optimization_step = arg_dict["optimization_step"]
    last_saved_step = optimization_step

    n_game = 0
    loss_lst, pi_loss_lst, v_loss_lst, entropy_lst, move_entropy_lst, div_loss_lst, div_entropy_lst = [], [], [], [], [], [], []
    self_play_board = {}

    win_evaluation, score_evaluation = [], []
    tot_data = {}
    div_step = 1
    
    while True:

        if optimization_step // arg_dict["div_lr_step"] == div_step:
            div_step += 1
            signal_queue.put(1)
            div_loss, div_entropy = div_algo.train(div_model, tot_data)
            print("Diverse Network: step:", optimization_step, "loss", div_loss)

            div_loss_lst.append(div_loss)
            div_entropy_lst.append(div_entropy)
            center_div_model.load_state_dict(div_model.state_dict())
            save_div_model(div_model, arg_dict, div_step)

            _ = signal_queue.get() 

        elif queue.qsize() > arg_dict["batch_size"]*arg_dict["buffer_size"]:
            #last_saved_step = save_model(model, arg_dict, optimization_step, last_saved_step)
            
            signal_queue.put(1)
            rl_data, tot_data = get_data(queue, arg_dict, model, tot_data)
            loss, pi_loss, v_loss, entropy, move_entropy = algo.train(model, rl_data)
            optimization_step += arg_dict["batch_size"]*arg_dict["buffer_size"]*arg_dict["k_epoch"]

            print("Model: step :", optimization_step, "loss", loss, "data_q", queue.qsize(), "summary_q", summary_queue.qsize(), "buffer_size", tot_data["player_div"].shape[1])
            
            loss_lst.append(loss)
            pi_loss_lst.append(pi_loss)
            v_loss_lst.append(v_loss)
            entropy_lst.append(entropy)
            move_entropy_lst.append(move_entropy)
            center_model.load_state_dict(model.state_dict())
            
            if queue.qsize() > arg_dict["batch_size"]*arg_dict["buffer_size"]:
                print("warning. data remaining. queue size : ", queue.qsize())
            
            if summary_queue.qsize() > arg_dict["summary_game_window"]:
                win_evaluation, score_evaluation, last_saved_step = write_summary(writer, arg_dict, summary_queue, n_game, loss_lst, pi_loss_lst, 
                                                                 v_loss_lst, entropy_lst, move_entropy_lst, optimization_step, 
                                                                 self_play_board, win_evaluation, score_evaluation, div_loss_lst, div_entropy_lst, model, last_saved_step)

                loss_lst, pi_loss_lst, v_loss_lst, entropy_lst, move_entropy_lst, div_loss_lst, div_entropy_lst = [], [], [], [], [], [], []
                n_game += arg_dict["summary_game_window"]
                
            _ = signal_queue.get() 

        else:
            time.sleep(0.1)
            
def seperate_learner(i, center_div_model, center_model, queue, signal_queue, summary_queue, arg_dict, writer):
    print("Learner process started")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if i==0:
        imported_model = importlib.import_module("models." + arg_dict["model_att"])
        imported_div_model = importlib.import_module("models." + arg_dict["div_model"])
        
        div_model = imported_div_model.Model(arg_dict, device)
        div_model.load_state_dict(center_div_model.state_dict())
        div_model.optimizer.load_state_dict(center_div_model.optimizer.state_dict())
        div_model.to(device)

    else:
        imported_model = importlib.import_module("models." + arg_dict["model_def"])

    imported_algo = importlib.import_module("algos." + arg_dict["algorithm"])
    imported_div_algo = importlib.import_module("algos." + arg_dict["div_algorithm"])
    model = imported_model.Model(arg_dict, device)
    model.load_state_dict(center_model.state_dict())
    model.optimizer.load_state_dict(center_model.optimizer.state_dict())
    
    algo = imported_algo.Algo(arg_dict)
    div_algo = imported_div_algo.Algo(arg_dict)
    
    for state in model.optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    model.to(device)
    
    optimization_step = 0
    if "optimization_step" in arg_dict:
        optimization_step = arg_dict["optimization_step"]
    last_saved_step = optimization_step

    n_game = 0
    loss_lst, pi_loss_lst, v_loss_lst, entropy_lst, move_entropy_lst, div_loss_lst, div_entropy_lst = [], [], [], [], [], [], []
    self_play_board = {}

    win_evaluation, score_evaluation = [], []
    tot_data = {}
    div_step = 1
    
    while True:

        if optimization_step // arg_dict["div_lr_step"] == div_step and i == 0:
            div_step += 1
            signal_queue.put(1)
            div_loss, div_entropy = div_algo.train(div_model, tot_data)
            print("Diverse Network: step:", optimization_step, "loss", div_loss)

            div_loss_lst.append(div_loss)
            div_entropy_lst.append(div_entropy)
            center_div_model.load_state_dict(div_model.state_dict())
            save_div_model(div_model, arg_dict, div_step)

            _ = signal_queue.get() 

        elif queue.qsize() > arg_dict["batch_size"]*arg_dict["buffer_size"]:
            last_saved_step = seperate_save_model(i, model, arg_dict, optimization_step, last_saved_step)
            
            signal_queue.put(1)
            rl_data, tot_data = sep_get_data(queue, arg_dict, model, tot_data, i)
            loss, pi_loss, v_loss, entropy, move_entropy = algo.train(model, rl_data, i)
            optimization_step += arg_dict["batch_size"]*arg_dict["buffer_size"]*arg_dict["k_epoch"]

            if i == 0:
                print("Attack model: step :", optimization_step, "loss", loss, "data_q", queue.qsize(), "summary_q", summary_queue.qsize(), "buffer_size", tot_data["player"].shape[1])
            else:
                print("Defence model: step :", optimization_step, "loss", loss, "data_q", queue.qsize(), "summary_q", summary_queue.qsize())
            
            loss_lst.append(loss)
            pi_loss_lst.append(pi_loss)
            v_loss_lst.append(v_loss)
            entropy_lst.append(entropy)
            move_entropy_lst.append(move_entropy)
            center_model.load_state_dict(model.state_dict())
            
            if queue.qsize() > arg_dict["batch_size"]*arg_dict["buffer_size"]:
                print("warning. data remaining. queue size : ", queue.qsize())
            
            if summary_queue.qsize() > arg_dict["summary_game_window"]:
                win_evaluation, score_evaluation = write_summary(i, writer, arg_dict, summary_queue, n_game, loss_lst, pi_loss_lst, 
                                                                 v_loss_lst, entropy_lst, move_entropy_lst, optimization_step, 
                                                                 self_play_board, win_evaluation, score_evaluation, div_loss_lst, div_entropy_lst)

                loss_lst, pi_loss_lst, v_loss_lst, entropy_lst, move_entropy_lst, div_loss_lst, div_entropy_lst = [], [], [], [], [], [], []
                n_game += arg_dict["summary_game_window"]
                
            _ = signal_queue.get() 

        else:
            time.sleep(0.1)


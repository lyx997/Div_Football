import gfootball.env as football_env
import time, pprint, importlib, random, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp 
from os import listdir
from os.path import isfile, join
import numpy as np

from datetime import datetime, timedelta

def find_most_z_idx(prob_z):
    z_most_idx = prob_z.sort(descending=True)[1][0]
    return int(z_most_idx)

def state_to_tensor(state_dict, h_in):
    player_state = torch.from_numpy(state_dict["player"]).float().unsqueeze(0).unsqueeze(0)
    ball_state = torch.from_numpy(state_dict["ball"]).float().unsqueeze(0).unsqueeze(0)
    left_team_state = torch.from_numpy(state_dict["left_team"]).float().unsqueeze(0).unsqueeze(0)
    left_closest_state = torch.from_numpy(state_dict["left_closest"]).float().unsqueeze(0).unsqueeze(0)
    right_team_state = torch.from_numpy(state_dict["right_team"]).float().unsqueeze(0).unsqueeze(0)
    right_closest_state = torch.from_numpy(state_dict["right_closest"]).float().unsqueeze(0).unsqueeze(0)
    avail = torch.from_numpy(state_dict["avail"]).float().unsqueeze(0).unsqueeze(0)

    state_dict_tensor = {
      "player" : player_state,
      "ball" : ball_state,
      "left_team" : left_team_state,
      "left_closest" : left_closest_state,
      "right_team" : right_team_state,
      "right_closest" : right_closest_state,
      "avail" : avail,
      "hidden" : h_in
    }
    return state_dict_tensor


def get_action(a_prob, m_prob):
    
    a = Categorical(a_prob).sample().item()    
    m, need_m = 0, 0
    prob_selected_a = a_prob[0][0][a].item()
    prob_selected_m = 0
    if a==0:
        real_action = a
        prob = prob_selected_a
    elif a==1:
        m = Categorical(m_prob).sample().item()
        need_m = 1
        real_action = m + 1
        prob_selected_m = m_prob[0][0][m].item()
        prob = prob_selected_a* prob_selected_m
    else:
        real_action = a + 7
        prob = prob_selected_a

    assert prob != 0, 'prob 0 ERROR!!!! a : {}, m:{}  {}, {}'.format(a,m,prob_selected_a,prob_selected_m)
    
    return real_action, a, m, need_m, prob, prob_selected_a, prob_selected_m

def actor(actor_num, center_div_model, center_model, data_queue, signal_queue, summary_queue, arg_dict):
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    print("Actor process {} started".format(actor_num))
    fe_module = importlib.import_module("encoders." + arg_dict["encoder"])
    rewarder = importlib.import_module("rewarders." + arg_dict["rewarder"])
    imported_model = importlib.import_module("models." + arg_dict["model"])
    
    fe = fe_module.FeatureEncoder()
    model = imported_model.Model(arg_dict)
    model.load_state_dict(center_model.state_dict())
    
    env = football_env.create_environment(env_name=arg_dict["env"], representation="raw", stacked=False, logdir='/tmp/football', \
                                          write_goal_dumps=False, write_full_episode_dumps=False, render=False)
    n_epi = 0
    rollout = []
    while True: # episode loop
        env.reset()   
        done = False
        steps, score, tot_reward, win = 0, 0, 0, 0
        n_epi += 1
        h_out = (torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float), 
                 torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float))
        
        loop_t, forward_t, wait_t = 0.0, 0.0, 0.0
        obs = env.observation()
        
        while not done:  # step loop
            init_t = time.time()
            
            is_stopped = False
            while signal_queue.qsize() > 0:
                time.sleep(0.02)
                is_stopped = True
            if is_stopped:
                model.load_state_dict(center_model.state_dict())
            wait_t += time.time() - init_t
            
            h_in = h_out
            state_dict = fe.encode(obs[0])
            state_dict_tensor = state_to_tensor(state_dict, h_in)
            
            t1 = time.time()
            with torch.no_grad():
                a_prob, m_prob, _, h_out = model(state_dict_tensor)
            forward_t += time.time()-t1 
            real_action, a, m, need_m, prob, prob_selected_a, prob_selected_m = get_action(a_prob, m_prob)

            prev_obs = obs
            obs, rew, done, info = env.step(real_action)
            fin_r = rewarder.calc_reward(rew, prev_obs[0], obs[0])
            state_prime_dict = fe.encode(obs[0])
            
            (h1_in, h2_in) = h_in
            (h1_out, h2_out) = h_out
            state_dict["hidden"] = (h1_in.numpy(), h2_in.numpy())
            state_prime_dict["hidden"] = (h1_out.numpy(), h2_out.numpy())
            transition = (state_dict, a, m, fin_r, state_prime_dict, prob, done, need_m)
            rollout.append(transition)
            if len(rollout) == arg_dict["rollout_len"]:
                data_queue.put(rollout)
                rollout = []
                model.load_state_dict(center_model.state_dict())

            steps += 1
            score += rew
            tot_reward += fin_r
          
            loop_t += time.time()-init_t
            
            if done:
                if score > 0:
                    win = 1
                print("score",score,"total reward",tot_reward)
                summary_data = (win, score, tot_reward, steps, 0, loop_t/steps, forward_t/steps, wait_t/steps)
                summary_queue.put(summary_data)

def select_opponent(arg_dict):
    onlyfiles_lst = [f for f in listdir(arg_dict["log_dir_policy"]) if isfile(join(arg_dict["log_dir_policy"], f))]
    model_num_lst = []
    for file_name in onlyfiles_lst:
        if file_name[:6] == "model_":
            model_num = file_name[6:]
            model_num = model_num[:-4]
            model_num_lst.append(int(model_num))
    model_num_lst.sort()
            
    coin = random.random()
    if coin<arg_dict["latest_ratio"]:
        if len(model_num_lst) > arg_dict["latest_n_model"]:
            opp_model_num = random.randint(len(model_num_lst)-arg_dict["latest_n_model"],len(model_num_lst)-1)
        else:
            opp_model_num = len(model_num_lst)-1
    else:
        opp_model_num = random.randint(0,len(model_num_lst)-1)
        
    model_name = "/model_"+str(model_num_lst[opp_model_num])+".tar"
    opp_model_path = arg_dict["log_dir_policy"] + model_name
    return opp_model_num, opp_model_path
                
                
def actor_self(actor_num, div_model, center_model, data_queue, signal_queue, summary_queue, arg_dict):
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    print("Actor process {} started".format(actor_num))
    cpu_device = torch.device('cpu')
    fe_module = importlib.import_module("encoders." + arg_dict["encoder"])
    rewarder = importlib.import_module("rewarders." + arg_dict["rewarder"])
    imported_model = importlib.import_module("models." + arg_dict["model"])
    imported_div_model = importlib.import_module("models." + arg_dict["div_model"])
    
    fe = fe_module.FeatureEncoder()
    state_to_tensor = fe_module.state_to_tensor
    
    model = imported_model.Model(arg_dict)
    model.load_state_dict(center_model.state_dict())

    model_div = imported_div_model.Model(arg_dict)
    model_div.load_state_dict(div_model.state_dict())

    opp_model = imported_model.Model(arg_dict)
    opp_model.load_state_dict(center_model.state_dict())

    env = football_env.create_environment(env_name=arg_dict['env'], representation="raw", stacked=False, logdir=arg_dict["log_dir_dump_left"], \
                                          number_of_left_players_agent_controls=1,
                                          number_of_right_players_agent_controls=1,
                                          write_goal_dumps=False, write_full_episode_dumps=False, render=False)
    n_epi = 0
    rollout = []
    
    while True: # episode loop
        opp_model_num, opp_model_path = select_opponent(arg_dict)

        checkpoint = torch.load(opp_model_path, map_location=cpu_device)
        opp_model.load_state_dict(checkpoint['model_state_dict'])
        #print("Current Opponent attack model Num:{}, Path:{} successfully loaded".format(opp_att_model_num, opp_att_model_path))
        #print("Current Opponent defence model Num:{}, Path:{} successfully loaded".format(opp_def_model_num, opp_def_model_path))
        del checkpoint

        [obs, opp_obs] = env.reset()   

        done = False
        steps, score, tot_reward, win= 0, 0, 0, 0
        n_epi += 1
        h_out = (torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float), 
                torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float))
        h_div_out = (torch.zeros([1, 1, arg_dict["div_lstm_size"]], dtype=torch.float), 
                 torch.zeros([1, 1, arg_dict["div_lstm_size"]], dtype=torch.float)) 
        opp_h_out = (torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float), 
                     torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float))
        opp_h_div_out = (torch.zeros([1, 1, arg_dict["div_lstm_size"]], dtype=torch.float), 
                     torch.zeros([1, 1, arg_dict["div_lstm_size"]], dtype=torch.float))

        skill = torch.zeros([1,1, arg_dict["div_num"]], dtype=torch.float)
        skill_num = random.randint(0, arg_dict["div_num"]-1)
        skill[:,:,skill_num] = 1.0
        skill_acc, skill_steps = 0.0, 0.0
        active_num = obs["active"]

        opp_skill = torch.zeros([1,1, arg_dict["div_num"]], dtype=torch.float)
        opp_skill_num = random.randint(0, arg_dict["div_num"]-1)
        opp_skill[:,:,opp_skill_num] = 1.0
        opp_num = opp_obs["active"]
        
        loop_t, forward_t, wait_t = 0.0, 0.0, 0.0
        ball_owned_team = obs["ball_owned_team"] #-1

        while not done:

            init_t = time.time()

            is_stopped = False
            while signal_queue.qsize() > 0:
                time.sleep(0.02)
                is_stopped = True
            if is_stopped:
                model.load_state_dict(center_model.state_dict())
            wait_t += time.time() - init_t

            h_in = h_out
            h_div_in = h_div_out
            opp_h_in = opp_h_out
            opp_h_div_in = opp_h_div_out
            state_dict = fe.encode(obs)
            state_dict_tensor = state_to_tensor(state_dict, h_in, skill, h_div_in)
            opp_state_dict = fe.encode(opp_obs)
            opp_state_dict_tensor = state_to_tensor(opp_state_dict, opp_h_in, opp_skill, opp_h_div_in)

            t1 = time.time()
            with torch.no_grad():
                a_prob, m_prob, _, h_out, _, _ = model(state_dict_tensor)
                opp_a_prob, opp_m_prob, _, opp_h_out, _, _ = opp_model(opp_state_dict_tensor)
            forward_t += time.time()-t1 
            real_action, a, m, need_m, prob, _, _ = get_action(a_prob, m_prob)
            opp_real_action, _, _, _, _, _, _ = get_action(opp_a_prob, opp_m_prob)

            prev_obs = obs

            [obs, opp_obs], [rew, _], done, info = env.step([real_action, opp_real_action])

            ball_owned_team = obs["ball_owned_team"]
            active_num = obs["active"]
            opp_num = opp_obs["active"]

            state_prime_dict = fe.encode(obs)
            state_prime_dict_tensor = state_to_tensor(state_prime_dict, h_in, skill, h_div_in)
            #state_dict_tensor["player_div_prime"] = state_prime_dict_tensor["player_div"]
            #state_dict_tensor["opp_div_prime"] = state_prime_dict_tensor["opp_div"]
            #state_dict_tensor["ball_prime"] = state_prime_dict_tensor["ball"]

            with torch.no_grad():
                prob_z, h_div_out = model_div(state_prime_dict_tensor)
                z_most_idx = find_most_z_idx(prob_z)

            if ball_owned_team == 0:
                fin_r = rewarder.calc_reward(rew, prev_obs, obs, float(prob_z[skill_num])*(arg_dict["div_num"]))
            else:
                fin_r = rewarder.calc_reward(rew, prev_obs, obs, None)

            (h1_in, h2_in) = h_in
            (h1_out, h2_out) = h_out
            (h1_div_in, h2_div_in) = h_div_in
            state_dict["hidden"] = (h1_in.numpy(), h2_in.numpy())
            state_prime_dict["hidden"] = (h1_out.numpy(), h2_out.numpy())
            state_prime_dict["hidden_div"] = (h1_div_in.numpy(), h2_div_in.numpy())
            state_dict["skill"] = skill.numpy()
            state_prime_dict["skill"] = skill.numpy()

            transition = (state_dict, a, m, fin_r, state_prime_dict, prob, prob_z.numpy(), done, need_m)

            rollout.append(transition)
            if len(rollout) == arg_dict["rollout_len"]:
                data_queue.put(rollout)
                rollout = []
                model.load_state_dict(center_model.state_dict())
                
            if z_most_idx == skill_num and ball_owned_team == 0:
                skill_acc += 1
            if ball_owned_team == 0:
                skill_steps += 1

            steps += 1
            score += rew
            tot_reward += fin_r

            loop_t += time.time()-init_t

            if done:
                if score > 0:
                    win = 1
                print("Model score",score,"total reward",tot_reward, "opp_num", opp_model_num, "skill", skill_num)
                summary_data = (win, score, tot_reward, steps, str(opp_model_num), loop_t/steps, forward_t/steps, wait_t/steps, skill_acc/skill_steps)
                summary_queue.put(summary_data)
             

def seperate_actor(actor_num, center_model, data_queue, signal_queue, summary_queue, arg_dict):
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    print("Actor process {} started".format(actor_num))
    fe_module = importlib.import_module("encoders." + arg_dict["encoder"])
    rewarder = importlib.import_module("rewarders." + arg_dict["rewarder"])
    imported_model_att = importlib.import_module("models." + arg_dict["model_att"])
    imported_model_def = importlib.import_module("models." + arg_dict["model_def"])
    
    fe = fe_module.FeatureEncoder()
    state_to_tensor = fe_module.state_to_tensor
    
    model_att = imported_model_att.Model(arg_dict)
    model_att.load_state_dict(center_model[0].state_dict())

    model_def = imported_model_def.Model(arg_dict)
    model_def.load_state_dict(center_model[1].state_dict())
    
    env_left = football_env.create_environment(env_name=arg_dict['env'], representation="raw", stacked=False, logdir=arg_dict["log_dir_dump_left"], \
                                          number_of_left_players_agent_controls=1,
                                          number_of_right_players_agent_controls=0,
                                          write_goal_dumps=False, write_full_episode_dumps=False, render=False)
    env_right = football_env.create_environment(env_name=arg_dict['env'], representation="raw", stacked=False, logdir=arg_dict["log_dir_dump_right"], \
                                          number_of_left_players_agent_controls=0,
                                          number_of_right_players_agent_controls=1,
                                          write_goal_dumps=False, write_full_episode_dumps=False, render=False)
    n_epi = 0
    rollout_att = []
    rollout_def = []
    
    while True: # episode loop
        seed = random.random()
        if seed < 0.5:
            env_left.reset()   
            obs = env_left.observation()
            our_team = 0
        else:
            env_right.reset()   
            obs = env_right.observation()
            our_team = 1

        done = False
        steps, score, tot_reward, win= 0, 0, 0, 0
        n_epi += 1
        h_out = (torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float), 
                 torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float))
        
        loop_t, forward_t, wait_t = 0.0, 0.0, 0.0
        ball_owned_team = obs[0]["ball_owned_team"] #-1

        while not done:

            #attack model
            while not done:  # step loop
                init_t = time.time()

                if ball_owned_team == 1: #ball owned by opp change to model_def
                    break

                is_stopped = False
                while signal_queue[0].qsize() > 0:
                    time.sleep(0.02)
                    is_stopped = True
                if is_stopped:
                    model_att.load_state_dict(center_model[0].state_dict())
                wait_t += time.time() - init_t

                h_in = h_out
                state_dict = fe.encode(obs[0])
                state_dict_tensor = state_to_tensor(state_dict, h_in)

                t1 = time.time()
                with torch.no_grad():
                    a_prob, m_prob, _, h_out = model_att(state_dict_tensor)
                forward_t += time.time()-t1 
                real_action, a, m, need_m, prob, _, _ = get_action(a_prob, m_prob)

                prev_obs = obs

                if our_team == 0:
                    obs, rew, done, info = env_left.step(real_action)
                else:
                    obs, rew, done, info = env_right.step(real_action)

                ball_owned_team = obs[0]["ball_owned_team"]
                fin_r = rewarder.calc_reward(rew, prev_obs[0], obs[0])
                state_prime_dict = fe.encode(obs[0])

                (h1_in, h2_in) = h_in
                (h1_out, h2_out) = h_out
                state_dict["hidden"] = (h1_in.numpy(), h2_in.numpy())
                state_prime_dict["hidden"] = (h1_out.numpy(), h2_out.numpy())

                if ball_owned_team == 1:
                    transition = (state_dict, a, m, fin_r, state_prime_dict, prob, done, need_m) #change to model defence
                else:
                    transition = (state_dict, a, m, fin_r, state_prime_dict, prob, done, need_m)

                rollout_att.append(transition)
                if len(rollout_att) == arg_dict["rollout_len"]:
                    data_queue[0].put(rollout_att)
                    rollout_att = []
                    model_att.load_state_dict(center_model[0].state_dict())

                steps += 1
                score += rew
                tot_reward += fin_r

                loop_t += time.time()-init_t

                if done:
                    if score > 0:
                        win = 1
                    if our_team == 0:
                        print("model in left score",score,"total reward",tot_reward)
                    else:
                        print("model in right score",score,"total reward",tot_reward)
                    summary_data = (win, score, tot_reward, steps, 0, loop_t/steps, forward_t/steps, wait_t/steps)
                    summary_queue.put(summary_data)

             
            #defence model
            while not done:  # step loop
                init_t = time.time()

                if ball_owned_team == 0: #ball owned by us so change to model_att
                    break

                is_stopped = False
                while signal_queue[1].qsize() > 0:
                    time.sleep(0.02)
                    is_stopped = True
                if is_stopped:
                    model_def.load_state_dict(center_model[1].state_dict())
                wait_t += time.time() - init_t

                h_in = h_out
                state_dict = fe.encode(obs[0])
                state_dict_tensor = state_to_tensor(state_dict, h_in)

                t1 = time.time()
                with torch.no_grad():
                    a_prob, m_prob, _, h_out = model_att(state_dict_tensor)
                forward_t += time.time()-t1 
                real_action, a, m, need_m, prob, _, _ = get_action(a_prob, m_prob)

                prev_obs = obs

                if our_team == 0:
                    obs, rew, done, info = env_left.step(real_action)
                else:
                    obs, rew, done, info = env_right.step(real_action)

                ball_owned_team = obs[0]["ball_owned_team"]
                fin_r = rewarder.calc_reward(rew, prev_obs[0], obs[0])
                state_prime_dict = fe.encode(obs[0])

                (h1_in, h2_in) = h_in
                (h1_out, h2_out) = h_out
                state_dict["hidden"] = (h1_in.numpy(), h2_in.numpy())
                state_prime_dict["hidden"] = (h1_out.numpy(), h2_out.numpy())

                if ball_owned_team == 0:
                    transition = (state_dict, a, m, fin_r, state_prime_dict, prob, done, need_m) # change to model attack
                else:
                    transition = (state_dict, a, m, fin_r, state_prime_dict, prob, done, need_m)

                rollout_def.append(transition)
                if len(rollout_def) == arg_dict["rollout_len"]:
                    data_queue[1].put(rollout_def)
                    rollout_def = []
                    model_def.load_state_dict(center_model[1].state_dict())

                steps += 1
                score += rew
                tot_reward += fin_r

                loop_t += time.time()-init_t

                if done:
                    if score > 0:
                        win = 1
                    if our_team == 0:
                        print("model in left score",score,"total reward",tot_reward)
                    else:
                        print("model in right score",score,"total reward",tot_reward)
                    summary_data = (win, score, tot_reward, steps, 0, loop_t/steps, forward_t/steps, wait_t/steps)
                    summary_queue.put(summary_data)

def select_att_opponent(arg_dict):
    onlyfiles_lst = [f for f in listdir(arg_dict["log_dir_att"]) if isfile(join(arg_dict["log_dir_att"], f))]
    model_num_lst = []
    for file_name in onlyfiles_lst:
        if file_name[:10] == "model_att_":
            model_num = file_name[10:]
            model_num = model_num[:-4]
            model_num_lst.append(int(model_num))
    model_num_lst.sort()
            
    coin = random.random()
    if coin<arg_dict["latest_ratio"]:
        if len(model_num_lst) > arg_dict["latest_n_model"]:
            opp_model_num = random.randint(len(model_num_lst)-arg_dict["latest_n_model"],len(model_num_lst)-1)
        else:
            opp_model_num = len(model_num_lst)-1
    else:
        opp_model_num = random.randint(0,len(model_num_lst)-1)
        
    model_name = "/model_att_"+str(model_num_lst[opp_model_num])+".tar"
    opp_model_path = arg_dict["log_dir_att"] + model_name
    return opp_model_num, opp_model_path

def select_def_opponent(arg_dict):
    onlyfiles_lst = [f for f in listdir(arg_dict["log_dir_def"]) if isfile(join(arg_dict["log_dir_def"], f))]
    model_num_lst = []
    for file_name in onlyfiles_lst:
        if file_name[:10] == "model_def_":
            model_num = file_name[10:]
            model_num = model_num[:-4]
            model_num_lst.append(int(model_num))
    model_num_lst.sort()
    opp_model_num = model_num_lst[-1]
    #coin = random.random()
    #if coin<arg_dict["latest_ratio"]:
    #    if len(model_num_lst) > arg_dict["latest_n_model"]:
    #        opp_model_num = random.randint(len(model_num_lst)-arg_dict["latest_n_model"],len(model_num_lst)-1)
    #    else:
    #        opp_model_num = len(model_num_lst)-1
    #else:
    #    opp_model_num = random.randint(0,len(model_num_lst)-1)
        
    model_name = "/model_def_"+str(opp_model_num)+".tar"
    opp_model_path = arg_dict["log_dir_def"] + model_name
    return opp_model_num, opp_model_path


def seperate_actor_self(actor_num, div_model, center_model, data_queue, signal_queue, summary_queue, arg_dict):
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    print("Actor process {} started".format(actor_num))
    cpu_device = torch.device('cpu')
    fe_module = importlib.import_module("encoders." + arg_dict["encoder"])
    rewarder = importlib.import_module("rewarders." + arg_dict["rewarder"])
    att_rewarder = importlib.import_module("rewarders." + arg_dict["att_rewarder"])
    imported_model_att = importlib.import_module("models." + arg_dict["model_att"])
    imported_model_def = importlib.import_module("models." + arg_dict["model_def"])
    imported_div_model = importlib.import_module("models." + arg_dict["div_model"])
    
    fe = fe_module.FeatureEncoder()
    state_to_tensor = fe_module.state_to_tensor
    
    model_att = imported_model_att.Model(arg_dict)
    model_att.load_state_dict(center_model[0].state_dict())

    model_def = imported_model_def.Model(arg_dict)
    model_def.load_state_dict(center_model[1].state_dict())

    model_div = imported_div_model.Model(arg_dict)
    model_div.load_state_dict(div_model.state_dict())

    opp_model_att = imported_model_att.Model(arg_dict)
    opp_model_att.load_state_dict(center_model[0].state_dict())

    opp_model_def = imported_model_def.Model(arg_dict)
    opp_model_def.load_state_dict(center_model[1].state_dict())
    
    env = football_env.create_environment(env_name=arg_dict['env'], representation="raw", stacked=False, logdir=arg_dict["log_dir_dump_left"], \
                                          number_of_left_players_agent_controls=1,
                                          number_of_right_players_agent_controls=1,
                                          write_goal_dumps=False, write_full_episode_dumps=False, render=False)
    n_epi = 0
    rollout_att = []
    rollout_def = []
    
    while True: # episode loop
        opp_att_model_num, opp_att_model_path = select_att_opponent(arg_dict)
        opp_def_model_num, opp_def_model_path = select_def_opponent(arg_dict)

        att_checkpoint = torch.load(opp_att_model_path, map_location=cpu_device)
        opp_model_att.load_state_dict(att_checkpoint['model_state_dict'])
        def_checkpoint = torch.load(opp_def_model_path, map_location=cpu_device)
        opp_model_def.load_state_dict(def_checkpoint['model_state_dict'])
        #print("Current Opponent attack model Num:{}, Path:{} successfully loaded".format(opp_att_model_num, opp_att_model_path))
        #print("Current Opponent defence model Num:{}, Path:{} successfully loaded".format(opp_def_model_num, opp_def_model_path))
        del att_checkpoint
        del def_checkpoint

        [obs, opp_obs] = env.reset()   

        done = False
        steps, score, tot_reward, win= 0, 0, 0, 0
        n_epi += 1
        h_out = (torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float), 
                torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float))
        h_div_out = (torch.zeros([1, 1, arg_dict["div_lstm_size"]], dtype=torch.float), 
                 torch.zeros([1, 1, arg_dict["div_lstm_size"]], dtype=torch.float)) 
        opp_h_out = (torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float), 
                     torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float))
        opp_h_div_out = (torch.zeros([1, 1, arg_dict["div_lstm_size"]], dtype=torch.float), 
                     torch.zeros([1, 1, arg_dict["div_lstm_size"]], dtype=torch.float))

        skill = torch.zeros([1,1, arg_dict["div_num"]], dtype=torch.float)
        skill_num = random.randint(0, arg_dict["div_num"]-1)
        skill[:,:,skill_num] = 1.0
        skill_acc, skill_steps = 0.0, 0.0
        
        loop_t, forward_t, wait_t = 0.0, 0.0, 0.0
        ball_owned_team = obs["ball_owned_team"] #-1

        while not done:

            #attack model
            while not done:  # step loop
                init_t = time.time()

                if ball_owned_team == 1: #ball owned by opp change to model_def
                    break

                is_stopped = False
                while signal_queue[0].qsize() > 0:
                    time.sleep(0.02)
                    is_stopped = True
                if is_stopped:
                    model_att.load_state_dict(center_model[0].state_dict())
                wait_t += time.time() - init_t

                h_in = h_out
                h_div_in = h_div_out
                opp_h_in = opp_h_out
                opp_h_div_in = opp_h_div_out
                state_dict = fe.encode(obs)
                state_dict_tensor = state_to_tensor(state_dict, h_in, skill, h_div_in)
                opp_state_dict = fe.encode(opp_obs)
                opp_state_dict_tensor = state_to_tensor(opp_state_dict, opp_h_in, skill, 0)

                t1 = time.time()
                with torch.no_grad():
                    a_prob, m_prob, _, h_out, _, _ = model_att(state_dict_tensor)
                    prob_z, h_div_out = model_div(state_dict_tensor)
                    z_most_idx = find_most_z_idx(prob_z)
                    opp_a_prob, opp_m_prob, _, opp_h_out = opp_model_def(opp_state_dict_tensor)
                forward_t += time.time()-t1 
                real_action, a, m, need_m, prob, _, _ = get_action(a_prob, m_prob)
                opp_real_action, _, _, _, _, _, _ = get_action(opp_a_prob, opp_m_prob)

                prev_obs = obs

                [obs, opp_obs], [rew, _], done, info = env.step([real_action, opp_real_action])

                ball_owned_team = obs["ball_owned_team"]
                opp_num = opp_obs["active"]
                fin_r = att_rewarder.calc_reward(rew, prev_obs, obs, float(prob_z[skill_num])*(arg_dict["div_num"]))
                state_prime_dict = fe.encode(obs)

                (h1_in, h2_in) = h_in
                (h1_out, h2_out) = h_out
                (h1_div_in, h2_div_in) = h_div_in
                state_dict["hidden"] = (h1_in.numpy(), h2_in.numpy())
                state_prime_dict["hidden"] = (h1_out.numpy(), h2_out.numpy())
                state_dict["hidden_div"] = (h1_div_in.numpy(), h2_div_in.numpy())
                state_dict["skill"] = skill.numpy()
                state_prime_dict["skill"] = skill.numpy()

                if ball_owned_team == 1:
                    transition = (state_dict, a, m, fin_r, state_prime_dict, prob, prob_z.numpy(), True, need_m) #change to model defence
                else:
                    transition = (state_dict, a, m, fin_r, state_prime_dict, prob, prob_z.numpy(), done, need_m)

                rollout_att.append(transition)
                if len(rollout_att) == arg_dict["rollout_len"]:
                    data_queue[0].put(rollout_att)
                    rollout_att = []
                    model_att.load_state_dict(center_model[0].state_dict())
                
                if z_most_idx == skill_num:
                    skill_acc += 1

                skill_steps += 1

                steps += 1
                score += rew
                tot_reward += fin_r

                loop_t += time.time()-init_t

                if done:
                    if score > 0:
                        win = 1
                    print("Model score",score,"total reward",tot_reward)
                    summary_data = (win, score, tot_reward, steps, str(opp_att_model_num), loop_t/steps, forward_t/steps, wait_t/steps, skill_acc/skill_steps)
                    summary_queue.put(summary_data)

             
            #defence model
            while not done:  # step loop
                init_t = time.time()

                if ball_owned_team == 0: #ball owned by us so change to model_att
                    #skill = torch.zeros([1,1, arg_dict["div_num"]], dtype=torch.float)
                    #skill_num = random.randint(0, arg_dict["div_num"]-1)
                    #skill[:,:,skill_num] = 1.0
                    break

                is_stopped = False
                while signal_queue[1].qsize() > 0:
                    time.sleep(0.02)
                    is_stopped = True
                if is_stopped:
                    model_def.load_state_dict(center_model[1].state_dict())
                wait_t += time.time() - init_t

                h_in = h_out
                opp_h_in = opp_h_out
                state_dict = fe.encode(obs)
                state_dict_tensor = state_to_tensor(state_dict, h_in, skill, 0)
                opp_state_dict = fe.encode(opp_obs)
                opp_state_dict_tensor = state_to_tensor(opp_state_dict, opp_h_in, skill, 0)

                t1 = time.time()
                with torch.no_grad():
                    a_prob, m_prob, _, h_out = model_def(state_dict_tensor)
                    opp_a_prob, opp_m_prob, _, opp_h_out, _, _ = opp_model_att(opp_state_dict_tensor)
                forward_t += time.time()-t1 

                real_action, a, m, need_m, prob, _, _ = get_action(a_prob, m_prob)
                opp_real_action, _, _, _, _, _, _ = get_action(opp_a_prob, opp_m_prob)

                prev_obs = obs

                [obs, opp_obs], [rew, _], done, info = env.step([real_action, opp_real_action])

                ball_owned_team = obs["ball_owned_team"]
                opp_num = opp_obs["active"]
                fin_r = rewarder.calc_reward(rew, prev_obs, obs, opp_num)
                state_prime_dict = fe.encode(obs)

                (h1_in, h2_in) = h_in
                (h1_out, h2_out) = h_out
                state_dict["hidden"] = (h1_in.numpy(), h2_in.numpy())
                state_prime_dict["hidden"] = (h1_out.numpy(), h2_out.numpy())
                state_dict["skill"] = skill.numpy()
                state_prime_dict["skill"] = skill.numpy()

                if ball_owned_team == 0:
                    transition = (state_dict, a, m, fin_r, state_prime_dict, prob, True, need_m) # change to model attack
                else:
                    transition = (state_dict, a, m, fin_r, state_prime_dict, prob, done, need_m)

                rollout_def.append(transition)
                if len(rollout_def) == arg_dict["rollout_len"]:
                    data_queue[1].put(rollout_def)
                    rollout_def = []
                    model_def.load_state_dict(center_model[1].state_dict())

                steps += 1
                score += rew
                tot_reward += fin_r

                loop_t += time.time()-init_t

                if done:
                    if score > 0:
                        win = 1
                    print("Model score",score,"total reward",tot_reward)
                    summary_data = (win, score, tot_reward, steps, str(opp_att_model_num), loop_t/steps, forward_t/steps, wait_t/steps, skill_acc/skill_steps)
                    summary_queue.put(summary_data)
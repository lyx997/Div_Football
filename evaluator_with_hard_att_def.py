import gfootball.env as football_env
import time, pprint, importlib, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp 
from os import listdir
from os.path import isfile, join
from datetime import datetime, timedelta

def find_most_z_idx(prob_z):
    z_most_idx = prob_z.sort(descending=True)[1][0]
    return int(z_most_idx)

def split_att_idx(all_sorted_idx):
    team_att_idx_list = []
    opp_att_idx_list = []
    for idx in all_sorted_idx:
        if idx > 10:
            opp_att_idx_list.append(idx % 11)
        else:
            team_att_idx_list.append(idx)
    
    return team_att_idx_list, opp_att_idx_list

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

def seperate_evaluator(center_model, signal_queue, summary_queue, arg_dict):
    print("Evaluator process started")
    fe_module1 = importlib.import_module("encoders." + arg_dict["encoder"])
    #fe_module2 = importlib.import_module("encoders." + "encoder_basic")
    rewarder = importlib.import_module("rewarders." + arg_dict["rewarder"])
    #opp_import_model = importlib.import_module("models." + "conv1d_self")
    
    fe1 = fe_module1.FeatureEncoder()
    state_to_tensor1 = fe_module1.state_to_tensor
    #fe2 = fe_module2.FeatureEncoder()
    #state_to_tensor2 = fe_module2.state_to_tensor
    model_att = center_model[0]
    model_def = center_model[1]
    #opp_model = opp_import_model.Model(arg_dict)
    #opp_model_checkpoint = torch.load(arg_dict["env_evaluation"])
    #opp_model.load_state_dict(opp_model_checkpoint['model_state_dict'])

    #
    #env = football_env.create_environment(env_name=arg_dict["env"], number_of_right_players_agent_controls=1, representation="raw", \
    #                                      stacked=False, logdir='/tmp/football', write_goal_dumps=False, write_full_episode_dumps=False, \
    #                                      render=False)
    env_left = football_env.create_environment(env_name=arg_dict["env_evaluation"], representation="raw", stacked=False, logdir=arg_dict["log_dir_dump_left"], \
                                          number_of_left_players_agent_controls=1,
                                          number_of_right_players_agent_controls=0,
                                          write_goal_dumps=False, write_full_episode_dumps=True, render=False, write_video=True)
    env_right = football_env.create_environment(env_name=arg_dict["env_evaluation"], representation="raw", stacked=False, logdir=arg_dict["log_dir_dump_right"], \
                                          number_of_left_players_agent_controls=0,
                                          number_of_right_players_agent_controls=1,
                                          write_goal_dumps=False, write_full_episode_dumps=True, render=False, write_video=True)
    n_epi = 0
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
        steps, score, tot_reward, win = 0, 0, 0, 0
        n_epi += 1
        h_out = (torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float), 
                 torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float))
        
        loop_t, forward_t, wait_t = 0.0, 0.0, 0.0

        ball_owned_team = obs[0]["ball_owned_team"] #-1

        while not done:  # step loop
        
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
                    pass
                wait_t += time.time() - init_t
                
                h_in = h_out
                state_dict = fe1.encode(obs[0])
                state_dict_tensor = state_to_tensor1(state_dict, h_in)
                
                t1 = time.time()
                with torch.no_grad():
                    a_prob, m_prob, _, h_out = model_att(state_dict_tensor)
                forward_t += time.time()-t1 
    
                real_action, a, m, need_m, prob, prob_selected_a, prob_selected_m = get_action(a_prob, m_prob)
    
                prev_obs = obs
                if our_team == 0:
                    obs, rew, done, info = env_left.step([real_action])
                else:
                    obs, rew, done, info = env_right.step([real_action])

                rew = rew[0]
                ball_owned_team = obs[0]["ball_owned_team"]
                fin_r = rewarder.calc_reward(rew, prev_obs[0], obs[0])
                state_prime_dict = fe1.encode(obs[0])
                

                (h1_in, h2_in) = h_in
                (h1_out, h2_out) = h_out
                state_dict["hidden"] = (h1_in.numpy(), h2_in.numpy())
                state_prime_dict["hidden"] = (h1_out.numpy(), h2_out.numpy())
                transition = (state_dict, a, m, fin_r, state_prime_dict, prob, done, need_m)
    
                steps += 1
                score += rew
                tot_reward += fin_r
                
                loop_t += time.time()-init_t
                
                if done:
                    if score > 0:
                        win = 1

                    if our_team == 0:
                        print("model in left evaluate with ", arg_dict["env_evaluation"]," model: score",score,"total reward",tot_reward)
                    else:
                        print("model in right evaluate with ", arg_dict["env_evaluation"]," model: score",score,"total reward",tot_reward)
                    summary_data = (win, score, tot_reward, steps, arg_dict['env_evaluation'], loop_t/steps, forward_t/steps, wait_t/steps)
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
                    pass
                wait_t += time.time() - init_t
                
                h_in = h_out
                state_dict = fe1.encode(obs[0])
                state_dict_tensor = state_to_tensor1(state_dict, h_in)
                
                t1 = time.time()
                with torch.no_grad():
                    a_prob, m_prob, _, h_out = model_def(state_dict_tensor)
                forward_t += time.time()-t1 
    
                real_action, a, m, need_m, prob, prob_selected_a, prob_selected_m = get_action(a_prob, m_prob)
    
                prev_obs = obs
                if our_team == 0:
                    obs, rew, done, info = env_left.step([real_action])
                else:
                    obs, rew, done, info = env_right.step([real_action])

                rew = rew[0]
                ball_owned_team = obs[0]["ball_owned_team"]
                fin_r = rewarder.calc_reward(rew, prev_obs[0], obs[0])
                state_prime_dict = fe1.encode(obs[0])
                
                (h1_in, h2_in) = h_in
                (h1_out, h2_out) = h_out
                state_dict["hidden"] = (h1_in.numpy(), h2_in.numpy())
                state_prime_dict["hidden"] = (h1_out.numpy(), h2_out.numpy())
                transition = (state_dict, a, m, fin_r, state_prime_dict, prob, done, need_m)
    
                steps += 1
                score += rew
                tot_reward += fin_r
                
                loop_t += time.time()-init_t
                
                if done:
                    if score > 0:
                        win = 1
                    if our_team == 0:
                        print("model in left evaluate with ", arg_dict["env_evaluation"]," model: score",score,"total reward",tot_reward)
                    else:
                        print("model in right evaluate with ", arg_dict["env_evaluation"]," model: score",score,"total reward",tot_reward)
                    summary_data = (win, score, tot_reward, steps, arg_dict['env_evaluation'], loop_t/steps, forward_t/steps, wait_t/steps)
                    summary_queue.put(summary_data)

def test(center_model, div_idx, arg_dict):
    print("Test skill:", div_idx, "process started")
    fe_module1 = importlib.import_module("encoders." + arg_dict["encoder"])
    
    fe1 = fe_module1.FeatureEncoder()
    state_to_tensor1 = fe_module1.state_to_tensor
    model = center_model
    
    env_left = football_env.create_environment(env_name=arg_dict["env_evaluation"], representation="raw", stacked=False, logdir=arg_dict["log_dir_dump_left_skill_"+str(div_idx)], \
                                          number_of_left_players_agent_controls=1,
                                          number_of_right_players_agent_controls=0,
                                          write_goal_dumps=arg_dict["write_goal_dumps"], write_full_episode_dumps=arg_dict["write_full_episode_dumps"], render=False, write_video=True)
    n_epi = 0
    while True: # episode loop

        done = False
        steps, score = 0, 0
        n_epi += 1
        h_out = (torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float), 
                 torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float))
        
        h_div_out = (torch.zeros([1, 1, arg_dict["div_lstm_size"]], dtype=torch.float), 
                 torch.zeros([1, 1, arg_dict["div_lstm_size"]], dtype=torch.float))
        
        skill = torch.zeros([1,1, arg_dict["div_num"]], dtype=torch.float)
        skill[:,:,div_idx] = 1.0

        loop_t, forward_t, wait_t = 0.0, 0.0, 0.0

        obs = env_left.reset()


        while not done:  # step loop
            
            h_in = h_out
            h_div_in = h_div_out
            state_dict = fe1.encode(obs[0], 0)
            state_dict_tensor = state_to_tensor1(state_dict, h_in, skill, h_div_in)
            
            with torch.no_grad():
                a_prob, m_prob, _, h_out, _, _ = model(state_dict_tensor)

            real_action, a, m, need_m, prob, prob_selected_a, prob_selected_m = get_action(a_prob, m_prob)

            prev_obs = obs
            obs, rew, done, info = env_left.step([real_action])

            steps += 1
            score += rew
            
            if done:
                print("Model evaluate with ", arg_dict["env_evaluation"]," model: score",score, "skill", div_idx)


def evaluator(div_model, center_model, signal_queue, summary_queue, arg_dict):
    print("Evaluator process started")
    fe_module1 = importlib.import_module("encoders." + arg_dict["encoder"])
    #fe_module2 = importlib.import_module("encoders." + "encoder_basic")
    rewarder = importlib.import_module("rewarders." + arg_dict["rewarder"])
    #opp_import_model = importlib.import_module("models." + "conv1d_self")
    
    fe1 = fe_module1.FeatureEncoder()
    state_to_tensor1 = fe_module1.state_to_tensor
    #fe2 = fe_module2.FeatureEncoder()
    #state_to_tensor2 = fe_module2.state_to_tensor
    model = center_model
    #opp_model = opp_import_model.Model(arg_dict)
    #opp_model_checkpoint = torch.load(arg_dict["env_evaluation"])
    #opp_model.load_state_dict(opp_model_checkpoint['model_state_dict'])

    #
    #env = football_env.create_environment(env_name=arg_dict["env"], number_of_right_players_agent_controls=1, representation="raw", \
    #                                      stacked=False, logdir='/tmp/football', write_goal_dumps=False, write_full_episode_dumps=False, \
    #                                      render=False)
    
    env_left = football_env.create_environment(env_name=arg_dict["env_evaluation"], representation="raw", stacked=False, logdir=arg_dict["log_dir_dump_left"], \
                                          number_of_left_players_agent_controls=1,
                                          number_of_right_players_agent_controls=0,
                                          write_goal_dumps=True, write_full_episode_dumps=False, render=False, write_video=True)
    #env_right = football_env.create_environment(env_name=arg_dict["env_evaluation"], representation="raw", stacked=False, logdir=arg_dict["log_dir_dump_right"], \
    #                                      number_of_left_players_agent_controls=0,
    #                                      number_of_right_players_agent_controls=1,
    #                                      write_goal_dumps=False, write_full_episode_dumps=True, render=False, write_video=True)
    n_epi = 0
    while True: # episode loop

        done = False
        steps, score, tot_reward, win = 0, 0, 0, 0
        n_epi += 1
        h_out = (torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float), 
                 torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float))
        
        h_div_out = (torch.zeros([1, 1, arg_dict["div_lstm_size"]], dtype=torch.float), 
                 torch.zeros([1, 1, arg_dict["div_lstm_size"]], dtype=torch.float))
        
        skill = torch.zeros([1,1, arg_dict["div_num"]], dtype=torch.float)
        skill_num = random.randint(0, arg_dict["div_num"]-1)
        skill[:,:,skill_num] = 1.0
        skill_acc, skill_steps = 0.0, 0.0

        loop_t, forward_t, wait_t = 0.0, 0.0, 0.0

        obs = env_left.reset()

        ball_owned_team = obs[0]["ball_owned_team"] #-1

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
            h_div_in = h_div_out
            state_dict = fe1.encode(obs[0])
            state_dict_tensor = state_to_tensor1(state_dict, h_in, skill, h_div_in)
            
            t1 = time.time()
            with torch.no_grad():
                a_prob, m_prob, _, h_out, _, _ = model(state_dict_tensor)
            forward_t += time.time()-t1 

            real_action, a, m, need_m, prob, prob_selected_a, prob_selected_m = get_action(a_prob, m_prob)

            prev_obs = obs
            obs, rew, done, info = env_left.step([real_action])
            ball_owned_team = obs[0]["ball_owned_team"]


            state_prime_dict = fe1.encode(obs[0])
            state_prime_dict_tensor = state_to_tensor1(state_prime_dict, h_in, skill, h_div_in)
            #state_dict_tensor["player_div_prime"] = state_prime_dict_tensor["player_div"]
            #state_dict_tensor["opp_div_prime"] = state_prime_dict_tensor["opp_div"]
            #state_dict_tensor["ball_prime"] = state_prime_dict_tensor["ball"]

            with torch.no_grad():
                prob_z, h_div_out = div_model(state_prime_dict_tensor)
                z_most_idx = find_most_z_idx(prob_z)
                
            if ball_owned_team == 0:
                fin_r = rewarder.calc_reward(rew, prev_obs[0], obs[0], float(prob_z[skill_num])*(arg_dict["div_num"]))
            else:
                fin_r = rewarder.calc_reward(rew, prev_obs[0], obs[0], None)
            
            (h1_in, h2_in) = h_in
            (h1_out, h2_out) = h_out
            (h1_div_in, h2_div_in) = h_in
            state_dict["hidden"] = (h1_in.numpy(), h2_in.numpy())
            state_prime_dict["hidden_div"] = (h1_div_in.numpy(), h2_div_in.numpy())
            state_prime_dict["hidden"] = (h1_out.numpy(), h2_out.numpy())
            transition = (state_dict, a, m, fin_r, state_prime_dict, prob, done, need_m)

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
                print("Model evaluate with ", arg_dict["env_evaluation"]," model: score",score, "reward", tot_reward, "skill", skill_num)
                summary_data = (win, score, tot_reward, steps, arg_dict['env_evaluation'], loop_t/steps, forward_t/steps, wait_t/steps, skill_acc/skill_steps)
                summary_queue.put(summary_data)


def seperate_evaluator(div_model, center_model, signal_queue, summary_queue, arg_dict):
    print("Evaluator process started")
    fe_module1 = importlib.import_module("encoders." + arg_dict["encoder"])
    #fe_module2 = importlib.import_module("encoders." + "encoder_basic")
    rewarder = importlib.import_module("rewarders." + arg_dict["rewarder"])
    #opp_import_model = importlib.import_module("models." + "conv1d_self")
    
    fe1 = fe_module1.FeatureEncoder()
    state_to_tensor1 = fe_module1.state_to_tensor
    #fe2 = fe_module2.FeatureEncoder()
    #state_to_tensor2 = fe_module2.state_to_tensor
    model_att = center_model[0]
    model_def = center_model[1]
    #opp_model = opp_import_model.Model(arg_dict)
    #opp_model_checkpoint = torch.load(arg_dict["env_evaluation"])
    #opp_model.load_state_dict(opp_model_checkpoint['model_state_dict'])

    #
    #env = football_env.create_environment(env_name=arg_dict["env"], number_of_right_players_agent_controls=1, representation="raw", \
    #                                      stacked=False, logdir='/tmp/football', write_goal_dumps=False, write_full_episode_dumps=False, \
    #                                      render=False)
    
    env_left = football_env.create_environment(env_name=arg_dict["env_evaluation"], representation="raw", stacked=False, logdir=arg_dict["log_dir_dump_left"], \
                                          number_of_left_players_agent_controls=1,
                                          number_of_right_players_agent_controls=0,
                                          write_goal_dumps=True, write_full_episode_dumps=False, render=False, write_video=True)
    #env_right = football_env.create_environment(env_name=arg_dict["env_evaluation"], representation="raw", stacked=False, logdir=arg_dict["log_dir_dump_right"], \
    #                                      number_of_left_players_agent_controls=0,
    #                                      number_of_right_players_agent_controls=1,
    #                                      write_goal_dumps=False, write_full_episode_dumps=True, render=False, write_video=True)
    n_epi = 0
    while True: # episode loop

        done = False
        steps, score, tot_reward, win = 0, 0, 0, 0
        n_epi += 1
        h_out = (torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float), 
                 torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float))
        
        h_div_out = (torch.zeros([1, 1, arg_dict["div_lstm_size"]], dtype=torch.float), 
                 torch.zeros([1, 1, arg_dict["div_lstm_size"]], dtype=torch.float))
        
        skill = torch.zeros([1,1, arg_dict["div_num"]], dtype=torch.float)
        skill_num = random.randint(0, arg_dict["div_num"]-1)
        skill[:,:,skill_num] = 1.0
        skill_acc, skill_steps = 0.0, 0.0

        loop_t, forward_t, wait_t = 0.0, 0.0, 0.0

        obs = env_left.reset()

        ball_owned_team = obs[0]["ball_owned_team"] #-1

        while not done:  # step loop
        
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
                    model_att.load_state_dict(center_model.state_dict())
                wait_t += time.time() - init_t
                
                h_in = h_out
                h_div_in = h_div_out
                state_dict = fe1.encode(obs[0])
                state_dict_tensor = state_to_tensor1(state_dict, h_in, skill, h_div_in)
                
                t1 = time.time()
                with torch.no_grad():
                    a_prob, m_prob, _, h_out, _, _ = model_att(state_dict_tensor)
                    prob_z, h_div_out = div_model(state_dict_tensor)
                    z_most_idx = find_most_z_idx(prob_z)
                forward_t += time.time()-t1 
    
                real_action, a, m, need_m, prob, prob_selected_a, prob_selected_m = get_action(a_prob, m_prob)
    
                prev_obs = obs

                obs, rew, done, info = env_left.step([real_action])

                ball_owned_team = obs[0]["ball_owned_team"]
                fin_r = rewarder.calc_reward(rew, prev_obs[0], obs[0], 0)
                state_prime_dict = fe1.encode(obs[0])
                

                (h1_in, h2_in) = h_in
                (h1_out, h2_out) = h_out
                (h1_div_in, h2_div_in) = h_in
                state_dict["hidden"] = (h1_in.numpy(), h2_in.numpy())
                state_dict["hidden_div"] = (h1_div_in.numpy(), h2_div_in.numpy())
                state_prime_dict["hidden"] = (h1_out.numpy(), h2_out.numpy())
                transition = (state_dict, a, m, fin_r, state_prime_dict, prob, done, need_m)
    
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

                    print("Model evaluate with ", arg_dict["env_evaluation"]," model: score",score,"total reward",tot_reward)
                    summary_data = (win, score, tot_reward, steps, arg_dict['env_evaluation'], loop_t/steps, forward_t/steps, wait_t/steps, skill_acc/skill_steps)
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
                    pass
                wait_t += time.time() - init_t
                
                h_in = h_out
                state_dict = fe1.encode(obs[0])
                state_dict_tensor = state_to_tensor1(state_dict, h_in, skill, 0)
                
                t1 = time.time()
                with torch.no_grad():
                    a_prob, m_prob, _, h_out = model_def(state_dict_tensor)
                forward_t += time.time()-t1 
    
                real_action, a, m, need_m, prob, prob_selected_a, prob_selected_m = get_action(a_prob, m_prob)
    
                prev_obs = obs

                obs, rew, done, info = env_left.step([real_action])

                ball_owned_team = obs[0]["ball_owned_team"]
                fin_r = rewarder.calc_reward(rew, prev_obs[0], obs[0], 0)
                state_prime_dict = fe1.encode(obs[0])
                
                (h1_in, h2_in) = h_in
                (h1_out, h2_out) = h_out
                state_dict["hidden"] = (h1_in.numpy(), h2_in.numpy())
                state_prime_dict["hidden"] = (h1_out.numpy(), h2_out.numpy())
    
                steps += 1
                score += rew
                tot_reward += fin_r
                
                loop_t += time.time()-init_t
                
                if done:
                    if score > 0:
                        win = 1
                    print("Model evaluate with ", arg_dict["env_evaluation"]," model: score",score,"total reward",tot_reward)
                    summary_data = (win, score, tot_reward, steps, arg_dict['env_evaluation'], loop_t/steps, forward_t/steps, wait_t/steps, skill_acc/skill_steps)
                    summary_queue.put(summary_data)
           
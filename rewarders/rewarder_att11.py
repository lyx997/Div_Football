import numpy as np
import math

def calc_reward(rew, prev_obs, obs, skill_reward, opp_num, left_owned_ball):
    ball_x, ball_y, ball_z = obs['ball']
    MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
    PENALTY_Y, END_Y = 0.27, 0.42

    ball_position_r = 0.0
    if   (-END_X <= ball_x and ball_x < -PENALTY_X)and (-PENALTY_Y < ball_y and ball_y < PENALTY_Y):
        ball_position_r = -2.0
    elif (-PENALTY_X <= ball_x and ball_x < -MIDDLE_X) and (-END_Y < ball_y and ball_y < END_Y):
        ball_position_r = -1.0
    elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (-END_Y < ball_y and ball_y < END_Y):
        ball_position_r = 0.0
    elif (PENALTY_X < ball_x  and ball_x <= END_X) and (-PENALTY_Y < ball_y and ball_y < PENALTY_Y):
        ball_position_r = 2.0
    elif (MIDDLE_X < ball_x   and ball_x <= END_X) and (-END_Y < ball_y and ball_y < END_Y):
        ball_position_r = 1.0
    else:
        ball_position_r = 0.0
    
    win_reward = 0.0
    if obs['steps_left'] == 0:
        [my_score, opponent_score] = obs['score']
        if my_score > opponent_score:
            win_reward = 1.0
        elif my_score < opponent_score:
            win_reward = -1.0

    yellow_r = 0.0
    change_ball_owned_reward = 0.0
    safe_pass_reward = 0.0
    attention_reward = 0.0
    good_pass_counts = 0
    att_pass_counts = 0
    active = obs['active']
    active_tired = obs['left_team_tired_factor'][active]
    tired_reward = 0
    not_owned_ball_dis_reward = 0
    owned_ball_dis_reward = 0
    skill_div_reward = 0.0
    reward = 0.0

    if prev_obs:

        yellow_r = np.sum(prev_obs["left_team_yellow_card"]) - np.sum(obs["left_team_yellow_card"]) 
        #right_yellow = np.sum(obs["right_team_yellow_card"]) -  np.sum(prev_obs["right_team_yellow_card"])
        #yellow_r = right_yellow - left_yellow
        prev_active = prev_obs['active']
        prev_active_tired = prev_obs['left_team_tired_factor'][prev_active]

        #if active == prev_active and (PENALTY_X > ball_x or -PENALTY_Y > ball_y or ball_y > PENALTY_Y):
        #    tired_reward = prev_active_tired - active_tired

        prev_ball_x = prev_obs["ball"][0]
        prev_owned_ball_team = prev_obs["ball_owned_team"]
        prev_owned_ball_player = prev_obs["ball_owned_player"]
        prev_obs_right_team_x = np.array(prev_obs['right_team'][1:][0])
        prev_max_right_team_x = float(np.min(prev_obs_right_team_x))
        owned_ball_team = obs["ball_owned_team"]
        owned_ball_player = obs["ball_owned_player"]
        active_pos_x = obs['left_team'][active][0]


        if owned_ball_team == 0 and prev_owned_ball_team == 1 and owned_ball_player != 0:
            change_ball_owned_reward = 3.0
        elif owned_ball_team == 1 and prev_owned_ball_team == 0 and owned_ball_player != 0:
            change_ball_owned_reward = -3.0
        #elif owned_ball_team == 1 and prev_owned_ball_team == 1 and prev_owned_ball_player != owned_ball_player and owned_ball_player != 0:
        #    change_ball_owned_reward = -1.0
        #elif owned_ball_team == 0 and prev_owned_ball_team == 0 and prev_owned_ball_player != owned_ball_player and owned_ball_player != 0 :
        #    good_pass_counts = 1
        #    change_ball_owned_reward = 1.0

        active_dis_to_ball = np.linalg.norm(np.array(obs['left_team'][active] - obs['right_team'][opp_num]), axis=0, keepdims=True)
        if not left_owned_ball:
            if active_dis_to_ball <= 0.03:
                not_owned_ball_dis_reward = 0
            elif active_dis_to_ball > 0.03 and active_dis_to_ball <= 0.06:
                not_owned_ball_dis_reward = -1
            elif active_dis_to_ball > 0.06 and active_dis_to_ball <= 0.1:
                not_owned_ball_dis_reward = -2
            elif active_dis_to_ball > 0.1:
                not_owned_ball_dis_reward = -3
        else:
            if active_dis_to_ball <= 0.03:
                owned_ball_dis_reward = 0
            elif active_dis_to_ball > 0.03 and active_dis_to_ball <= 0.06:
                owned_ball_dis_reward = 1
            elif active_dis_to_ball > 0.06 and active_dis_to_ball <= 0.1:
                owned_ball_dis_reward = 2
            elif active_dis_to_ball > 0.1:
                owned_ball_dis_reward = 3

        if skill_reward:
            skill_div_reward = np.log(skill_reward)

        if skill_div_reward > 0.1 or not left_owned_ball:
            reward = 5.0*win_reward + 5.0*rew + 0.01*yellow_r + 0.003*ball_position_r + 0.01*not_owned_ball_dis_reward + 0.01*owned_ball_dis_reward + 0.1*change_ball_owned_reward 
        elif skill_div_reward > 0 and skill_div_reward <=0.1:
            reward = 0.01*skill_div_reward + 0.01*ball_position_r + 0.1*change_ball_owned_reward
        else:
            reward = 0.05*skill_div_reward
    return reward
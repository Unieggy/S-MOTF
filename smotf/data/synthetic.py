"""
Synthetic Data Generator for smotf

this module provides a mathmatically deterministic fake physics engine and expert policy to generate synthetic trajectories
"""
import torch

#taking your config obj, the num of episodes,and episode length to generate data
def generate_synthetic_data(cfg,n_trajectories=100,trajectory_length=200):
    torch.manual_seed(42)

    #extracts dimensions from the config
    d_base=cfg.dims.base # 12 (RPY, ang-vel, lin-vel, gravity)
    d_legs=cfg.dims.legs # 24 (12 joint angles + 12 velocities)
    d_contacts=cfg.dims.contacts # 4 (foot contact booleans)
    d_command=cfg.dims.command # 3 (target velocities: vx, vy, wz)
    d_action=cfg.dims.action # 12 (target joint angles)

    #the fake perfect controller
    #matrix that maps base+command to action
    #action=W_action*state (12,15) matrix
    W_action=torch.randn(d_action,d_base+d_command)/(d_base+d_command)**0.5

    #nextstate=W_dyn**(state+action)
    #matrix that maps base+action to next base
    W_dyn=torch.randn(d_base,d_base+d_action)/(d_base+d_action)**0.5

    dataset=[]

    #simulation loop
    for _ in range(n_trajectories):
        #create empty lists to hold the history of this epi
        ep_base,ep_legs,ep_contacts,ep_command,ep_action,ep_next=[],[],[],[],[],[]
        ep_reward=[]   # Phase 2: a deterministic (hence learnable) fake reward

        #random starting posture
        current_base=torch.randn(d_base)

        #inner loop:step forward in time from t=0 to t=trajectory_length
        for t in range(trajectory_length):

            current_legs=torch.randn(d_legs)
            #contacts are boolean
            current_contacts=torch.randint(0,2,(d_contacts,)).float()
            current_command=torch.rand(d_command)

            #calculate the action
            #glue base and command together to a 15 dim vector
            policy_input=torch.cat([current_base,current_command])

            #multiply w_action by the vector to get the action
            clean_action=W_action@policy_input

            #add gaussian noise to the action since flow matching network needs messy data so it can learn
            noisy_action=clean_action+0.05*torch.randn_like(clean_action)

            #deterministic (hence learnable) fake reward: stay near origin, act small
            reward=-0.1*(current_base**2).sum()-0.01*(noisy_action**2).sum()

            #save all data into lists
            ep_base.append(current_base)
            ep_legs.append(current_legs)
            ep_command.append(current_command)
            ep_contacts.append(current_contacts)
            ep_action.append(noisy_action)
            ep_reward.append(reward.reshape(1))

            #compute the next state
            physics_input=torch.cat([current_base,noisy_action])
            next_base=W_dyn@physics_input
            ep_next.append(next_base)
            current_base=next_base

        episode_dict={
            "base":torch.stack(ep_base),
            "legs":torch.stack(ep_legs),
            "contacts":torch.stack(ep_contacts),
            "action":torch.stack(ep_action),
            "command":torch.stack(ep_command),
            "s_next":torch.stack(ep_next),
            "reward":torch.stack(ep_reward),      # [T, 1]
        }
        dataset.append(episode_dict)

    return dataset

    
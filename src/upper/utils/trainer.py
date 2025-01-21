import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os 
import copy
import time 
import random
import numpy as np 

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
class A2CTrainer(object):
    
    def __init__(self, actor, critic, args, env):
        self.actor = actor
        self.critic = critic 
        self.args = args 
        self.env = env 
        self.test_seed = 100
        print("Trainer has been initialized")
        
    def train(self):
        args = self.args 
        env = self.env 
        actor = self.actor
        critic = self.critic 
        actor.train()
        critic.train()
        max_epochs = args['n_train']

        actor_optim = optim.Adam(actor.parameters(), lr=args['actor_net_lr'])
        critic_optim = optim.Adam(critic.parameters(), lr=args['critic_net_lr'])
        
        best_model = 2_500
        val_model = 2_500
        r_test = []
        r_val = []
        s_t = time.time()
        print("training started")
        for i in range(max_epochs):
            
            state, _ = env.reset(batch_size=args['batch_size'])
            obs = state['obs']
            mask = state['mask']
            last = torch.from_numpy(state['last']).to(device)
            
            static_data = torch.from_numpy(obs.astype(np.float32)).to(device)
            # [b_s, hidden_dim, n_nodes]
            static_hidden = actor.emd_stat(static_data)
            # critic inputs 
            static_critic = static_data.permute(0, 2, 1).to(device)
            mask_critic = torch.from_numpy(mask.astype(np.float32)).to(device)
            # Query the critic for an estimate of the reward
            critic_est = critic(static_critic, mask_critic.unsqueeze(2)).view(-1)
            
            # lstm initial states 
            hx = torch.zeros(1, env.batch_size, args['hidden_dim']).to(device)
            cx = torch.zeros(1, env.batch_size, args['hidden_dim']).to(device)

            # prepare input 
            ter = np.zeros(env.batch_size).astype(np.float32)
            decoder_input = static_hidden[:, :, 0].unsqueeze(2) # 0: warehouse
            
            # storage containers 
            logs = []
            actions = []
            probs = []
            time_step = 0
            
            while time_step < args['decode_len']:
                terminated = torch.from_numpy(ter.astype(np.float32)).to(device)
                mask = torch.from_numpy(mask.astype(np.float32)).to(device)
                dynamic = torch.from_numpy(obs.astype(np.float32)).to(device)
                dynamic_b = torch.from_numpy(env.get_battery_dynamic().astype(np.float32)).to(device)
                decoder_input =  torch.gather(static_hidden, 2, last.view(-1, 1, 1).expand(env.batch_size, args['hidden_dim'], 1)).detach()
                action, prob, logp, hx, cx = actor.forward(dynamic, static_hidden, decoder_input, hx, cx, terminated, battery_dynamic=dynamic_b, mask=mask)
                logs.append(logp.unsqueeze(1))
                actions.append(action.unsqueeze(1))
                probs.append(prob.unsqueeze(1))
                
                state, reward, ter, _, _ = env.step(action.cpu().numpy())
                if np.all(ter):
                    break
                obs = state['obs']
                mask = state['mask']
                last = torch.from_numpy(state['last']).to(device)
                
                time_step += 1

            print("epochs: ", i)
            actions = torch.cat(actions, dim=1)  # (batch_size, seq_len)
            logs = torch.cat(logs, dim=1)  # (batch_size, seq_len)

            R = env.time_steps().astype(np.float32) 
            R = torch.from_numpy(R).to(device)
            advantage = (R - critic_est)
            actor_loss = torch.mean(advantage.detach() * logs.sum(dim=1))
            critic_loss = torch.mean(advantage ** 2)

            actor_optim.zero_grad()
            actor_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(actor.parameters(), args['max_grad_norm'])
            actor_optim.step()
            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), args['max_grad_norm'])
            critic_optim.step()
        
            e_t = time.time() - s_t
            print("e_t: ", e_t)
            if i % args['test_interval'] == 0:
        
                R = self.test()
                r_test.append(R)
                np.savetxt("trained_models/test_rewards.txt", r_test)
            
                print("testing average rewards: ", R)
                if R < best_model:
                 #   R_val = self.test(inference=False, val=False)
                    best_model = R
                    num = str(i // args['save_interval'])
                    torch.save(actor.state_dict(), 'trained_models/' + '/' + 'best_model' + '_actor_truck_params.pkl')
                    torch.save(critic.state_dict(), 'trained_models/' + '/' + 'best_model' + '_critic_params.pkl')
                    print(f"new best testing reward!: {R}")
                   
            if i % args['save_interval'] ==0:
                num = str(i // args['save_interval'])
                torch.save(actor.state_dict(), 'trained_models/' + '/' + num + '_actor_truck_params.pkl')
                torch.save(critic.state_dict(), 'trained_models/' + '/' + num + '_critic_params.pkl')


    def test(self):
        args = self.args 
        env = self.env 
        state, _ = env.reset(seed=self.test_seed, batch_size=args['test_size'])
        obs = state['obs']
        mask = state['mask']
        last = torch.from_numpy(state['last']).to(device)
        actor = self.actor
        actor.eval()

        with torch.no_grad():
            static_data = torch.from_numpy(obs.astype(np.float32)).to(device)
            # [b_s, hidden_dim, n_nodes]
            static_hidden = actor.emd_stat(static_data)
            
            # lstm initial states 
            hx = torch.zeros(1, env.batch_size, args['hidden_dim']).to(device)
            cx = torch.zeros(1, env.batch_size, args['hidden_dim']).to(device)
        
            # prepare input 
            ter = np.zeros(env.batch_size).astype(np.float32)
            decoder_input = static_hidden[:, :, 0].unsqueeze(2)
            time_step = 0
            while time_step < args['decode_len']:
                terminated = torch.from_numpy(ter.astype(np.float32)).to(device)
                mask = torch.from_numpy(mask.astype(np.float32)).to(device)
                dynamic = torch.from_numpy(obs.astype(np.float32)).to(device)
                dynamic_b = torch.from_numpy(env.get_battery_dynamic().astype(np.float32)).to(device)
                decoder_input =  torch.gather(static_hidden, 2, last.view(-1, 1, 1).expand(env.batch_size, args['hidden_dim'], 1)).detach()
                action, prob, logp, hx, cx = actor.forward(dynamic, static_hidden, decoder_input, hx, cx, terminated, battery_dynamic=dynamic_b, mask=mask)
                
                state, reward, ter, _, _ = env.step(action.cpu().numpy())
                if np.all(ter):
                    break
                obs = state['obs']
                mask = state['mask']
                last = torch.from_numpy(state['last']).to(device)
                
                time_step += 1
                
        R = copy.copy(env.time_steps())
        print("finished: ", sum(ter))
        
        fname = 'test_results-{}-len-{}.txt'.format(args['test_size'], args['n_nodes'])
        fname = 'results/' + fname
        np.savetxt(fname, R)
        
        actor.train()
        return R.mean()
    
    
    def sampling_batch(self, sample_size):
        val_size = self.dataGen.get_test_all().shape[0]
        best_rewards = np.ones(sample_size)*100000
        best_sols = np.zeros([sample_size, self.args['decode_len'], 2])
        args = self.args 
        env = self.env 
        dataGen = self.dataGen
        actor  = self.actor
     
     
        actor.eval()
        actor.set_sample_mode(True)
        times = []
        initial_t = time.time()
        data = dataGen.get_test_all()
        # print("data: ", data)
        data_list = [np.expand_dims(data[i, ...], axis=0) for i in range(data.shape[0])]
        best_rewards_list = []
        for d in data_list:
            data = np.repeat(d, sample_size, axis=0)
            env.input_data = data 
            
            state, avail_actions = env.reset()
            
            
            time_vec_truck = np.zeros([sample_size, 2])
            time_vec_drone = np.zeros([sample_size, 3])
            with torch.no_grad():
                data = torch.from_numpy(data[:, :, :2].astype(np.float32)).to(device)
                # [b_s, hidden_dim, n_nodes]
                static_hidden = actor.emd_stat(data).permute(0, 2, 1)
            
                # lstm initial states 
                hx = torch.zeros(1, sample_size, args['hidden_dim']).to(device)
                cx = torch.zeros(1, sample_size, args['hidden_dim']).to(device)
                last_hh = (hx, cx)
        
                # prepare input 
                ter = np.zeros(sample_size).astype(np.float32)
                decoder_input = static_hidden[:, :, env.n_nodes-1].unsqueeze(2)
                time_step = 0
                while time_step < args['decode_len']:
                    terminated = torch.from_numpy(ter).to(device)
                    for j in range(2):
                        # truck takes action 
                        if j == 0:
                            avail_actions_truck = torch.from_numpy(avail_actions[:, :, 0].reshape([sample_size, env.n_nodes]).astype(np.float32)).to(device)
                            dynamic_truck = torch.from_numpy(np.expand_dims(state[:, :, 0], 2)).to(device)
                            # (batch_size, hidden_dim, n_nodes), 
                            # (batch_size, n_nodes, 1), 
                            # (batch_size, hidden_dim, 1), 
                            # ((1, batch_size, hidden_dim), (1, batch_size, hidden_dim)), 
                            # (1), 
                            # (batch_size, n_nodes)
                            idx_truck, prob, logp, last_hh = actor.forward(static_hidden, dynamic_truck, decoder_input, last_hh, 
                                                                        terminated, avail_actions_truck)
                            b_s = np.where(np.logical_and(avail_actions[:, :, 1].sum(axis=1)>1, env.sortie==0))[0]
                            avail_actions[b_s, idx_truck[b_s].cpu(), 1] = 0
                            avail_actions_drone = torch.from_numpy(avail_actions[:, :, 1].reshape([sample_size, env.n_nodes]).astype(np.float32)).to(device)
                            idx = idx_truck 
                        
                        else:
                            dynamic_drone = torch.from_numpy(np.expand_dims(state[:, :, 1], 2)).to(device)
                            idx_drone, prob, logp, last_hh = actor.forward(static_hidden, dynamic_drone, decoder_input, last_hh, 
                                                                        terminated, avail_actions_drone)
                            idx = idx_drone 

                        decoder_input =  torch.gather(static_hidden, 2, idx.view(-1, 1, 1).expand(sample_size, args['hidden_dim'], 1)).detach()
                
                    state, avail_actions, ter, time_vec_truck, time_vec_drone = env.step(idx_truck.cpu().numpy(), idx_drone.cpu().numpy(), time_vec_truck, time_vec_drone, ter)
                    # print("state: ", state)
                    time_step += 1
                    
                    
            
            R = copy.copy(env.current_time)
            print('R.shape:', R.shape)
            best_rewards = R.min(axis=0)
            print('best_rewards:', best_rewards)
            t = time.time() - initial_t
            times.append(t)
            best_rewards_list.append(best_rewards)        
        
        np.savetxt(f'results/best_rewards_list_{sample_size}_samples.txt', best_rewards_list)
        return best_rewards, times 
    
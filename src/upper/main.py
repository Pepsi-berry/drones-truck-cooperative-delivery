import numpy as np 
import os 
import torch 
import random
from utils.options import ParseParams
from utils.env_decoupled import env_creator
from model.hybrid_model import HMActor, Critic
from utils.trainer import A2CTrainer

if __name__ == '__main__':
    args = ParseParams()
    
    random_seed = args['random_seed']
    if random_seed is not None and random_seed > 0:
        print("# Set random seed to %d" % random_seed)
    else:
        random_seed = random.randint(0, 100_000)
        print("# Set random seed to %d" % random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    max_epochs = args['n_train']
    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
    save_path = args['save_path']
    n_nodes = args['n_nodes']
    env = env_creator()
    actor = HMActor(args['hidden_dim'])
    critic = Critic(args['hidden_dim'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        path = save_path + 'n' + str(n_nodes) + '/best_model_actor_truck_params_9.pkl'
        if os.path.exists(path):
            actor.load_state_dict(torch.load(path, map_location='cpu'))
            path = save_path + 'n' + str(n_nodes) + '/best_model_critic_params_9.pkl'
            critic.load_state_dict(torch.load(path, map_location='cpu'))
            print("Succesfully loaded keys")
    
    # agent = A2CAgent(actor, critic, args, env, dataGen)
    trainer = A2CTrainer(actor, critic, args, env)

    if args['train']:
        trainer.train()
    else:
        if args['sampling']:
            best_R = trainer.sampling_batch(args['n_samples'])
        else:
            R = trainer.test()
            print(R)

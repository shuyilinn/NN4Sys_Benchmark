'''
This script is to train the model for Pensieve.
'''
import os
import numpy as np
import torch.multiprocessing as mp
from torch.distributions import Categorical
import torch
import torch.optim as optim
import test
import env
import model
import load_trace
import warnings
import argparse

warnings.filterwarnings('ignore')
cuda = torch.cuda.is_available()

# Hyperparameters and settings
S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
NUM_AGENTS = 5
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
HD_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0  # MILLISECONDS_IN_SECOND
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
SUMMARY_DIR = './results'
TRAIN_TRACES = './cooked_traces/'
NN_MODEL = None
max_epoch = 50000
REWARD = 'linear'  # We only need linear reward trained model because it performs best

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train the Pensieve model.')
parser.add_argument('--model', choices=['small', 'mid', 'big', 'all'], default='big',
                    help='Choose the model size: small, mid, big, or all to train all models sequentially.')
args = parser.parse_args()


def create_actor_model(model_name):
    if model_name == 'mid':
        return model.ActorNetwork_mid(state_dim=[S_INFO, S_LEN], action_dim=A_DIM, learning_rate=ACTOR_LR_RATE)
    elif model_name == 'big':
        return model.ActorNetwork_big(state_dim=[S_INFO, S_LEN], action_dim=A_DIM, learning_rate=ACTOR_LR_RATE)
    else:
        return model.ActorNetwork_small(state_dim=[S_INFO, S_LEN], action_dim=A_DIM, learning_rate=ACTOR_LR_RATE)


def train_model(model_name, exp_queues, net_params_queues):
    print(f"Training {model_name} model...")
    actor = create_actor_model(model_name)
    if cuda:
        actor.cuda()

    # Load pre-trained model if provided
    if NN_MODEL is not None:
        if cuda:
            para = torch.load(NN_MODEL, map_location='cuda')
        else:
            para = torch.load(NN_MODEL, map_location='cpu')
        actor.load_state_dict(para)
        print("Model restored.")

    actor_net_para = actor.state_dict()
    if cuda:
        for k, v in actor_net_para.items():
            actor_net_para[k] = v.cpu()

    for i in range(NUM_AGENTS):
        net_params_queues[i].put(actor_net_para)
    del actor_net_para

    # Create critic model
    critic = model.CriticNetwork(state_dim=[S_INFO, S_LEN], learning_rate=CRITIC_LR_RATE)
    if cuda:
        critic.cuda()

    epoch = 0
    optimizer_actor = optim.RMSprop(actor.parameters(), lr=ACTOR_LR_RATE)
    optimizer_critic = optim.RMSprop(critic.parameters(), lr=CRITIC_LR_RATE)
    actor.train()
    critic.train()
    optimizer_actor.zero_grad()
    optimizer_critic.zero_grad()

    best_perf = -999999

    while True:
        total_reward = 0.0
        total_td_loss = 0.0
        total_entropy = 0.0
        total_agents = 0.0
        total_batch_len = 0.0

        for i in range(NUM_AGENTS):
            s_batch, a_batch, r_batch, terminal, info = exp_queues[i].get()

            actor_loss, critic_loss, td_batch = model.compute_loss(
                s_batch=np.stack(s_batch, axis=0),
                a_batch=np.vstack(a_batch),
                r_batch=np.vstack(r_batch),
                terminal=terminal, actor=actor, critic=critic)

            del s_batch, a_batch, terminal

            total_reward += np.sum(r_batch)
            total_td_loss += np.sum(td_batch.detach().numpy())
            total_batch_len += len(r_batch)
            total_agents += 1.0
            total_entropy += np.sum(info['entropy'])

            actor_loss.backward()
            critic_loss.backward()

            optimizer_critic.step()
            optimizer_actor.step()
            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()

        # Update model parameters
        actor_net_para = actor.state_dict()
        if cuda:
            for k, v in actor_net_para.items():
                actor_net_para[k] = v.cpu()
        for i in range(NUM_AGENTS):
            net_params_queues[i].put(actor_net_para)
        del actor_net_para

        # Logging and saving model
        avg_reward = total_reward / total_agents
        avg_td_loss = total_td_loss / total_batch_len
        avg_entropy = total_entropy / total_batch_len

        print(f'Epoch: {epoch} TD_loss: {avg_td_loss} Reward: {avg_reward} Entropy: {avg_entropy}')

        if epoch % MODEL_SAVE_INTERVAL == 0:
            save_path = f"{SUMMARY_DIR}/nn_model_ep_{model_name}_{REWARD}_{epoch}.pth"
            torch.save(actor.state_dict(), save_path)
            print(f"Model saved in file: {save_path}")
            cur_perf = test.test(save_path, model_name)
            if cur_perf > best_perf:
                best_perf = cur_perf
                save_path = f"{SUMMARY_DIR}/nn_model_ep_BEST_{model_name}_{REWARD}_{epoch}.pth"
                torch.save(actor.state_dict(), save_path)
                print(f"Best model saved in file: {save_path}")

        if epoch > max_epoch:
            return
        epoch += 1


def agent(agent_id, all_cooked_time, all_cooked_bw, exp_queue, actor_params_queue):
    net_env = env.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw, random_seed=agent_id)
    actor = create_actor_model(args.model)

    actor_net_para = actor_params_queue.get()
    if cuda:
        for k, v in actor_net_para.items():
            actor_net_para[k] = v.cuda()
    if cuda:
        actor.cuda()
    actor.load_state_dict(actor_net_para)
    del actor_net_para

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    s_batch = [np.zeros((S_INFO, S_LEN))]
    a_batch = [np.zeros(A_DIM)]
    r_batch = []
    entropy_record = []

    while True:
        delay, sleep_time, buffer_size, rebuf, video_chunk_size, next_video_chunk_sizes, end_of_video, video_chunk_remain = net_env.get_video_chunk(bit_rate)

        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K - REBUF_PENALTY * rebuf - SMOOTH_PENALTY * np.abs(
            VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
        r_batch.append(reward)

        last_bit_rate = bit_rate

        state = np.array(s_batch[-1], copy=True)
        state = np.roll(state, -1, axis=1)
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
        state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR
        state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K
        state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

        with torch.no_grad():
            actor_input = torch.from_numpy(np.reshape(state, (S_INFO, S_LEN)))
            if cuda:
                actor_input = actor_input.cuda()
            action_prob = actor.forward(actor_input)
            if cuda:
                action_prob = action_prob.cpu()

        m = Categorical(action_prob)
        action = m.sample().item()
        bit_rate = action
        entropy_record.append(model.compute_entropy(action_prob[0]))

        if len(r_batch) >= TRAIN_SEQ_LEN or end_of_video:
            exp_queue.put([s_batch[1:], a_batch[1:], r_batch[1:], end_of_video, {'entropy': entropy_record}])
            actor_net_para = actor_params_queue.get()
            if cuda:
                for k, v in actor_net_para.items():
                    actor_net_para[k] = v.cuda()
            actor.load_state_dict(actor_net_para)
            del actor_net_para, s_batch[:], a_batch[:], r_batch[:], entropy_record[:]

        if end_of_video:
            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY
            s_batch.append(np.zeros((S_INFO, S_LEN)))
            a_batch.append(bit_rate)
        else:
            s_batch.append(state)
            a_batch.append(bit_rate)


def main():
    np.random.seed(RANDOM_SEED)
    assert len(VIDEO_BIT_RATE) == A_DIM

    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)
    all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(TRAIN_TRACES)

    mp.set_start_method('spawn')
    processes = []
    exp_queues = []
    net_params_queues = []

    for i in range(NUM_AGENTS):
        exp_queues.append(mp.Queue())
        net_params_queues.append(mp.Queue())

    # If 'all' is selected, train all three models sequentially
    if args.model == 'all':
        for model_name in ['small', 'mid', 'big']:
            central_p = mp.Process(target=train_model, args=(model_name, exp_queues, net_params_queues))
            central_p.start()
            for i in range(NUM_AGENTS):
                p = mp.Process(target=agent, args=(i, all_cooked_time, all_cooked_bw, exp_queues[i], net_params_queues[i]))
                processes.append(p)
            for p in processes:
                p.start()
            central_p.join()
    else:
        central_p = mp.Process(target=train_model, args=(args.model, exp_queues, net_params_queues))
        central_p.start()
        for i in range(NUM_AGENTS):
            p = mp.Process(target=agent, args=(i, all_cooked_time, all_cooked_bw, exp_queues[i], net_params_queues[i]))
            processes.append(p)
        for p in processes:
            p.start()
        central_p.join()
    
    print(f"[Done] Finished training {args.model} model")


if __name__ == '__main__':
    main()

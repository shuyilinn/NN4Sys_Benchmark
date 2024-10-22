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
print("cuda", cuda)

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
M_IN_K = 1000.0     # MILLISECONDS_IN_SECOND
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = './results'
LOG_FILE = './results/log'
TEST_LOG_FOLDER = './test_results/'
TRAIN_TRACES = './cooked_traces/'
NN_MODEL = None

max_epoch = 50000
REWARD_LIST = ['linear', 'log', 'hd']
REWARD = REWARD_LIST[0]  # We only need linear reward trained model


MODEL_LIST = ['small', 'mid', 'big', 'all']


def central_agent(exp_queues, net_params_queues, MODEL):
    # create actor model
    if MODEL == 'mid':
        actor = model.ActorNetwork_mid(state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                       learning_rate=ACTOR_LR_RATE)
    if MODEL == 'big':
        actor = model.ActorNetwork_big(state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                       learning_rate=ACTOR_LR_RATE)
    if MODEL == 'small':
        actor = model.ActorNetwork_small(state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                         learning_rate=ACTOR_LR_RATE)

    nn_model = NN_MODEL
    if cuda:
        actor.cuda()

    # add actor paras into the queue
    if nn_model is not None:  # nn_model is the path to file
        if cuda:
            para = torch.load(nn_model, map_location='cuda')
        else:
            para = torch.load(nn_model, map_location='cpu')
        actor.load_state_dict(para)
        print("Model restored.")

    actor_net_para = actor.state_dict()
    if cuda:
        for k, v in actor_net_para.items():
            actor_net_para[k] = v.cpu()

    for i in range(NUM_AGENTS):
        net_params_queues[i].put(actor_net_para)
    del actor_net_para

    # create critic model
    critic = model.CriticNetwork(state_dim=[S_INFO, S_LEN],
                                 learning_rate=CRITIC_LR_RATE)
    if cuda:
        critic.cuda()

    epoch = 2000
    epoch_set = []
    qoe_set = []
    optimizer_actor = optim.RMSprop(actor.parameters(), lr=ACTOR_LR_RATE)
    optimizer_critic = optim.RMSprop(critic.parameters(), lr=CRITIC_LR_RATE)
    actor.train()
    critic.train()
    optimizer_actor.zero_grad()
    optimizer_critic.zero_grad()

    best_perf = -999999

    # assemble experiences from agents, compute the gradients
    while True:

        # record average reward and td loss change
        # in the experiences from the agents
        total_batch_len = 0.0
        total_reward = 0.0
        total_td_loss = 0.0
        total_entropy = 0.0
        total_agents = 0.0

        # assemble experiences from the agents
        actor_loss_num = 0
        critic_loss_num = 0
        actor_total_loss = 0
        critic_total_loss = 0

        for i in range(NUM_AGENTS):
            s_batch, a_batch, r_batch, terminal, info = exp_queues[i].get()

            actor_loss, critic_loss, td_batch = \
                model.compute_loss(
                    s_batch=np.stack(s_batch, axis=0),
                    a_batch=np.vstack(a_batch),
                    r_batch=np.vstack(r_batch),
                    terminal=terminal, actor=actor, critic=critic)
            del s_batch
            del a_batch
            del terminal

            actor_total_loss += actor_loss.detach().numpy()
            critic_total_loss += critic_loss.detach().numpy()

            actor_loss.backward()
            critic_loss.backward()

            actor_loss_num += 1
            critic_loss_num += 1

            total_reward += np.sum(r_batch)
            td_batch = td_batch.detach().numpy()
            total_td_loss += np.sum(td_batch)

            total_batch_len += len(r_batch)
            total_agents += 1.0
            total_entropy += np.sum(info['entropy'])
            del r_batch
            del info

            optimizer_critic.step()
            optimizer_actor.step()
            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()

        actor_net_para = actor.state_dict()
        if cuda:
            for k, v in actor_net_para.items():
                actor_net_para[k] = v.cpu()
        for i in range(NUM_AGENTS):
            net_params_queues[i].put(actor_net_para)
        del actor_net_para

        # compute aggregated gradient
        assert NUM_AGENTS == actor_loss_num
        assert actor_loss_num == critic_loss_num

        # log training information
        epoch_set.append(epoch)
        epoch += 1
        avg_reward = total_reward / total_agents
        qoe_set.append(avg_reward)
        avg_td_loss = total_td_loss / total_batch_len
        avg_entropy = total_entropy / total_batch_len
        actor_avg_loss = actor_total_loss / total_batch_len
        critic_avg_loss = critic_total_loss / total_batch_len

        print('Epoch: ' + str(epoch) +
                ' TD_loss: ' + str(avg_td_loss) +
                ' Actor_loss: ' + str(actor_avg_loss) +
                ' Critic_loss: ' + str(critic_avg_loss) +
                ' Avg_reward: ' + str(avg_reward) +
                ' Avg_entropy: ' + str(avg_entropy))

        if epoch % MODEL_SAVE_INTERVAL == 0:
            # Save the neural net parameters to disk.
            save_path = SUMMARY_DIR + "/nn_model_ep_" + "_" + MODEL + "_" + REWARD + "_" + str(epoch) + ".pth"
            torch.save(actor.state_dict(), save_path)
            print("Model saved in file: " + save_path)
            cur_perf = test.test(save_path, MODEL)
            if cur_perf > best_perf:
                best_perf = cur_perf
                save_path = SUMMARY_DIR + "/nn_model_ep_BEST" + "_" + MODEL + "_" + REWARD + "_" + str(
                    epoch) + ".pth"
                torch.save(actor.state_dict(), save_path)
                print("Best model saved in file: " + save_path)

        if epoch > max_epoch:
            return


def agent(agent_id, all_cooked_time, all_cooked_bw, exp_queue, actor_params_queue, MODEL):
    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw,
                              random_seed=agent_id)


    # Create actor model
    if MODEL == 'mid':
        actor = model.ActorNetwork_mid(state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                        learning_rate=ACTOR_LR_RATE)
    if MODEL == 'big':
        actor = model.ActorNetwork_big(state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                        learning_rate=ACTOR_LR_RATE)
    if MODEL == 'small':
        actor = model.ActorNetwork_small(state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                            learning_rate=ACTOR_LR_RATE)

    # initial synchronization of the network parameters from the coordinator
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

    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1

    s_batch = [np.zeros((S_INFO, S_LEN))]
    a_batch = [action_vec]
    r_batch = []
    entropy_record = []

    time_stamp = 0
    while True:  # experience video streaming forever
        # the action is from the last decision
        # this is to make the framework similar to the real
        # synchronize the network parameters from the coordinator

        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain = \
            net_env.get_video_chunk(bit_rate)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        # -- linear reward --
        # reward is video quality - rebuffer penalty - smoothness
        if REWARD == REWARD_LIST[0]:
            REBUF_PENALTY = 4.3
            reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                        - REBUF_PENALTY * rebuf \
                        - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                                VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

        r_batch.append(reward)

        last_bit_rate = bit_rate

        # retrieve previous state
        if len(s_batch) == 0:
            state = [np.zeros((S_INFO, S_LEN))]
        else:
            state = np.array(s_batch[-1], copy=True)

        # dequeue history record
        state = np.roll(state, -1, axis=1)

        # this should be S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

        # compute action probability vector
        with torch.no_grad():
            actor_input = torch.from_numpy(np.reshape(state, (S_INFO, S_LEN)))
            if cuda:
                actor_input = actor_input.cuda()
            action_prob = actor.forward(actor_input)

            if cuda:
                action_prob = action_prob.cpu()

        # get bit_rate decision. It's training part, so randomly sample
        m = Categorical(action_prob)
        action = m.sample().item()
        bit_rate = action

        entropy_record.append(model.compute_entropy(action_prob[0]))

        # report experience to the coordinator
        if len(r_batch) >= TRAIN_SEQ_LEN or end_of_video:
            exp_queue.put([s_batch[1:],  # ignore the first chuck
                            a_batch[1:],  # since we don't have the
                            r_batch[1:],  # control over it
                            end_of_video,
                            {'entropy': entropy_record}])

            actor_net_para = actor_params_queue.get()
            if cuda:
                for k, v in actor_net_para.items():
                    actor_net_para[k] = v.cuda()
            actor.load_state_dict(actor_net_para)
            del actor_net_para
            del s_batch[:]
            del a_batch[:]
            del r_batch[:]
            del entropy_record[:]


        # store the state and action into batches
        if end_of_video:
            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here
            s_batch.append(np.zeros((S_INFO, S_LEN)))
            a_batch.append(bit_rate)

        else:
            s_batch.append(state)
            a_batch.append(bit_rate)


def train_model(model_type):
    print(f"Starting training for {model_type} model...")
    np.random.seed(RANDOM_SEED)
    assert len(VIDEO_BIT_RATE) == A_DIM

    # create result directory
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

    # create central agent
    central_p = mp.Process(target=central_agent, args=(exp_queues, net_params_queues, model_type))
    central_p.start()

    for i in range(NUM_AGENTS):
        p = mp.Process(target=agent,
                       args=(i, all_cooked_time, all_cooked_bw, exp_queues[i], net_params_queues[i], model_type))
        processes.append(p)

    for i in range(NUM_AGENTS):
        processes[i].start()

    central_p.join()

    print(f"Training for {model_type} model finished.")


def main():
    parser = argparse.ArgumentParser(description='Train and Test Pensieve Models')
    parser.add_argument('--model', choices=MODEL_LIST, default='all', help='Choose the model size (small, mid, big, all)')
    args = parser.parse_args()

    global MODEL
    MODEL = args.model  # Set the MODEL value based on command-line argument

    if MODEL == 'all':
        # Train all models in sequence: small, mid, big
        for model_type in MODEL_LIST[:-1]:  # Skip 'all' in the list
            train_model(model_type)
    else:
        # Train the specified model
        train_model(MODEL)


if __name__ == '__main__':
    main()

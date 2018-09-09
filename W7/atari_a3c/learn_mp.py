import torch
from torch.optim import Adam
import torch.nn as nn
import time
from .atari import create_atari_env
from .models import Agent
from tqdm import tqdm
import torch.multiprocessing as mp


LEARNING_RATE = 1e-4
WORKERS = 4
JOB_BLOCK = 50
ACTOR_WEIGHT = 0.5
MAX_PLAY_STEPS = 20

train_progress_queue = mp.Queue(1000)


def play(env, agent, first_state,
         max_steps=20, render=False, action_code=(0, 2, 3)):
    done = False
    steps = 0
    state = first_state
    trajectory = {'states': [],
                  'rewards': [],
                  'actions_logprob': [],
                  'actions': [],
                  'critic_values': []}
    while not done and steps < max_steps:
        if render:
            env.render()

        # agent: assessing state and draw action
        state_tensor = torch.from_numpy(state).unsqueeze(0)
        action_likelihood, critic_val = agent(state_tensor)
        action_prob = nn.functional.softmax(action_likelihood, dim=1)
        action_logprob = nn.functional.log_softmax(action_likelihood, dim=1)
        action = action_prob.multinomial(1)
        action = action.item()  # this tensor has only one element, extract it

        # commit action
        new_state, reward, done, _ = env.step(action_code[action])

        # record history
        # trajectory['states'].append(state)
        trajectory['rewards'].append(reward)
        trajectory['actions'].append(action)  # extract single-num
        trajectory['actions_logprob'].append(action_logprob[0, action])
        trajectory['critic_values'].append(critic_val)

        state = new_state
        if render:
            time.sleep(0.05)
        steps += 1

    return trajectory, state, done
    # return state/done for learning-controller
    # if not done, the returned state can be used as the "first_state"
    # in the next round of calling play()


def estimate_policy_gradient(trajectory, optim, gamma=0.99):
    """
    :param trajectory:
    :param optim:
    :type optim: torch.optim.Adam
    :param gamma: discount factor
    :return:
    """
    total_reward = 0
    actor_objective = 0.0
    critic_objective = 0.0
    for r, logp, cv in zip(reversed(trajectory['rewards']),
                       reversed(trajectory['actions_logprob']),
                       reversed(trajectory['critic_values'])):
        # the accumulated reward starting from this time-step
        # until the end of the episode
        total_reward = total_reward*gamma + r
        advantage = total_reward - cv
        actor_objective += advantage.detach() * logp  # for actor,
        # we treat advantage as a coefficient, not some value to adjust

        critic_objective += advantage ** 2

    total_objective = critic_objective - actor_objective * ACTOR_WEIGHT
    optim.zero_grad()
    total_objective.backward()
    optim.step()


def policy_gradient(rank, env_name, agent):
    """
    :param rank:
    :param env:
    :param agent:
    :type agent: torch.nn.Module
    :return:
    NB: agent is shared between multiple processes
    """
    env = create_atari_env(env_name)
    optim = Adam(agent.parameters(), lr=LEARNING_RATE)
    state = env.reset()

    worker_train_steps = 0
    while True:
        trj, state, done = play(env, agent, state, max_steps=MAX_PLAY_STEPS)
        if done:
            state = env.reset()  # start over when game-over
        estimate_policy_gradient(trj, optim)
        worker_train_steps += 1
        if worker_train_steps % JOB_BLOCK == 0:
            # print("W{}:{}".format(rank, sum(trj['rewards'])))
            train_progress_queue.put(JOB_BLOCK)


def running_avg(a, x, f):
    return x if a is None else (1.0-f) * x + f * a


def evaluate_policy(env_name, agent):
    """
    :param env_name:
    :param agent: shared
    :return:
    """
    render_every_n_evaluation = 5
    save_every_n_evaluation = 5
    report_every_n_steps = 2000
    env = create_atari_env(env_name)

    total_steps = 0
    render_count = render_every_n_evaluation
    save_count = save_every_n_evaluation
    report_count = report_every_n_steps
    report_train_steps = 0
    pbar = tqdm(total=report_every_n_steps)

    avg_score_l = None
    avg_score_s = None
    avg_len_l = None
    avg_len_s = None
    lf = 0.99
    sf = 0.75

    while True:
        n = train_progress_queue.get()
        total_steps += n

        if report_count < 1:
            report_count = report_every_n_steps
            do_render = False

            if render_count < 1:
                render_count = render_every_n_evaluation
                do_render = True
            render_count -= 1

            # let's check the agent's performance
            state = env.reset()
            trj, _, _ = play(env, agent, state, max_steps=5000, render=do_render)
            rs = trj['rewards']
            score = sum(rs)
            tlen = len(rs)
            avg_score_l = running_avg(avg_score_l, score, lf)
            avg_score_s = running_avg(avg_score_s, score, sf)
            avg_len_l = running_avg(avg_len_l, tlen, lf)
            avg_len_s = running_avg(avg_len_s, tlen, sf)
            print("Train {}: Test score {}, Trajectory Length {}"
                  "Long/Short Avg R {:.3f} / {:.3f}"
                  "Long/Short Avg Length {:.3f} / {:.3f}"
                .format(total_steps, score, tlen, avg_score_l, avg_score_s,
                        avg_len_l, avg_len_s))

            pbar.close()
            pbar = tqdm(total=report_every_n_steps)

            if save_count < 1:
                save_count = save_every_n_evaluation
                state_dict = agent.state_dict()
                torch.save(state_dict,
                       'atari_a3c/checkpoints/'
                       'a3c_{}.pth'.format(total_steps))
            save_count -= 1

        report_count -= n
        pbar.update(n)


def start_learning(num_workders, env_name):
    shared_agent = Agent(input_channels=4, feat_num=288, actions=3)
    shared_agent.share_memory()
    processes = []
    for rank in range(num_workders):
        p = mp.Process(target=policy_gradient,
                       args=(rank, env_name, shared_agent))
        p.start()
        processes.append(p)

    p = mp.Process(target=evaluate_policy, args=(env_name, shared_agent))
    p.start()
    processes.append(p)

    for p in processes:
        p.join()


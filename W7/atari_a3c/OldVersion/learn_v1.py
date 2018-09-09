import torch
from torch.optim import Adam
import torch.nn as nn
import time
from tqdm import tqdm


def play(env, agent, first_state,
         max_steps=20, render=False, action_code=(0, 2, 3)):
    done = False
    steps = 0
    state = first_state
    trajectory = {'states': [],
                  'rewards': [],
                  'actions_logprob': [],
                  'actions': []}
    while not done and steps < max_steps:
        if render:
            env.render()

        # agent: assessing state and draw action
        state_tensor = torch.from_numpy(state).unsqueeze(0)
        action_likelihood = agent(state_tensor)
        action_prob = nn.functional.softmax(action_likelihood, dim=1)
        action_logprob = nn.functional.log_softmax(action_likelihood, dim=1)
        action = action_prob.multinomial(1)
        action = action.item()  # this tensor has only one element, extract it

        # commit action
        new_state, reward, done, _ = env.step(action_code[action])

        # record history
        trajectory['states'].append(state)
        trajectory['rewards'].append(reward)
        trajectory['actions'].append(action)  # extract single-num
        trajectory['actions_logprob'].append(action_logprob[0, action])

        state = new_state
        if render:
            time.sleep(0.05)
        steps += 1

    return trajectory, state, done
    # return state/done for learning-controller
    # if not done, the returned state can be used as the "first_state"
    # in the next round of calling play()


def estimate_policy_gradient_v0(trajectory, optim):
    """
    :param trajectory:
    :param optim:
    :type optim: torch.optim.Adam
    :return:
    """
    total_reward = 0
    total_objective = 0.0
    for r, logp in zip(reversed(trajectory['rewards']),
                       reversed(trajectory['actions_logprob'])):
        total_reward += r  # the accumulated reward starting from this time-step
        # until the end of the episode
        total_objective += total_reward * logp

    total_objective = -total_objective
    optim.zero_grad()
    total_objective.backward()
    optim.step()


def policy_gradient_v0(env, agent):
    """
    :param env:
    :param agent:
    :type agent: torch.nn.Module
    :return:
    """
    optim = Adam(agent.parameters(), lr=5e-5)
    state = env.reset()
    train_steps = 0
    report_train_steps = 0
    report_every_n_steps = 200

    pbar = tqdm(total=report_every_n_steps)
    while True:
        trj, state, done = play(env, agent, state, max_steps=20)
        if done:
            state = env.reset()  # start over when game-over
            if report_train_steps > report_every_n_steps:
                # let's check the agent's performance
                trj, _, _ = play(env, agent, state, max_steps=1000000)
                state = env.reset()
                print("Train {}: Test score {}".format(
                    train_steps, sum(trj['rewards'])))
                report_train_steps = 0
                pbar.close()
                pbar = tqdm(total=report_every_n_steps)
                state_dict = agent.state_dict()
                torch.save(state_dict,
                           'atari_a3c/checkpoints/'
                           'vanilla_{}.pth'.format(train_steps))

        estimate_policy_gradient_v0(trj, optim)
        train_steps += 1
        report_train_steps += 1
        pbar.update()



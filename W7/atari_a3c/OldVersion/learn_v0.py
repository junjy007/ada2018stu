import torch
import torch.nn as nn
import time


def play(env, agent, first_state,
         max_steps=20, render=False, action_code=[0, 2, 3]):
    done = False
    steps = 0
    state = first_state
    while not done and steps < max_steps:
        if render:
            env.render()

        state_tensor = torch.from_numpy(state).unsqueeze(0)
        action_likelihood = agent(state_tensor)
        action_prob = nn.functional.softmax(action_likelihood, dim=1)
        action = action_prob.multinomial(1)

        new_state, reward, done, _ = env.step(action_code[action])
        state = new_state
        if render:
            time.sleep(0.05)
        steps += 1

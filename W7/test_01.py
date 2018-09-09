from atari_a3c import Agent, create_atari_env
from atari_a3c.learn import policy_gradient

if __name__ == '__main__':
    g_env = create_atari_env('PongDeterministic-v4')
    g_agent = Agent(input_channels=4, feat_num=288, actions=3)
    policy_gradient(g_env, g_agent)
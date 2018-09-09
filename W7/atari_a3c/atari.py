import cv2
import gym
import numpy as np
from gym.spaces.box import Box


# Taken from https://github.com/openai/universe-starter-agent
def create_atari_env(args):
    # This is to allow short-cut create such as
    # create_atari_env('PongDeterministic-v4')
    if isinstance(args, str) or args is None:
        from .options import parser
        args_ = [] if args is None else ['--env-name', args]
        args = parser.parse_args(args_)

    env = gym.make(args.env_name).env
    env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env._max_episode_steps = args.max_episode_length
    env = AtariRescale42x42(env, not args.colour_frame)
    env = NormalizedEnv(env)
    env = RepeatActionFrame(env, args.repeat_action_steps)
    return env


def _process_frame42(frame, mono_colour=True):
    """
    :param frame:
    :param mono_colour: if process image as mono colour
    :return:
    """
    frame = frame[34:34 + 160, :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    if mono_colour:
        frame = frame.mean(2, keepdims=True)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.moveaxis(frame, -1, 0)
    return frame


class AtariRescale42x42(gym.ObservationWrapper):
    def __init__(self, env, mono=True):
        super(AtariRescale42x42, self).__init__(env)
        self.mono_ = mono
        self.observation_space = Box(0.0, 1.0, [1, 42, 42], dtype=np.uint8)

    # noinspection PyMethodMayBeStatic
    def observation(self, observation):
        return _process_frame42(observation, self.mono_)


class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # for Qbert sometimes we stay in lives == 0 condtion for a few
            # frames so its important to keep lives > 0, so that we only reset
            # once the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, self.was_real_done

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class RepeatActionFrame(gym.Wrapper):
    def __init__(self, env, num_repeat):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        self.num_repeat = num_repeat

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs = np.concatenate([obs]*self.num_repeat, axis=0)
        return obs

    def step(self, ac):
        ss = []
        rew = 0.
        done = False
        obs, info = None, None
        for _ in range(self.num_repeat):
            if done:
                ss.append(obs)
                continue
            obs, reward, done, info = self.env.step(ac)
            ss.append(obs)
            rew += reward
        obs = np.concatenate(ss, axis=0)
        return obs, rew, done, info

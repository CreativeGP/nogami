import time

import numpy as np
import gym

def rootdir(s):
    import sys
    return sys.path[0]+s

def show_autograd_graph(grad_fn, indent=0):
    print((' '*indent) + grad_fn.__class__.__name__)
    if indent >= 10:
        return
    if not hasattr(grad_fn, 'next_functions'):
        return
    for i in grad_fn.next_functions:
        show_autograd_graph(i[0], indent+1)
def count_autograd_graph(grad_fn):
    if not hasattr(grad_fn, 'next_functions'):
        return 0
    res = len(grad_fn.next_functions)
    for i in grad_fn.next_functions:
        res += count_autograd_graph(i[0])
    return res

# Class for time measurement
class Timer:
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.times.append(time.time() - self.start_time)

    def mean(self):
        return np.mean(self.times)

    def sum(self):
        return np.sum(self.times)
    
    def clear(self):
        self.times = []

    def total(self):
        return time.time() - self.times[0]

    def __str__(self):
        return f"Timer: {self.mean()}s"

    def __repr__(self):
        return self.__str__()

class Logger():
    def __init__(self, run_name, args):
        import torch.utils.tensorboard

        self.run_name = run_name
        self.args = args
        self.writer = torch.utils.tensorboard.SummaryWriter(rootdir(f"runs/{run_name}"))
        self.data = {}

        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.args).items()])),
        )
    
    # helper function to write data
    def write_mean(self, title, key, global_step):
        self.writer.add_scalar(title, sum(self.data[key]) / len(self.data[key]), global_step)

    def write_histogram(self, title, key, global_step):
        self.writer.add_histogram(title, np.array(self.data[key]), global_step)
    
    def close(self):
        self.writer.close()


# NOTE(cgp): infoの扱いがちょっと面倒なので、ちゃんとarrayを返すようにした.
class CustomizedSyncVectorEnv(gym.vector.SyncVectorEnv):
    def reset(self):
        obs, _infos = super().reset()
        infos = []
        for i in range(self.num_envs):
            infos.append({key: value[i] for key, value in _infos.items()})
        return obs, infos

    
    def step(self, actions):
        obs, rewards, terminates, truncates, _infos = super().step(actions)
        infos = []
        for i in range(self.num_envs):
            infos.append({key: value[i] for key, value in _infos.items()})
        return obs, rewards, terminates, truncates, infos

class CustomizedAsyncVectorEnv(gym.vector.AsyncVectorEnv):
    def reset(self):
        obs, _infos = super().reset()
        infos = []
        for i in range(self.num_envs):
            infos.append({key: value[i] for key, value in _infos.items()})
        return obs, infos

    
    def step(self, actions):
        obs, rewards, terminates, truncates, _infos = super().step(actions)
        infos = []
        for i in range(self.num_envs):
            infos.append({key: value[i] for key, value in _infos.items()})
        return obs, rewards, terminates, truncates, infos

def tracefunc(frame, event, arg, indent=[0]):
    if event == "call":
        indent[0] += 2
        try:
            print("-" * indent[0] + "> call function", frame.f_code.co_name, frame.f_locals)
        except:
            print("-" * indent[0] + "> call function", frame.f_code.co_name)
    elif event == "return":
        print("<" + "-" * indent[0], "exit function", frame.f_code.co_name)
        indent[0] -= 2
    return tracefunc
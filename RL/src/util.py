import time
from socket import gethostname
import sys

import numpy as np
import gym

# NOTE(cgp): /は含まないディレクトリパスを返すので注意
def rootdir(s):
    root = '/'.join(__file__.split('/')[:-2])
    # if not root.endswith('/'):
    #     root += '/'
    if root.endswith('/'):
        root = root[:-1]
    return root+s

# a.grad_fn
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

def print_random_states(show_hash=False):
    import random,json,hashlib,io,pickle
    import torch
    import numpy as np

    if show_hash:
        l = 5
        md5_hash = hashlib.md5()
        md5_hash.update(pickle.dumps(random.getstate()))
        print("python random Random State:", md5_hash.hexdigest()[:l])

        md5_hash = hashlib.md5()
        buff = io.BytesIO(); torch.save(torch.get_rng_state(), buff); buff.seek(0)
        md5_hash.update(buff.read())
        print("torch CPU Random State:", md5_hash.hexdigest()[:l])

        if torch.cuda.is_available():
            md5_hash = hashlib.md5()
            buff = io.BytesIO(); torch.save(torch.cuda.get_rng_state(), buff); buff.seek(0)
            md5_hash.update(buff.read())
            print("GPU Random State:", md5_hash.hexdigest()[:l])

        md5_hash = hashlib.md5()
        md5_hash.update(pickle.dumps(np.random.get_state()))
        print("Numpy random Random State:", md5_hash.hexdigest()[:l])
    else:
        print("python random Random State:", random.getstate())
        print("torch CPU Random State:",torch.get_rng_state())
        if torch.cuda.is_available():
            print("GPU Random State:",torch.cuda.get_rng_state(device='cuda'))
        print("Numpy random Random State:", np.random.get_state())


# 接頭辞を指定して、その接頭辞を持つ引数をdictにまとめる
# def load_args(args: "argparse.Namespace", prepared_dict={}, prefix=""):
#     if len(prepared_dict) == 0:
#         d = {}
#         for key, value in vars(args).items():
#             if key.startswith(prefix):
#                 d[key] = value
#         return d
#     else:
#         for key, value in vars(args).items():
#             if key.startswith(prefix) and key in prepared_dict:
#                 prepared_dict[key] = value
#         return prepared_dict

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
    def __init__(self, run_name, args, use_wandb=False):
        import torch.utils.tensorboard

        self.run_name = run_name
        self.args = args
        self.writer = torch.utils.tensorboard.SummaryWriter(rootdir(f"/runs/{run_name}"))
        self.data = {}
        
        self.use_wandb = use_wandb

        if use_wandb:
            import random
            randst = random.getstate()
            # NOTE: これ、randomを変えるようなのでビックリ
            import wandb
            wandb.login(key='07513fa9ea56e3b37748d95ff8c09a39650e397b')
            wandb.init(project="zxrl", name=run_name, config=vars(args), )
            random.setstate(randst)
        
        self.write_dict("hyperparameters", vars(args))
        # self.writer.add_text(
        #     "hyperparameters",
        #     "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.args).items()])),
        # )
        self.write_text("hostname", gethostname())
        # self.writer.add_text("hostname", gethostname())

    def write_text(self, title, text):
        self.writer.add_text(title, text)
        if self.use_wandb:
            import wandb
            wandb.config[title] = text
    
    def write_dict(self, title:str, d:dict):
        self.writer.add_text(title, "|key|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in d.items()]))
        )
        if self.use_wandb:
            import wandb
            wandb.config[title] = d

    def write_scalar(self, title, value, global_step, only_wandb=False):
        if not only_wandb:
            self.writer.add_scalar(title, value, global_step)
        if self.use_wandb:
            import wandb
            wandb.log({title: value}, step=global_step)
    
    # helper function to write data
    def write_mean(self, title, key, global_step, only_wandb=False):
        m = sum(self.data[key]) / len(self.data[key])
        if not only_wandb:
            self.writer.add_scalar(title, m, global_step)
        if self.use_wandb:
            import wandb
            wandb.log({title: m}, step=global_step)

    def write_histogram(self, title, key, global_step):
        self.writer.add_histogram(title, np.array(self.data[key]), global_step)

        # wandb does not support histogram
    
    def close(self):
        self.writer.close()
        if self.use_wandb:
            import wandb
            wandb.finish()


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


import sys
from itertools import chain
from collections import deque

def compute_object_size(o, handlers={}):
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = sys.getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = sys.getsizeof(o, default_size)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)

def all_memory_usage(thres=1e6):
    import inspect

    memsum = 0
    for obj in gc.get_objects():
        size = compute_object_size(obj)
        memsum += size
        if size > thres:
            print(size, inspect.getmembers(obj) )

    print(f"Memory usage: {memsum} bytes")


# get random shuffled minibatches
# originalと若干実装が異なるが、PPGは完全一致はできないからいいのでは
def for_minibatches(batch_size, minibatch_size):
    b_inds = np.arange(batch_size)
    np.random.shuffle(b_inds)  
    for start in range(0, batch_size, minibatch_size):
        yield b_inds[start:start + minibatch_size]

# nn.Moduleに対して、以下の全てのモジュールにforward_hookを登録する
def register_all_forward_hooks(model, prefix="", indent=0):
    model.register_forward_hook(forward_hook)
    model.ud_indent = indent
    model.ud_prefix = prefix
    if len(model._modules) > 0:
        model.register_forward_pre_hook(forward_pre_hook)
        model.ud_has_children = True
    else:
        model.ud_has_children = False
    for name, module in model._modules.items():
        if module is None:
            continue
        register_all_forward_hooks(module, prefix=prefix+'.'+name, indent=indent+1)

# for register_forward_hook
def forward_hook(model, inputs, output):
    import torch
    in_shapes = []
    for i in inputs:
        if isinstance(i, torch.Tensor):
            in_shapes.append(list(i.shape))
        else:
            in_shapes.append(None)
    INDENT = "    "*model.ud_indent
    print(INDENT, )
    if not model.ud_has_children:
        print(INDENT, '>'*model.ud_indent, f"({model.ud_prefix})", model.__class__.__name__)
        for name, param in model.named_parameters():
            print(INDENT, '+', name, param.data.shape)
        print(INDENT, "inputs", in_shapes, "\n".join([INDENT+l for l in inputs.__repr__().split('\n')]))
    else:
        print(INDENT, '↳', f"({model.ud_prefix})", model.__class__.__name__)
    print(INDENT, "output", output.shape, "\n".join([INDENT+l for l in output.__repr__().split('\n')]))

# for register_forward_pre_hook
def forward_pre_hook(model, inputs):
    import torch
    in_shapes = []
    for i in inputs:
        if isinstance(i, torch.Tensor):
            in_shapes.append(list(i.shape))
        else:
            in_shapes.append(None)
    INDENT = "    "*model.ud_indent
    print(INDENT, )
    print(INDENT, '>'*model.ud_indent, f"({model.ud_prefix})", model.__class__.__name__)
    print(INDENT, "inputs", in_shapes, "\n".join([INDENT+l for l in inputs.__repr__().split('\n')]))

# 重みの統計情報を表示
def print_weight_summary(model):
    print("Weights for ", model.__class__.__name__)
    for name, param in model.named_parameters():
        s = str(list(param.data.shape))
        print(f"{name:<50} {s:<10} μ={param.data.mean().item():.3f}\tstd={param.data.std().item():.3f}")

# 重みの値を表示
def print_weights(model, tag:str=""):
    for name, param in model.named_parameters():
        if tag in name or tag == "":
            print(name)
            print(param.data)

# 勾配の統計情報を表示
def print_grad_summary(model):
    print("Gradients for ", model.__class__.__name__)
    for name, param in model.named_parameters():
        s = str(list(param.grad.shape))
        print(f"{name:<50} {s:<10} μ={param.grad.mean().item():.3e}\tstd={param.grad.std().item():.3e}")

# 勾配の値を表示
def print_grads(model, tag:str=""):
    for name, param in model.named_parameters():
        if name == tag or tag == "":
            print(name)
            print(param.grad)
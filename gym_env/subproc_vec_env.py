import gym
import queue as queue
import numpy as np
import multiprocessing as mp
from misc.utils import make_env


class EnvWorker(mp.Process):
    """Each worker for each environment
    Ref: https://github.com/tristandeleu/pytorch-maml-rl/blob/master/maml_rl/envs/subproc_vec_env.py
    """
    def __init__(self, remote, queue, lock, log, args):
        super(EnvWorker, self).__init__()
        self.remote = remote
        self.env = make_env(log, args).env
        self.queue = queue
        self.lock = lock
        self.log = log
        self.args = args
        self.done = False

    def empty_step(self):
        observations = [np.ones(self.args.batch_size, dtype=np.int64) * -1 for _ in range(self.args.n_agent)]
        rewards = [np.ones(self.args.batch_size, dtype=np.float32) for _ in range(self.args.n_agent)]
        done = True
        return observations, rewards, done, {}

    def try_reset(self):
        with self.lock:
            try:
                self.id = self.queue.get(True)
                self.done = (self.id is None)
            except queue.Empty:
                self.done = True

        if self.done:
            observations = [np.ones(self.args.batch_size, dtype=np.int64) * -1 for _ in range(self.args.n_agent)]
        else:
            observations = self.env.reset()
        return observations

    def set_new_task(self):
        self.env.set_new_task()

    def run(self):
        while True:
            command, data = self.remote.recv()
            if command == 'step':
                if self.done:
                    observations, rewards, done, _ = self.empty_step()
                else:
                    observations, rewards, done, _ = self.env.step(data)
                if done and (not self.done):
                    observations = self.try_reset()
                self.remote.send((observations, rewards, done, self.id))
            elif command == 'reset':
                observations = self.try_reset()
                self.remote.send((observations, self.id))
            elif command == 'close':
                self.remote.close()
                break
            else:
                raise ValueError("Invalid command")


class SubprocVecEnv(gym.Env):
    def __init__(self, n_worker, queue, log, args):
        self.lock = mp.Lock()
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(n_worker)])
        self.workers = [
            EnvWorker(remote, queue, self.lock, log, args)
            for remote in self.work_remotes]
        for worker in self.workers:
            worker.daemon = True
            worker.start()
        for remote in self.work_remotes:
            remote.close()
        self.waiting = False
        self.closed = False

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        observations, rewards, done, id = zip(*results)
        return np.stack(observations), np.stack(rewards), np.stack(done), id

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        observations, id = zip(*results)
        return np.stack(observations), id

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for worker in self.workers:
            worker.join()
        self.closed = True

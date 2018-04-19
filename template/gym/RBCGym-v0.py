import math
import gym
from gym import *
from gym import spaces
from gym.utils import seeding
import numpy as np
import pandas as pd

from statsmodels.tsa.vector_ar import var_model

def limiter10(x):
    if x<=0.0:
        return 1e-4
    if x>=1.0:
        return 1.0-1e-4
    return(x)
class MyGym(gym.Env):
    # based on description of model in
    metadata = {'render.modes': ['human']}
    def __init__(self,**kwargs):

        self.rho=.9
        self.sigma=np.sqrt(.0038)
        self.alpha=.7
        self.delta=.025
        self.phi=.5 # value of consumption relative to leisure -- .5 is balanced value between the two
        self.episode_len=100

        self._reset()


        self.action_space = spaces.Box(low=1e-8, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-1.0e8, 1.0e8, shape=self.observation.shape, dtype=np.float32)

    @property
    def state(self):
        return {
            "k_t":self.k_t,
            "k_t1":self.k_t1,
            "z_t":self.z_t,
            "y_t":self.y_t
        }

    @property
    def observation(self):
        return np.array([
            self.y_t,
            self.y_fd,
            self.y_fd0,
            self.k_t,
            self.k_t1,
            self.k_fd,
            self.k_sd,
            self.z_t,
            self.t])

    def store_last_obs(self):
        self.y_t0=self.y_t
        self.y_fd0=self.y_fd
        self.k_t=self.k_t1
        self.k_fd0=self.k_fd

    def mk_diff(self):
        self.y_fd=self.y_t-self.y_t0
        self.y_sd=self.y_fd-self.y_fd0
        self.k_fd=self.k_t1-self.k_t
        self.k_sd=self.k_fd-self.k_fd0


    def _step(self,action):
        self.store_last_obs()
        # get economic shocks:
        #self.z_t=np.exp((self.rho*np.log(self.z_t))+np.random.normal(0,self.sigma**2,1)[0])
        self.l_t=limiter10(action[1])
        self.c_t=limiter10(action[0])
        self.c_action=limiter10(action[0])

        # set labor based on chosen level of leisure
        self.n_t=1.0-self.l_t

        # get current output given labor and shocks
        self.y_t = self.z_t * (self.k_t ** self.alpha) * self.n_t ** (1.0 - self.alpha)

        # set consumption and leisure based on actions
        self.c_t = self.c_t * self.y_t if self.y_t>=0 else 9e-5

        # set investment (k_1
        self.i_t=self.y_t-self.c_t
        self.k_t1=self.k_t*(1-self.delta)+self.i_t

        self.u_t = (self.phi * np.log(self.c_t)) + ((1 - self.phi) * np.log(self.l_t))
        self.t += 1

        # add to total utilities since beginning of episode
        self.sum_u_t+= self.u_t
        done=(self.t>=self.episode_len or np.isnan(self.u_t)) #or self.sum_u_t<=-10.0
        #done = False
        self.mk_diff()
        return self.observation, self.u_t, done, {"reward":self.u_t, "c":self.c_t, "l":self.l_t}

    def _render(self,mode='human',close=False):
        if not close:
            return {"t":self.t,"n_t":self.n_t,"action":[   self.c_action,self.l_t],'reward':[self.u_t,self.sum_u_t],"state":self.state}

    def _reset(self):
        # initialize time and sum_u_t vars
        self.t=0
        self.sum_u_t=0.0

        # initialize k_t and k_t1 vars and fd
        self.k_t1=self.k_t=1.0
        self.k_fd=self.k_fd0=0.0
        self.k_sd=0.0

        # initialize z_t
        self.z_t=1.0

        # initialize
        self.y_t=self.y_t0=1.0
        self.y_fd=self.y_fd0=0.0
        self.y_sd=0.0


        return self.observation
    def _seed(self,seed=None):
        if seed:
            np.random.seed(seed)
        else:
            np.random.seed(123)


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
        return 1e-12
    if x>=1.0:
        return 1.0-1e-4
    return(x)
class MyGym(gym.Env):
    # based on description of model in
    metadata = {'render.modes': ['human']}
    def __init__(self,**kwargs):
        self.action_space=spaces.Box(np.array([1e-8]),np.array([1.0]))
        self._reset()
        print(self.observation.shape)
        self.observation_space = spaces.Box(-1.0e8, 1.0e8, self.observation.shape)
        self.z_t=1
        self.rho=.9
        self.sigma=np.sqrt(.0038)
        self.alpha=.7
        self.delta=.025
        self.phi=.8 # value of consumption relative to leisure -- .5 is balanced value between the two
        self.episode_len=99
        self.t=0
        self.sum_u_t=0.0
        self.k_t1=1.0
        self.l_t=kwargs.get("l_t",.3)
        self.n_t=1.0-self.l_t



    def _step(self,action):
        # set current capital based on last period ending capital
        old_observation=self.observation.copy()
        self.k_t=self.k_t1
        # get economic shocks:
        self.z_t=np.exp((self.rho*np.log(self.z_t))+np.random.normal(0,self.sigma**2,1)[0])
        #self.z_t=1
        action[0]=limiter10(action[0])
        self.c_action=action[0]
        # get current output given labor and shocks
        self.y_t = self.z_t * (self.k_t ** self.alpha) * self.n_t ** (1.0 - self.alpha)
        # set consumption and leisure based on actions
        self.c_t = action[0] * self.y_t if self.y_t>0 else 9e-5
        # set investment
        self.i_t=self.y_t-self.c_t
        self.k_t1=self.k_t*(1-self.delta)+self.i_t
        # get ready to output state
        # self.state={"k_t":self.k_t,"k_t1":self.k_t1,"y_t":self.y_t,"i_t":self.i_t,"c_t":self.c_t,"l_t":self.l_t,"n_t":self.n_t}
        self.state={
            "k_t":self.k_t,
            "k_t1":self.k_t1,
            "z_t":self.z_t,
            "y_t":self.y_t
        }
        self.set_observation(self.y_t, self.k_t, self.k_t1, self.z_t,old_observation=old_observation)
        self.u_t = (self.phi * np.log(self.c_t)) + ((1 - self.phi) * np.log(self.l_t))


        self.t += 1
        # add to total utilities since beginning of episode
        self.sum_u_t+= self.u_t
        done=(self.t>=self.episode_len or np.isnan(self.u_t)) or (self.y_t<0)
        return self.observation, self.u_t,done,{"reward":self.u_t}
    def _render(self,mode='human',close=False):
        if not close:
            print({"t":self.t,"n_t":self.n_t,"action":[   self.c_action,self.l_t],'reward':[self.u_t,self.sum_u_t],"state":self.state})
    def _reset(self):
        self.t=0
        self.sum_u_t=0.0
        self.k_t1=10.0
        self.z_t=1.0
        self.k_t=self.k_t1
        self.y_t=0.0
        self.c_t=0.0
        self.state={
            "k_t":self.k_t,
            "k_t1":self.k_t1,
            "z_t":self.z_t
        }
        #self.episode_len=np.random.choice(range(50,99))
        self.set_observation(self.y_t,self.k_t,self.k_t1,self.z_t,
                             old_observation=[self.y_t,self.k_t,self.k_t1,self.z_t])
        #
        # "k_t": self.k_t,
        # "k_t1": self.k_t1,
        # "y_t": self.y_t,
        # "i_t": self.i_t,
        # "c_t": self.c_t,
        # "l_t": self.l_t,
        # "n_t": self.n_t
        return self.observation
    def set_observation(self,y_t,k_t,k_t1,z_t,**kwargs):
        if kwargs.get('old_observation',None) is not None:
            obs=kwargs['old_observation']
            # dk_t=k_t/obs[0]
            # dk_t1=k_t1/obs[1]
            # dy_t=y_t/obs[3]
            # self.observation = np.array([k_t, k_t1,y_t,dk_t,dk_t1,dy_t])
            self.observation = np.array([k_t,self.c_t, k_t1, y_t,self.t])
        else:
            self.observation= np.array([k_t, k_t1, z_t,y_t])
        return  self.observation
    def _seed(self,seed=None):
        if seed:
            np.random.seed(seed)
        else:
            pass


# gym.envs.register(
#    	id='RBCGym-v0',
#    	entry_point='RBCGym',
# )

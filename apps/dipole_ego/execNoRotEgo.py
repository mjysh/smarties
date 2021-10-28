#!/usr/bin/env python3
##
##  smarties
##  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
##  Distributed under the terms of the MIT license.
##
##  Created by Guido Novati (novatig@ethz.ch).
##

import sys, os, numpy as np
from NoRotEgoOneSensor_env import *

#import pybullet_envs
#import pybulletgym
os.environ['MUJOCO_PY_FORCE_CPU'] = '1'
import smarties as rl

def getAction(comm, env):
  buf = comm.recvAction()
  if   hasattr(env.action_space, 'n'):
    action = int(buf[0])
  elif hasattr(env.action_space, 'spaces'):
    action = [int(buf[0])]
    for i in range(1, comm.nActions): action = action + [int(buf[i])]
  elif hasattr(env.action_space, 'shape'):
    action = buf
  else: assert(False)
  return action

def setupSmartiesCommon(comm):
    ############## Hyperparameters ##############
    mode = 'CFD'
    env_name = "singleDipole-v0"
    dt = 0.1
    render = False              # render the environment in training if true
    solved_reward = 250         # stop training if avg_reward > solved_reward
    log_interval = 1           # print avg reward in the interval
    max_episodes = 30000        # max training episodes
    max_timesteps = 495         # max timesteps in one episode
    
    update_timestep = 4000      # update policy every n timesteps
    action_std = 0.5            # constant std for action distribution (Multivariate Normal)
    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor
    
    lr = 0.0003                 # parameters for Adam optimizer
    betas = (0.9, 0.999)
    fishflex = 0.5                  # amount of change allowed in vortex strength
    # backgroundFlow = -1.0
    alpha = 4
    r = 3
    mu = 0.8
    trainregion_offset = 0.15
    random_seed = None
    list_setting = ["decision time step: {}".format(dt), \
                    "max episodes: {}".format(max_episodes), \
                    "max time steps: {}".format(max_timesteps), \
                    "update time steps: {}".format(update_timestep), \
                    "standard deviation in actions: {}".format(action_std), \
                    "number of update epochs: {}".format(K_epochs), \
                    "clip parameter: {}".format(eps_clip), \
                    "discount factor: {}".format(gamma), \
                    "Adam optimizer lr: {}".format(lr), \
                    "Adam optimizer betas: {}".format(betas),\
                    "fish flexibility: {}".format(fishflex), \
                    "mu: {}".format(mu), \
                    "offset: {}".format(trainregion_offset), \
                    "alpha: {}".format(alpha), \
                    "NN: 128*128",\
                    "Cut = 0.6, circular initial zone"]
    with open("./setting.txt","w") as file_setting:
        file_setting.write("\n".join(list_setting))
    #############################################
    
    # creating environment
    env = DipoleSingleEnv(mode=mode,dt = dt,mu=mu,alpha = alpha, flexibility = fishflex,cfd_dumpinterval=5,cfd_framerate = 1000,cfd_inittime = 150)
    
    # set the length of an episode
    from gym.wrappers.time_limit import TimeLimit
    env = TimeLimit(env, max_episode_steps=max_timesteps)
      
    ## setup MDP properties:
    # first figure out dimensionality of state
    dimState = 1
    if hasattr(env.observation_space, 'shape'):
      for i in range(len(env.observation_space.shape)):
        dimState *= env.observation_space.shape[i]
    elif hasattr(env.observation_space, 'n'):
      dimState = 1
    else: assert(False)

    # then figure out action dims and details
    if hasattr(env.action_space, 'spaces'):
      dimAction = len(env.action_space.spaces)
      comm.setStateActionDims(dimState, dimAction, 0) # 1 agent
      control_options = dimAction * [0]
      for i in range(dimAction):
        control_options[i] = env.action_space.spaces[i].n
      comm.setActionOptions(control_options, 0) # agent 0
    elif hasattr(env.action_space, 'n'):
      dimAction = 1
      comm.setStateActionDims(dimState, dimAction, 0) # 1 agent
      comm.setActionOptions(env.action_space.n, 0) # agent 0
    elif hasattr(env.action_space, 'shape'):
      dimAction = env.action_space.shape[0]
      comm.setStateActionDims(dimState, dimAction, 0) # 1 agent
      isBounded = dimAction * [True]
      comm.setActionScales(env.action_space.high, env.action_space.low, isBounded, 0)
    else: assert(False)

    return env

def app_main(comm):
  env = setupSmartiesCommon(comm)

  while True: #training loop
    observation = env.reset()
    t = 0
    comm.sendInitState(observation)
    while True: # simulation loop
      action = getAction(comm, env) #receive action from smarties
      observation, reward, done, info = env.step(action)
      t = t + 1
      if done == True and t >= env._max_episode_steps:
        comm.sendLastState(observation, reward)
      elif done == True:
        comm.sendTermState(observation, reward)
      else: comm.sendState(observation, reward)
      if done: break

if __name__ == '__main__':
  e = rl.Engine(sys.argv)
  if( e.parse() ): exit()
  e.run( app_main )

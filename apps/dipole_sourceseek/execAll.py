#!/usr/bin/env python3
##
# smarties
# Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
# Distributed under the terms of the MIT license.
##
# Created by Guido Novati (novatig@ethz.ch).
# Redesigned by Yusheng Jiao (jiaoyush@usc.edu)
# Kanso Bio-inspired Motion Lab, University of Southern California
import sys
import os
from gym.wrappers import time_limit
import numpy as np
import smarties as rl
sys.path.append('/home/yusheng/navigation_envs/dipole_new')
from sourcefind_env import *
from gym import wrappers
import time

#import pybullet_envs
#import pybulletgym
os.environ['MUJOCO_PY_FORCE_CPU'] = '1'


def getAction(comm, env):
    buf = comm.recvAction()
    if hasattr(env.action_space, 'n'):
        action = int(buf[0])
    elif hasattr(env.action_space, 'spaces'):
        action = [int(buf[0])]
        for i in range(1, comm.nActions):
            action = action + [int(buf[i])]
    elif hasattr(env.action_space, 'shape'):
        action = buf
    else:
        assert(False)
    return action

def setupSmartiesCommon(comm):
    ############## Hyperparameters ##############
    # env_name = "singleDipole-v0"
    # render = False            # render the environment in training if true
    max_timesteps = 1000         # max timesteps in one episode
    #############################################
    print(os.getcwd())
    # creating environment
    with open("../paramToUse.txt", "r") as paramFile:
        envSetting = paramFile.readline()
    env = DipoleSingleEnv(paramSource='envParam_' + envSetting)

    # set the length of an episode
    from gym.wrappers.time_limit import TimeLimit
    env = TimeLimit(env, max_episode_steps = max_timesteps)
    
    # setup MDP properties:
    # first figure out dimensionality of state
    dimState = 1
    if hasattr(env.observation_space, 'shape'):
        for i in range(len(env.observation_space.shape)):
            dimState *= env.observation_space.shape[i]
    elif hasattr(env.observation_space, 'n'):
        dimState = 1
    else:
        assert(False)

    # then figure out action dims and details
    if hasattr(env.action_space, 'spaces'):
        dimAction = len(env.action_space.spaces)
        comm.setStateActionDims(dimState, dimAction, 0)  # 1 agent
        control_options = dimAction * [0]
        for i in range(dimAction):
            control_options[i] = env.action_space.spaces[i].n
        comm.setActionOptions(control_options, 0)  # agent 0
    elif hasattr(env.action_space, 'n'):
        dimAction = 1
        comm.setStateActionDims(dimState, dimAction, 0)  # 1 agent
        comm.setActionOptions(env.action_space.n, 0)  # agent 0
    elif hasattr(env.action_space, 'shape'):
        dimAction = env.action_space.shape[0]
        comm.setStateActionDims(dimState, dimAction, 0)  # 1 agent
        isBounded = dimAction * [True]
        comm.setActionScales(env.action_space.high,
                             env.action_space.low, isBounded, 0)
    else:
        assert(False)
    return env


def train_main(comm):    
    env = setupSmartiesCommon(comm)
    while True:  # training loop
        observation = env.reset()
        t = 0
        comm.sendInitState(observation)
        while True:  # simulation loop
            action = getAction(comm, env)  # receive action from smarties
            # action = -np.sign(observation)
            observation, reward, done, info = env.step(action)
            t = t + 1
            if done and t >= env._max_episode_steps:
                comm.sendLastState(observation, reward)
            elif done:
                comm.sendTermState(observation, reward)
            else:
                comm.sendState(observation, reward)
            if done:
                break
def test_policy(comm):
    assert comm.isTraining() == False
    env = setupSmartiesCommon(comm)

    observation = env.reset()
    comm.sendInitState(observation)
    for k in range(101):
        observation  = [k/50 - 1]
        comm.sendState(observation, 0.0)
        action = getAction(comm, env)
        print(observation[0],' -> ',action[0])
    exit()
        # comm.sendInitState(observation)
        # while not comm.terminateTraining():  # simulation loop
        #     action = getAction(comm, env)  # receive action from smarties
        #     observation, reward, done, info = env.step(action)
        #     # print([f"{o:<.2f}" for o in observation],'a:', f"{action[0]:>.2f}")
        #     env.render()
        #     t = t + 1
        #     if done and t >= env.env._max_episode_steps:
        #         comm.sendLastState(observation, reward)
        #     elif done:
        #         comm.sendTermState(observation, reward)
        #     else:
        #         comm.sendState(observation, reward)
        #     if done:
        #         break
        # if comm.terminateTraining():
        #     break
def test_main(comm):
    assert comm.isTraining() == False
    env1 = setupSmartiesCommon(comm)
    while True:  # testing loop
        print('NEW EPISODE\n')
        env = wrappers.Monitor(env1, f'./TestMovies/rand{time.time()}',force = True)
        observation = env.reset()
        t = 0
        comm.sendInitState(observation)
        while not comm.terminateTraining():  # simulation loop
            action = getAction(comm, env)  # receive action from smarties
            observation, reward, done, info = env.step(action)
            # print([f"{o:<.2f}" for o in observation],'a:', f"{action[0]:>.2f}")
            env.render()
            t = t + 1
            if done and t >= env.env._max_episode_steps:
                comm.sendLastState(observation, reward)
            elif done:
                comm.sendTermState(observation, reward)
            else:
                comm.sendState(observation, reward)
            if done:
                break
        if comm.terminateTraining():
            break
def success_rate(comm):
    """ randomly initialize the environment and test the succeess rate"""
    assert comm.isTraining() == False
    totalNum = 0
    success = 0
    env = setupSmartiesCommon(comm)
    while True:  # testing episode loop
        totalNum += 1
        # print('NEW EPISODE\n')
        observation = env.reset()
        t = 0
        comm.sendInitState(observation)
        while not comm.terminateTraining():  # simulation step loop
            action = getAction(comm, env)  # receive action from smarties
            observation, reward, done, info = env.step(action)
            # print([f"{o:<.2f}" for o in observation],'a:', f"{action[0]:>.2f}")
            t = t + 1
            if done and t >= env._max_episode_steps: 
                # if time limit exceeded
                comm.sendLastState(observation, reward)
                break
            elif done:
                # if episode succeeds or fails
                if reward > 50:
                    success += 1
                comm.sendTermState(observation, reward)
                break
            else:
                # continue to next step
                comm.sendState(observation, reward)
        if comm.terminateTraining():
            print('total:', totalNum-1, 'success:', success)
            print('success rate:', success/(totalNum-1))
            break
def success_region(comm):
    """ systematically initialize the environment and test the succeess rate in different region"""
    assert comm.isTraining() == False
    totalNum = 0
    success = 0
    env = setupSmartiesCommon(comm)
    with open("../targetPos.txt", "r") as targetFile:
        x = float(targetFile.readline())
        y = float(targetFile.readline())
    with open("success_region.txt", "w") as recordFile:
            recordFile.write(f"target Position: {x:7.2f},{y:7.2f}\n")
    # thorough test loops
    for initX in np.arange(env.permittedL + 0.5, env.permittedR, 0.5):
        # for initY in np.arange(env.permittedD + 0.5, 0, 0.5):
        for initY in np.arange(0.5, env.permittedU, 0.5):
            for initTheta in np.arange(0, 2*np.pi, np.pi/4):
                totalNum += 1
                observation = env.reset(position = [initX, initY, initTheta],\
                                        target = [x, y])
                t = 0
                comm.sendInitState(observation)
                while not comm.terminateTraining():  # simulation loop
                    action = getAction(comm, env)  # receive action from smarties
                    observation, reward, done, info = env.step(action)
                    # print([f"{o:<.2f}" for o in observation],'a:', f"{action[0]:>.2f}")
                    t = t + 1
                    if done:
                        if t >= env._max_episode_steps: 
                            # if time limit exceeded
                            comm.sendLastState(observation, reward)
                        else:
                            if reward > 50:
                                success += 1
                            comm.sendTermState(observation, reward)
                        with open("success_region.txt", "a") as recordFile:
                            recordFile.write(f"init:{initX:7.2f},{initY:7.2f},{initTheta:7.2f}. reward:{reward:8.2f}\n")
                        break
                    else:
                        # continue to next step
                        comm.sendState(observation, reward)
                if comm.terminateTraining():
                    print('TESTS NOT FINISHED!!!!')
                    break
    print('total:', totalNum, 'success:', success)
    print('success rate:', success/totalNum)
    exit()
    

if __name__ == '__main__':
    print(sys.argv)
    print(os.getcwd())
    if "--runname" in sys.argv:
        temp = sys.argv.index("--runname")
        s = sys.argv[temp+1]
        del sys.argv[temp:temp+2]
        os.makedirs(s, exist_ok=True)
        os.chdir(os.path.abspath('./'+s))
    if "--setting" in sys.argv:
        temp = sys.argv.index("--setting")
        s = sys.argv[temp+1]
        del sys.argv[temp:temp+2]
        with open("paramToUse.txt", "w") as paramFile:
            paramFile.write(s)
    # print(e.run.__doc__.text)
    # exit()
    if "--TestRegion" in sys.argv and "--Target" in sys.argv:
        # sys.argv.append("--nEvalEpisodes")
        # sys.argv.append("100000000")
        temp = sys.argv.index("--Target")
        x,y = sys.argv[temp+1], sys.argv[temp+2]
        with open("targetPos.txt", "w") as targetFile:
            targetFile.write(x+"\n")
            targetFile.write(y+"\n")
        e = rl.Engine(sys.argv)
        if (e.parse()):
            exit()
        e.setNumEvaluationEpisodes(1000000)
        e.run(success_region)
    elif "--successRate" in sys.argv and "--nEvalEpisodes" in sys.argv:
        temp = sys.argv.index("--successRate")
        del sys.argv[temp]
        # exit()
        e = rl.Engine(sys.argv)
        if (e.parse()):
            exit()
        e.run(success_rate)
    elif "--nEvalEpisodes" in sys.argv:
        print("TESTING by random cases")
        e = rl.Engine(sys.argv)
        if (e.parse()):
            exit()
        e.run(test_main)
    elif "--nTrainSteps" in sys.argv:
        e = rl.Engine(sys.argv)
        if (e.parse()):
            exit()
        e.run(train_main)
    elif "--policyTest" in sys.argv:
        e = rl.Engine(sys.argv)
        if (e.parse()):
            exit()
        e.setNumEvaluationEpisodes(10)
        e.run(test_policy)
    else:
        sys.exit("More specific arguments required!")

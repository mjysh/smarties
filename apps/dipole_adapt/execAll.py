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
from turtle import position
from gym.wrappers import time_limit
import numpy as np
import smarties as rl
sys.path.append('/home/yusheng/navigation_envs/dipole_new')
import time
from gym import wrappers
from navigationAll_env import *
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
    max_timesteps = 595   # max timesteps in one episode
    #############################################
    # print(os.getcwd())
    # creating environment
    with open("../paramToUse.txt", "r") as paramFile:
        settings = paramFile.readlines()
    envSetting = settings[0].rstrip()
    print(settings)
    if len(settings) == 2:
        max_timesteps = int(settings[1])
    env = DipoleSingleEnv(paramSource='envParam_' + envSetting)

    # set the length of an episode
    from gym.wrappers.time_limit import TimeLimit
    env = TimeLimit(env, max_episode_steps=max_timesteps)

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


def success_region_trained(comm):
    """ systematically initialize the environment and test the succeess rate in training region"""
    assert comm.isTraining() == False
    totalNum = 0
    success = 0
    env = setupSmartiesCommon(comm)

    initXs = []
    initYs = []
    targetXs = []
    targetYs = []
    initThetas = []
    rewards = []
    ts = []
    
    # load pre-generated test loops
    positions = np.load('/home/yusheng/smarties/apps/dipole_adapt/positions.npy')
    targets = np.load('/home/yusheng/smarties/apps/dipole_adapt/targets.npy')
    assert positions.shape[0] == targets.shape[0]
    trajs = np.zeros((positions.shape[0], env._max_episode_steps + 1, 3))
    observations = np.zeros((positions.shape[0], env._max_episode_steps, env.observation_space.shape[0]))
    for i in range(positions.shape[0]):
        totalNum += 1

        # targets[i,0] *= -1
        # targets[i,1] *= -1
        # positions[i,0] *= -1
        # positions[i,1] *= -1
        # positions[i,2] += np.pi

        observation = env.reset(position=positions[i,:],
                                target=targets[i,:],init_time = 0)
        t = 0
        comm.sendInitState(observation)
        while not comm.terminateTraining():  # simulation loop
            # receive action from smarties
            observations[i,t,:] = observation
            trajs[i,t,:] = env.pos
            action = getAction(comm, env)
            observation, reward, done, info = env.step(action)
            t = t + 1
            if done:
                if t >= env._max_episode_steps:
                    # if time limit exceeded
                    comm.sendLastState(observation, reward)
                else:
                    if reward > 50:
                        success += 1
                    comm.sendTermState(observation, reward)
                initXs.append(positions[i,0])
                initYs.append(positions[i,1])
                targetXs.append(targets[i,0])
                targetYs.append(targets[i,1])
                initThetas.append(positions[i,2])
                rewards.append(reward)
                ts.append(t)
                trajs[i,t,:] = env.pos
                break
            else:
                # continue to next step
                comm.sendState(observation, reward)
        if comm.terminateTraining():
            print('TESTS NOT FINISHED!!!!')
            break
    mdic = {"initX": initXs,
            "initY": initYs,
            "initTheta": initThetas,
            "reward": rewards,
            "totTime": ts,
            "trajectory": trajs,
            "observation": observations,
            "targetX": targetXs,
            "targetY": targetYs
            }
    from scipy.io import savemat
    savemat(f"success_region_trained.mat", mdic, oned_as='row', do_compression=True)
    print('total:', totalNum, 'success:', success)
    print('success rate:', success/totalNum)
    exit()


def train_main(comm):
    """DO NOT CHANGE THIS PART"""
    env = setupSmartiesCommon(comm)
    while True:  # training loop
        observation = env.reset()
        t = 0
        comm.sendInitState(observation)
        while True:  # simulation loop
            action = getAction(comm, env)  # receive action from smarties
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


def test_random(comm):
    assert comm.isTraining() == False
    env1 = setupSmartiesCommon(comm)
    while True:  # testing loop
        print('NEW EPISODE\n')
        env = wrappers.Monitor(
            env1, f'./TestMovies/rand{time.time()}', force=True)
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

def test_case(comm):
    assert comm.isTraining() == False
    env1 = setupSmartiesCommon(comm)
    with open("../initialPositions.txt", "r") as posFile:
        tarX = float(posFile.readline())
        tarY = float(posFile.readline())
        initX = float(posFile.readline())
        initY = float(posFile.readline())
        initTheta = float(posFile.readline())
        initT = float(posFile.readline())
    targetPos = [tarX, tarY]
    initState = [initX, initY, initTheta]
    obs_traj = []
    act_traj = []
    state_traj = []
    time_traj = []
    totReward = 0
    env = wrappers.Monitor(env1, f'./TestMovies', force=True)
    observation = env.reset(position=initState, target=targetPos, init_time = initT)
    obs_traj.append(observation)
    state_traj.append(env.pos)
    time_traj.append(env.time)
    t_step = 0
    comm.sendInitState(observation)
    while not comm.terminateTraining():  # simulation loop
        action = getAction(comm, env)  # receive action from smarties
        observation, reward, done, info = env.step(action)
        act_traj.append(action)
        obs_traj.append(observation)
        state_traj.append(env.pos)
        time_traj.append(env.time)
        totReward += reward
        # print([f"{o:<.2f}" for o in observation],'a:', f"{action[0]:>.2f}")

        env.render()
        t_step += 1
        if done and t_step >= env.env._max_episode_steps:
            comm.sendLastState(observation, reward)
        elif done:
            comm.sendTermState(observation, reward)
        else:
            comm.sendState(observation, reward)
        if done:
            mdic = {"observations": obs_traj,
                    "actions": act_traj,
                    "states": state_traj,
                    "time": time_traj,
                    "target": [tarX, tarY],
                    "reward": totReward}
            from scipy.io import savemat
            savemat(f"trajectory{totReward:5.2f}.mat", mdic, oned_as='row', do_compression=True)
            exit()


def success_rate(comm):
    """ randomly initialize the environment and test the succeess rate"""
    assert comm.isTraining() == False
    totalNum = 0
    success = 0
    env = setupSmartiesCommon(comm)
    while True:  # testing episode loop
        totalNum += 1
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
    target = [x,y]
    initXs = []
    initYs = []
    initThetas = []
    rewards = []
    ts = []
    # with open("success_region.txt", "w") as recordFile:
    #     recordFile.write(f"target Position: {x:7.2f},{y:7.2f}\n")
    # thorough test loops
    for initX in np.arange(env.permittedL + 0.5, env.permittedR, 0.5):
        for initY in np.arange(env.permittedD + 0.5, 0, 0.5):
            # for initY in np.arange(0.5, env.permittedU, 0.5):
            for initTheta in np.arange(0, 2*np.pi, np.pi/18):
                totalNum += 1
                observation = env.reset(position=[initX, initY, initTheta],
                                        target=target)
                t = 0
                comm.sendInitState(observation)
                while not comm.terminateTraining():  # simulation loop
                    # receive action from smarties
                    action = getAction(comm, env)
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
                        initXs.append(initX)
                        initYs.append(initY)
                        initThetas.append(initTheta)
                        rewards.append(reward)
                        ts.append(t)
                        # with open("success_region.txt", "a") as recordFile:
                        #     recordFile.write(
                        #         f"init:{initX:7.2f},{initY:7.2f},{initTheta:7.2f}. reward:{reward:8.2f}. totTime:{t:7.2f}\n")
                        break
                    else:
                        # continue to next step
                        comm.sendState(observation, reward)
                if comm.terminateTraining():
                    print('TESTS NOT FINISHED!!!!')
                    break
    mdic = {"initX": initXs,
            "initY": initYs,
            "initTheta": initThetas,
            "reward": rewards,
            "totTime": ts,
            "target": target
            }
    from scipy.io import savemat
    savemat(f"success_region.mat", mdic, oned_as='row', do_compression=True)
    print('total:', totalNum, 'success:', success)
    print('success rate:', success/totalNum)
    exit()


if __name__ == '__main__':
    print(sys.argv)
    # print(os.getcwd())
    if "--runname" not in sys.argv:
        print("--runname argument required!!!")
        exit()
    if "--setting" not in sys.argv:
        print("--setting argument required!!!")
        exit()
    temp = sys.argv.index("--runname")
    s = sys.argv[temp+1]
    del sys.argv[temp:temp+2]
    os.makedirs(s, exist_ok=True)
    os.chdir(os.path.abspath('./'+s))
    temp = sys.argv.index("--setting")
    s = sys.argv[temp+1]
    del sys.argv[temp:temp+2]
    with open("paramToUse.txt", "w") as paramFile:
        paramFile.write(s)
    # print(e.run.__doc__.text)
    # exit()
    if "--MaxStepPerEpisode" in sys.argv:
        temp = sys.argv.index("--MaxStepPerEpisode")
        s = sys.argv[temp+1]
        del sys.argv[temp:temp+2]
        with open("paramToUse.txt", "a") as paramFile:
            paramFile.write('\n'+s)
    if "--TestRegion" in sys.argv and "--Target" in sys.argv:
        temp = sys.argv.index("--Target")
        x, y = sys.argv[temp+1], sys.argv[temp+2]
        with open("targetPos.txt", "w") as targetFile:
            targetFile.write(x+"\n")
            targetFile.write(y+"\n")
        e = rl.Engine(sys.argv)
        if (e.parse()):
            exit()
        e.setNumEvaluationEpisodes(1000000)
        e.run(success_region)
    elif "--TestTrainedRegion" in sys.argv:
        e = rl.Engine(sys.argv)
        if (e.parse()):
            exit()
        e.setNumEvaluationEpisodes(10000)
        e.run(success_region_trained)
    elif "--successRate" in sys.argv and "--nEvalEpisodes" in sys.argv:
        temp = sys.argv.index("--successRate")
        del sys.argv[temp]
        # exit()
        e = rl.Engine(sys.argv)
        if (e.parse()):
            exit()
        e.run(success_rate)
    elif "--Target" in sys.argv and "--initPos" in sys.argv:
        temp = sys.argv.index("--Target")
        targetx, targety = sys.argv[temp+1], sys.argv[temp+2]
        temp = sys.argv.index("--initPos")
        initx, inity, inittheta = sys.argv[temp +
                                           1], sys.argv[temp+2], sys.argv[temp+3]
        if "--initTime" in sys.argv:
            temp = sys.argv.index("--initTime")
            inittime = sys.argv[temp+1]
        else:
            inittime = '0'
        with open("initialPositions.txt", "w") as posFile:
            posFile.write(targetx+"\n")
            posFile.write(targety+"\n")
            posFile.write(initx+"\n")
            posFile.write(inity+"\n")
            posFile.write(inittheta+"\n")
            posFile.write(inittime)
        print("TESTING one particular cases")
        e = rl.Engine(sys.argv)
        if (e.parse()):
            exit()
        e.setNumEvaluationEpisodes(1)
        e.run(test_case)
    elif "--nEvalEpisodes" in sys.argv:
        print("TESTING by random cases")
        e = rl.Engine(sys.argv)
        if (e.parse()):
            exit()
        e.run(test_random)
    elif "--nTrainSteps" in sys.argv:
        e = rl.Engine(sys.argv)
        if (e.parse()):
            exit()
        e.run(train_main)
    else:
        sys.exit("Invalid arguments / More arguments required to specify the task!")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 15:07:00 2021

@author: yusheng
"""
# ------------------------------------------------------------------------
# required pacckages: NumPy, SciPy, openAI gym
# written in the framwork of gym
# ------------------------------------------------------------------------
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
# from os import path
from scipy import integrate
# from colorama import Fore, Back, Style
import CFDfunctions as cf

class DipoleSingleEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self, mode = 'CFD', dt = 0.1, alpha = 3, r = 3,mu = 0.8, flexibility = 0.5, train_offset=0.1,cfdpath='./',cfd_dumpinterval=10,cfd_framerate = 1000,cfd_inittime = 160):
        # size and parameters
        self.speed = mu
        self.bl = 1
        self.bw = 0.2
        self.dt = dt
        self.oldpos = None
        self.trail = []
        self.target = np.zeros((2,))
        self.flex = flexibility
        self.cfdpath = cfdpath
        # self.cfdinterval = cfd_dumpinterval
        self.cfd_framerate = cfd_framerate
        self.cfd_inittime = cfd_inittime
        self.region_offset = train_offset
        self.mode = mode
        if (mode == 'CFD'):
            self.permittedL = -15.5
            self.permittedR = 7.5
            self.permittedU = 5.5
            self.permittedD = -5.5
            print('begin reading CFD data')
            with np.load('/home/yusheng/newCFD/CFDdata.npz') as data:
                self.Uf = data['U']
                self.Vf = data['V']
                self.Of = data['Omega']
                self.xf = data['X']
                self.yf = data['Y']
            print('finished reading CFD data')
        elif (mode == 'reduced'):
            self.permittedL = -8
            self.permittedR = 8
            self.permittedU = 5.5
            self.permittedD = -5.5
            self.A = 0.5
            self.lam = alpha
            self.Gamma = r
            self.bgflow = -1.0
        
        # range of observtion variables
        # position, angular position of the target
        # high = np.array([np.finfo(np.float32).max, 2*np.pi])
        high = np.array([np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max])
        low = np.array([-np.finfo(np.float32).max,-np.finfo(np.float32).max,-np.finfo(np.float32).max,-np.finfo(np.float32).max])
        # create the observation space and the action space
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.double)
        self.action_space = spaces.Box(low = -1., high = 1., shape = (1,), dtype = np.double)

        self.viewer = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # self.time = 0
        terminal = False
        dt = self.dt
        self.oldpos = list(self.pos)
        # print(self.pos)
        # compute the fish nose position
        
        disToTarget_old = np.sqrt((self.pos[0]-self.target[0])**2+(self.pos[1]-self.target[1])**2)
        
        # integrate the dynamics system

        delta_gammma_normalized = action[0]*self.flex*2*self.speed
        # print(self.vel)
        # print(self.pos)
        def reach(t, y):
            return np.sqrt((y[0]-self.target[0])**2+(y[1]-self.target[1])**2)-0.2
        reach.terminal = True
        options = {'rtol':1e-4,'atol':1e-8,'max_step': 1e-2}

        sol = integrate.solve_ivp(lambda t,y: self.__firstorderdt(t,y,delta_gammma_normalized), [self.time,self.time+dt],
                            self.pos, method = 'RK45',events = None, vectorized=False,
                            dense_output = False, **options)
        self.pos = sol.y[:,-1]
        # update the time
        self.time = self.time + dt
        disToTarget_new = np.sqrt((self.pos[0]-self.target[0])**2+(self.pos[1]-self.target[1])**2)
        
        dDisToTarget = disToTarget_new - disToTarget_old
        # terminal = self.__terminal()          # termination condition
        # print(action)
        reward = 0
        reward += -dDisToTarget
        if disToTarget_new<0.1:
            reward += 200
            terminal = True
        if self.pos[0]>self.permittedR or self.pos[0]<self.permittedL or self.pos[1]<self.permittedD or self.pos[1]>self.permittedU:
            terminal = True

        return self._get_obs(), reward, terminal, {}
    def __firstorderdt(self,t,pos,delta_gammma_normalized):
        u = np.cos(self.pos[2])*self.speed
        v = np.sin(self.pos[2])*self.speed
        # angular velocity induced by strength difference
        w = delta_gammma_normalized/self.bw
        if (self.mode == 'CFD'):
            uVK,vVK,oVK = self.__VKflow((pos[0],pos[1]),t)
        elif (self.mode == 'reduced'):
            uVK,vVK,oVK = self.__VKflow_reduced((pos[0],pos[1]))
        vel = np.array([u+uVK, v+vVK, w+oVK])
        
        self.vel = vel
        # print(uVK,vVK)
        return vel
    """CFD wake"""
    def __VKflow(self, pos,t):
        posx = pos[0]-8
        posy = pos[1]
        # if t>29:
        #     print('t',t)
        uVK,vVK,oVK = cf.interp_vel_from_matrix(self.Uf,self.Vf,self.xf,self.yf,Of=self.Of,posX=posx,posY=posy,time=t,frame_rate = self.cfd_framerate,read_interval = 5)
        # u,v = cf.interp_vel(posX=posx,posY=posy,time=t+self.cfd_inittime,dump_interval = self.cfdinterval,frame_rate = self.cfd_framerate,rootpath = self.cfdpath)        
        # uVK = u
        # vVK = v
        return uVK,vVK,oVK
    """reduced order"""
    def __VKflow_reduced(self, pos):
        A = self.A
        Gamma = self.Gamma
        lam = self.lam
        z = pos[0]+1j*pos[1]
        U = Gamma/2/np.pi*np.tanh(2*np.pi*A/lam)+self.bgflow
        wVK = 1j*Gamma/2/lam*(1/np.tan(np.pi*(z + 1j*A-self.time*U)/lam) - 1/np.tan(np.pi*(z-lam/2-1j*A-self.time*U)/lam));
        uVK = np.real(wVK)
        vVK = -np.imag(wVK)
        cut = 0.6
        if uVK>cut:
            uVK = cut
        elif uVK<-cut:
            uVK = -cut
        if vVK>cut:
            vVK = cut
        elif vVK<-cut:
            vVK = -cut
        uVK += self.bgflow
        return uVK,vVK,0
    def __initialConfig(self,mode, init_num):
        # get the initial configuration of the fish
        # r = 2*np.sqrt(np.random.rand())
        # the = np.random.rand()*2*np.pi
        # center = (self.permittedR + self.permittedL)/2
        # X, Y, theta = center+np.cos(the)*r, -(2+self.region_offset)+np.sin(the)*r, np.random.rand()*2*np.pi

        X = (self.permittedR + 3*self.permittedL)/4+np.random.rand()*(self.permittedR - self.permittedL)/2
        Y = -self.region_offset + np.random.rand()*self.permittedD/2
        theta = np.random.rand()*2*np.pi
        return np.array([X, Y, theta])
    def __prescribedControl(self,time, position = None):
        # Used when manually precribing the control (disregarding the RL policy, mainly for test)
        """swimming"""
        return 0
    def reset(self, position = None, target = None):   # reset the environment setting
        #        print(Fore.RED + 'RESET FISH')
#        print(Style.RESET_ALL)

        # self.memory_targetAngle = target_angle
        
        if position is not None:
            self.pos = position
        else:
            self.pos = self.__initialConfig(1, 1)

        if target is not None:
            self.set_target(target[0],target[1])
        else:
            # r = 2*np.sqrt(np.random.rand())
            # the = np.random.rand()*2*np.pi
            # center = (self.permittedR + self.permittedL)/2
            # self.set_target(center+np.cos(the)*r,(2+self.region_offset)+np.sin(the)*r)
            tx = (self.permittedR + 3*self.permittedL)/4+np.random.rand()*(self.permittedR - self.permittedL)/2
            ty = self.region_offset + np.random.rand()*self.permittedU/2+1
            self.set_target(tx,ty)

        """"""
        # print(self.target)
        # print(self.pos)
        self.vel = np.zeros_like(self.pos)
        self.time = 0
        return self._get_obs()

    def _get_obs(self):                 # get the orientation (reltative to the targeted direction) and the shape
        target_angle = np.arctan2(self.target[1] - self.pos[1],self.target[0] - self.pos[0])
        # target_angle = np.unwrap([self.memory_targetAngle, target_angle])[1]
        
        theta = angle_normalize(target_angle-self.pos[-1])
        if (self.mode == 'CFD'):
            uVK,vVK,oVK = self.__VKflow(self.pos,self.time)
        elif (self.mode == 'reduced'):
            uVK,vVK,oVK = self.__VKflow_reduced(self.pos)
        relu = uVK*np.cos(self.pos[-1])+ vVK*np.sin(self.pos[-1])
        relv = -uVK*np.sin(self.pos[-1])+ vVK*np.cos(self.pos[-1])
        disToTarget = np.sqrt((self.pos[0]-self.target[0])**2+(self.pos[1]-self.target[1])**2)
        return np.array([theta,disToTarget,relu,relv])

    def render(self, mode='human'):
#        print(self.pos)
        from gym.envs.classic_control import rendering
        # from pyglet.gl import glRotatef, glPushMatrix

#        class Flip(rendering.Transform):
#            def __init__(self, flipx=False, flipy=False):
#                self.flipx = flipx
#                self.flipy = flipy
#            def enable(self):
#                glPushMatrix()
#                if self.flipx: glRotatef(180, 0, 1., 0)
#                if self.flipy: glRotatef(180, 1., 0, 0)

        def make_ellipse(major=10, minor=5, res=30, filled=True):
            points = []
            for i in range(res):
                ang = 2*np.pi*i / res
                points.append((np.cos(ang)*major, np.sin(ang)*minor))
            if filled:
                return rendering.FilledPolygon(points)
            else:
                return rendering.PolyLine(points, True)
        
        def draw_lasting_circle(Viewer, radius=10, res=30, filled=True, **attrs):
            geom = rendering.make_circle(radius=radius, res=res, filled=filled)
            rendering._add_attrs(geom, attrs)
            Viewer.add_geom(geom)
            return geom
        
        def draw_lasting_line(Viewer, start, end, **attrs):
            geom = rendering.Line(start, end)
            rendering._add_attrs(geom, attrs)
            Viewer.add_geom(geom)
            return geom
        
        def draw_ellipse(Viewer, major=10, minor=5, res=30, filled=True, **attrs):
            geom = make_ellipse(major=major, minor=minor, res=res, filled=filled)
            rendering._add_attrs(geom, attrs)
            Viewer.add_onetime(geom)
            return geom
        class bgimage(rendering.Image):
            def render1(self):
                l = 102
                r = 972
                b = 487
                t = 53
                # self.img.blit(-self.width/2/(r-l)*(l+r), -self.height/2/(b-t)*(self.img.height*2-b-t), width=self.width/(r-l)*self.img.width, height=self.height/(b-t)*self.img.height)
                self.img.blit(-self.width/2, -self.height/2, width=self.width, height=self.height)
        
        
        x,y,theta = self.pos
        if (self.mode == 'CFD'):
            leftbound = -16
            rightbound = 8
            lowerbound = -8
            upperbound = 8
        elif (self.mode == 'reduced'):
            leftbound = -8
            rightbound = 8
            lowerbound = -6
            upperbound = 6
        if self.viewer is None:
            scale = 50
            self.viewer = rendering.Viewer((rightbound-leftbound)*scale,(upperbound-lowerbound)*scale)
            # background = draw_lasting_circle(self.viewer,radius=100, res=10)
            # background.set_color(1.0,.8,0)
        
            # leftbound = -bound+self.target[0]/2
            # rightbound = bound+self.target[0]/2
            # lowerbound = -bound+self.target[1]/2
            # upperbound = bound+self.target[1]/2
            self.viewer.set_bounds(leftbound,rightbound,lowerbound,upperbound)
            
        
        if (self.mode == 'CFD'):
            """Load CFD images"""
            cfdimage = bgimage(cf.read_image(self.time, rootpath = self.cfdpath, dump_interval = 10, frame_rate = self.cfd_framerate),32,16)
            # cfdimage.flip = True
            self.viewer.add_onetime(cfdimage)
        elif (self.mode == 'reduced'):
            """"vortex street"""
            # vortexN = np.ceil((rightbound - leftbound)/self.lam)
            U = self.Gamma/2/np.pi*np.tanh(2*np.pi*self.A/self.lam)+self.bgflow
            phase = (U*self.time)%self.lam
            vorDownX = np.arange((phase-leftbound)%self.lam+leftbound,rightbound,self.lam)
            vortexN = len(vorDownX)
            for i in range(vortexN):
                vortexUp = self.viewer.draw_circle(radius = 0.1)
                vortexDown = self.viewer.draw_circle(radius = 0.1)
                vorUpTrans = rendering.Transform(translation=(vorDownX[i]+self.lam/2,self.A))
                vortexUp.add_attr(vorUpTrans)
                vortexUp.set_color(1,0,0)
                vorDownTrans = rendering.Transform(translation=(vorDownX[i],-self.A))
                vortexDown.add_attr(vorDownTrans)
                vortexDown.set_color(0,0,1)
        
        """draw the axes"""
#        self.viewer.draw_line((-1000., 0), (1000., 0))
#        self.viewer.draw_line((0,-1000.), (0,1000.))
        
        """target"""
        l = 0.06
        d1 = l*(np.tan(0.3*np.pi)+np.tan(0.4*np.pi))
        d2 = l/np.cos(0.3*np.pi)
        target = self.viewer.draw_polygon(v = [(d2*np.cos(np.pi*0.7),d2*np.sin(np.pi*0.7)),(d1*np.cos(np.pi*0.9),d1*np.sin(np.pi*0.9)),
                                               (d2*np.cos(np.pi*1.1),d2*np.sin(np.pi*1.1)),(d1*np.cos(np.pi*1.3),d1*np.sin(np.pi*1.3)),
                                               (d2*np.cos(np.pi*1.5),d2*np.sin(np.pi*1.5)),(d1*np.cos(np.pi*1.7),d1*np.sin(np.pi*1.7)),
                                               (d2*np.cos(np.pi*1.9),d2*np.sin(np.pi*1.9)),(d1*np.cos(np.pi*0.1),d1*np.sin(np.pi*0.1)),
                                               (d2*np.cos(np.pi*0.3),d2*np.sin(np.pi*0.3)),(d1*np.cos(np.pi*0.5),d1*np.sin(np.pi*0.5))])
        tgTrans = rendering.Transform(translation=(self.target[0], self.target[1]))
        target.add_attr(tgTrans)
        target.set_color(.0,.5,.2)
        """trail"""
        self.trail.append(self.pos[0:2])
        for i in range(len(self.trail)-1): 
            trail = self.viewer.draw_line(start=(self.trail[i][0],self.trail[i][1]), end=(self.trail[i+1][0],self.trail[i+1][1]))
            trail.linewidth.stroke = 2
            # trail = draw_lasting_circle(self.viewer, radius=0.05, res = 5)
            # trTrans = rendering.Transform(translation=(x,y))
            # trail.add_attr(trTrans)
#            trail.set_color(.6, .106, .118)
#            trail.linewidth.stroke = 10
            trail.set_color(0.2,0.2,0.2)        
        """distance line"""
        # Xnose = x + np.cos(theta)*self.bl/2
        # Ynose = y + np.sin(theta)*self.bl/2
        # self.viewer.draw_line((Xnose, Ynose), (self.target[0], self.target[1]))
        
        
        
        """fish shape"""
        for i in range(1):
            fish = draw_ellipse(self.viewer,major=self.bl/2, minor=self.bw/2, res=30, filled=False)
            fsTrans = rendering.Transform(rotation=theta,translation=(x,y))
            fish.add_attr(fsTrans)
            fish.set_linewidth(3)
            fish.set_color(.7, .3, .3)
        for i in range(2):
            eye = draw_ellipse(self.viewer,major=self.bl/10, minor=self.bl/10, res=30, filled=True)
            eyngle = theta+np.pi/5.25*(i-.5)*2;
            eyeTrans = rendering.Transform(translation=(x+np.cos(eyngle)*self.bl/4,y+np.sin(eyngle)*self.bl/4))
            eye.add_attr(eyeTrans)
            eye.set_color(.6,.3,.4)
        
       
        # from pyglet.window import mouse
        
        @self.viewer.window.event
#        def on_mouse_press(x, y, buttons, modifiers):
        def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
            leftbound = -self.viewer.transform.translation[0]/self.viewer.transform.scale[0]
            rightbound = (self.viewer.width-self.viewer.transform.translation[0])/self.viewer.transform.scale[0]
            lowerbound = -self.viewer.transform.translation[1]/self.viewer.transform.scale[1]
            upperbound = (self.viewer.height-self.viewer.transform.translation[1])/self.viewer.transform.scale[1]
            self.set_target((x)/self.viewer.width*(rightbound-leftbound) + leftbound,(y)/self.viewer.height*(upperbound-lowerbound) + lowerbound)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    def set_target(self,x,y):
        self.target[0] = x
        self.target[1] = y
        target_angle = np.arctan2(self.target[1] - self.pos[1],self.target[0] - self.pos[0])
        # self.memory_targetAngle = target_angle
        self.pos[-1] = angle_normalize(self.pos[-1], center = target_angle)
def angle_normalize(x,center = 0,half_period = np.pi):
    return (((x+half_period-center) % (2*half_period)) - half_period+center)

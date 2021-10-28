#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 17:07:24 2021

@author: yusheng
"""

import h5py
import os
import numpy as np
# import pandas as pd
from tqdm import tqdm
import re
import time as timer
import math
def adapt_interp_test(time = 0.634,posX = -0.883,posY = 0.43,
                 time_span = 1,read_interval = 1,
                 level_limit = 3,source_path = "/home/yusheng/CFDadapt/np/"):
    
    frame_rate,dx_list,init_time,time_span,UUU,VVV,OOO,XMIN,XMAX,YMIN,YMAX,l\
        = adapt_load_data(time_span,source_path,read_interval,level_limit)
    tn = int(time_span*frame_rate/read_interval)+1
    frame = time*frame_rate
    if frame>tn-1:
        frameDown = tn-1
        frameUp = frameDown
    else:
        frameUp = np.int64(np.ceil(frame/read_interval))*read_interval
        frameDown = np.int64(np.floor(frame/read_interval))*read_interval
    weightDown = (frameUp-frame)/read_interval
    weightUp = (frame-frameDown)/read_interval
    
    """velocity for frameDown"""
    UDown,VDown,ODown = adapt_space_Interp(posX,posY,UUU[frameDown],VVV[frameDown],OOO[frameDown],
                               XMIN[frameDown],XMAX[frameDown],
                               YMIN[frameDown],YMAX[frameDown],
                               dx_list[frameDown],l[frameDown])
    # #########################################################
    if frameDown != frameUp:
        """velocity for frameUp"""
        UUp,VUp,OUp = adapt_space_Interp(posX,posY,UUU[frameUp],VVV[frameUp],OOO[frameUp],
                               XMIN[frameUp],XMAX[frameUp],
                               YMIN[frameUp],YMAX[frameUp],
                               dx_list[frameUp],l[frameUp])
        # UUp, VUp = read_vel(posX, posY, frameUp)
        U_interp = UUp*weightUp+UDown*weightDown
        V_interp = VUp*weightUp+VDown*weightDown
        O_interp = OUp*weightUp+ODown*weightDown
    else:
        U_interp = UDown+0
        V_interp = VDown+0
        O_interp = ODown+0
    return U_interp,V_interp,O_interp
def adapt_time_interp(UUU,VVV,OOO,XMIN,XMAX,YMIN,YMAX,frame_rate,time = 0.634,posX = -0.883,posY = 0.43):
    tn = len(UUU)
    frame = time*frame_rate
    if frame>tn-1:
        frameDown = tn-1
        frameUp = frameDown
    else:
        frameUp = np.int64(np.ceil(frame))
        frameDown = np.int64(np.floor(frame))
    weightDown = (frameUp-frame)
    weightUp = (frame-frameDown)
    """velocity for frameDown"""
    UDown,VDown,ODown = adapt_space_Interp(posX,posY,UUU[frameDown],VVV[frameDown],OOO[frameDown],
                               XMIN[frameDown],XMAX[frameDown],
                               YMIN[frameDown],YMAX[frameDown])
    # #########################################################
    if frameDown != frameUp:
        """velocity for frameUp"""
        UUp,VUp,OUp = adapt_space_Interp(posX,posY,UUU[frameUp],VVV[frameUp],OOO[frameUp],
                               XMIN[frameUp],XMAX[frameUp],
                               YMIN[frameUp],YMAX[frameUp])
        # UUp, VUp = read_vel(posX, posY, frameUp)
        U_interp = UUp*weightUp+UDown*weightDown
        V_interp = VUp*weightUp+VDown*weightDown
        O_interp = OUp*weightUp+ODown*weightDown
    else:
        U_interp = UDown+0
        V_interp = VDown+0
        O_interp = ODown+0
    return U_interp,V_interp,O_interp
def adapt_load_data(time_span,source_path,level_limit):
    with np.load(os.path.join(source_path,"Info.npz")) as data:
        init_time = data["init_time"]
        time_span = min(data["time_span"],time_span)
        frame_rate = data["frame_rate"]
    tn = int(time_span*frame_rate)+1
    UUU = [[] for i in range(tn)]
    VVV = [[] for i in range(tn)]
    OOO = [[] for i in range(tn)]
    # IMIN = [[] for i in range(tn)]
    # IMAX = [[] for i in range(tn)]
    # JMIN = [[] for i in range(tn)]
    # JMAX = [[] for i in range(tn)]
    XMIN = [[] for i in range(tn)]
    XMAX = [[] for i in range(tn)]
    YMIN = [[] for i in range(tn)]
    YMAX = [[] for i in range(tn)]
    # l = [[] for i in range(tn)]
    tlist = []
    max_level_list = []
    num_patches_list = []
    dx_list = []
    for n in tqdm(range(tn)):
        path = os.path.join(source_path , str(n))
        with np.load(os.path.join(path,'basicInfo.npz')) as data:
            tlist.append(data["time"])
            max_level_list.append(data["maxLevel"])
            num_patches_list.append(data["num_patches"])
            dx_list.append(data["dx"])
        for level in reversed(range(min(max_level_list[-1],level_limit))):
            for patch in range(num_patches_list[-1][level]):
                filename=os.path.join(path,f"data_level_{level:02n}patch_{patch:03n}.npz")
                with np.load(filename) as data:
                    UUU[n].append(data["U"])
                    VVV[n].append(data["V"])
                    OOO[n].append(data["Omega"])
                    # IMIN[n].append(data["imin"])
                    # JMIN[n].append(data["jmin"])
                    # IMAX[n].append(data["imax"])
                    # JMAX[n].append(data["jmax"])
                    XMIN[n].append(data["xmin"]+dx_list[n][level][0]/2)
                    XMAX[n].append(data["xmax"]-dx_list[n][level][0]/2)
                    YMIN[n].append(data["ymin"]+dx_list[n][level][1]/2)
                    YMAX[n].append(data["ymax"]-dx_list[n][level][1]/2)
                    # l[n].append(level)
    return frame_rate,time_span,UUU,VVV,OOO,XMIN,XMAX,YMIN,YMAX
        # ,l
def adapt_space_Interp(posX,posY,UUU,VVV,OOO,XMIN,XMAX,YMIN,YMAX):
    for i in range(len(UUU)):
        if posX>XMIN[i] and posX<XMAX[i] and posY>YMIN[i] and posY<YMAX[i]:
            # print('patch',i)
            # print('left',XMIN[i])
            # print('right',XMAX[i])
            # print('bottom',YMIN[i])
            # print('top',YMAX[i])
            # print('level',l[i])
            Uf = UUU[i]
            Vf = VVV[i]
            Of = OOO[i]
            # dx, dy = dx_list[l[i]][0:2]
            dx = (XMAX[i]-XMIN[i])/(Uf.shape[1]-1)
            dy = (YMAX[i]-YMIN[i])/(Uf.shape[0]-1)
            indexX = math.ceil((posX-XMIN[i])/dx)
            rx = (posX-XMIN[i])%dx
            indexY = math.ceil((posY-YMIN[i])/dy)
            ry = (posY-YMIN[i])%dy
            # xf = np.linspace(XMIN[i],XMAX[i],Uf.shape[1])
            # yf = np.linspace(YMIN[i],YMAX[i],Uf.shape[0])
            break
    # indexX = np.searchsorted(xf,posX)
    # indexY = np.searchsorted(yf,posY)
    # print(indexX,indexY)
    # print(posX,posY)
    # dx = xf[1]-xf[0]
    # dy = yf[1]-yf[0]
    # xa = np.array([[xf[indexX] - posX, posX - xf[indexX-1]]])/dx
    # ya = np.array([[yf[indexY] - posY] , [posY - yf[indexY-1]]])/dy
    
    xa = np.array([[1-rx/dx, rx/dx]])
    ya = np.array([[1-ry/dy] , [ry/dy]])
    #########################################################
    """interpolate velocity"""
    QU = Uf[indexY-1:indexY+1,indexX-1:indexX+1].T
    QV = Vf[indexY-1:indexY+1,indexX-1:indexX+1].T
    QO = Of[indexY-1:indexY+1,indexX-1:indexX+1].T
    UInterp = (xa @ QU @ ya).squeeze()
    VInterp = (xa @ QV @ ya).squeeze()
    OInterp = (xa @ QO @ ya).squeeze()
    return UInterp,VInterp,OInterp
def read_image(time,rootpath,dump_interval,frame_rate):
    path = rootpath+f"Movie/movie{int(np.floor(time*frame_rate/dump_interval)):04n}.png"
    return path

def preprocess_matrix(rootpath, init_time, time_span ,dump_interval = 1,frame_rate = 1000,read_interval = 1):
    #  read velocity matrix from the coarse grid for all selected time steps
    tn = int(time_span*frame_rate/read_interval)+1
    path= rootpath +'viz_cylinder2d/visit_dump.{:05n}/'.format(0)
    name = 'summary.samrai'
    filename=os.path.join(path+name)
    f =  h5py.File(filename,'r')
    extent = f['extents']
    xn, yn = (extent['patch_extents'][0][1]-extent['patch_extents'][0][0])[0:2]+1
    left = extent['patch_extents'][0][-2][0]
    right = extent['patch_extents'][0][-1][0]
    
    up = extent['patch_extents'][0][-1][1]
    down = extent['patch_extents'][0][-2][1]
    dx, dy = f['BASIC_INFO']['dx'][0][0:2]
    centerx = np.linspace(left+dx/2,right-dx/2,xn)
    centery = np.linspace(down+dy/2,up-dy/2,yn)
    # XX, YY = np.meshgrid(centerx,centery)
    UUU = np.zeros((yn,xn,tn))
    VVV = np.zeros((yn,xn,tn))
    Omega = np.zeros((yn,xn,tn))
    initframe = int(init_time*frame_rate/dump_interval)
    for i in range(tn):
        frame = initframe + i*read_interval
        path= rootpath +'viz_cylinder2d/visit_dump.{:05n}/'.format(frame)
        name = 'processor_cluster.00000.samrai'
        filename=os.path.join(path+name)
        f =  h5py.File(filename,'r')
        UUU[:,:,i] = np.array(f['processor.00000']['level.00000']['patch.00000']['U_x']).reshape(yn,xn)
        VVV[:,:,i] = np.array(f['processor.00000']['level.00000']['patch.00000']['U_y']).reshape(yn,xn)
        Omega[:,:,i] = np.array(f['processor.00000']['level.00000']['patch.00000']['Omega']).reshape(yn,xn)
    print('loading velocity matrices seccessfully')
    return UUU,VVV,centerx,centery,Omega
def preprocess_matrix_adaptive(rootpath = "/home/yusheng/CFDadapt/",
                               targetpath = "/home/yusheng/CFDadapt/",
                               init_time = 200,
                               time_span = 1,
                               # frame_rate = 100,
                               read_interval = 1,
                               level_limit = 2):
    #  read velocity matrix from the coarse grid for all selected time steps
    # FRAME RATE HAS TO BE AN INTEGER
    def mkdir(path):
    # remove space at the beginning
        path=path.strip()
        # remove \ at the end
        path=path.rstrip("\\")
    
        if not os.path.exists(path):
            '''
            os.mkdir(path)与os.makedirs(path)的区别是,当父目录不存在的时候os.mkdir(path)不会创建，os.makedirs(path)则会创建父目录
            '''
            # use utf-8 for path
            os.makedirs(path) 
            print (path+' directory successfully created')
            return True
        else:
            print (path+' existing directory')
            return False
    rootpath="/home/yusheng/CFDadapt/"
    filename=os.path.join(rootpath+"viz_cylinder2d/dumps.visit")
    indexList=[]
    f = open(filename)               
    line = f.readline()            
    while line : 
        indexList.append(re.findall('\d+', line)[0])
        line = f.readline()
    f.close()
    
    name = "viz_cylinder2d/visit_dump."+indexList[0]+"/summary.samrai"
    filename = os.path.join(rootpath + name)
    ff0 =  h5py.File(filename,'r')
    name = "viz_cylinder2d/visit_dump."+indexList[1]+"/summary.samrai"
    filename = os.path.join(rootpath + name)
    ff1 =  h5py.File(filename,'r')
    frame_rate = round(1/(ff1['BASIC_INFO']['time'][0]-ff0['BASIC_INFO']['time'][0]))
    initframe = int(init_time*frame_rate)
    frame_rate = frame_rate/read_interval
    tn = int(time_span*frame_rate)+1
    mkdir(targetpath)
    np.savez(targetpath+"/Info", 
             init_time = init_time,
             time_span = time_span,
             frame_rate = frame_rate)
    for n in tqdm(range(tn)):
        index = indexList[initframe+n*read_interval]
        path  = os.path.join(targetpath,str(n))
        mkdir(path)
        name="viz_cylinder2d/visit_dump."+index+"/processor_cluster.00000.samrai"
        filename=os.path.join(rootpath + name)
        f1 =  h5py.File(filename,'r')
        name = "viz_cylinder2d/visit_dump."+index+"/summary.samrai"
        filename = os.path.join(rootpath + name)
        f2 =  h5py.File(filename,'r')
        time = (f2['BASIC_INFO']['time'][0])
        MAX_LEVELS = (f2['BASIC_INFO']['number_levels'][0])
        dx = np.array(f2['BASIC_INFO']['dx'])
        num_patches = np.array(f2['BASIC_INFO']['number_patches_at_level'],dtype = np.int32)
        x_min = np.array(f2['BASIC_INFO']['XLO'])
        patch_tot = 0
        nLevel = min(MAX_LEVELS,level_limit)
        filename = os.path.join(path , 'basicInfo')
        np.savez(filename,
                time = time,
                maxLevel = nLevel,
                num_patches = num_patches[:nLevel],
                dx = dx[:nLevel])
        for level in range(min(MAX_LEVELS,level_limit)):
            for patch in range(num_patches[level]):
                patch_extents = f2['extents']['patch_extents'][patch_tot]
                
                xn, yn = (patch_extents[1]-patch_extents[0])[0:2]+1
                omega = np.array(f1['processor.00000'][f'level.{level:05n}'][f'patch.{patch:05n}']['Omega']).reshape(yn,xn)
                # pressure=f1['processor.00000']['level.'+a]['patch.'+b]['P'][:]
                U = np.array(f1['processor.00000'][f'level.{level:05n}'][f'patch.{patch:05n}']['U_x']).reshape(yn,xn)
                V = np.array(f1['processor.00000'][f'level.{level:05n}'][f'patch.{patch:05n}']['U_y']).reshape(yn,xn)
                filename=os.path.join(path,f"data_level_{level:02n}patch_{patch:03n}")
                np.savez(filename,U=U,V=V,Omega=omega,
                         imin = int(patch_extents[0][0]),
                         jmin = int(patch_extents[0][1]),
                         imax = int(patch_extents[1][0]),
                         jmax = int(patch_extents[1][1]),
                         xmin = float(patch_extents[2][0]),
                         xmax = float(patch_extents[3][0]),
                         ymin = float(patch_extents[2][1]),
                         ymax = float(patch_extents[3][1]))
                patch_tot += 1
    print('Saved data seccessfully')
    return 0
def read_vel(posX, posY, frame, rootpath):
    path= rootpath +'viz_cylinder2d/visit_dump.{:05n}/'.format(frame)
    name = 'summary.samrai'
    filename=os.path.join(path+name)
    f1 =  h5py.File(filename,'r')
    Basic = f1['BASIC_INFO']
    extent = f1['extents']
        
    name = 'processor_cluster.00000.samrai'
    filename=os.path.join(path+name)
    f2 =  h5py.File(filename,'r')
    
    dx_coarse, dy_coarse = Basic['dx'][0][0:2]
    # Nx_coarse = np.zeros((2,),dtype=np.int64)
    # Ny_coarse = np.zeros((2,),dtype=np.int64)
    # Nx_coarse[0], Ny_coarse[0] = (extent['patch_extents'][0][1]-extent['patch_extents'][0][0])[0:2]
    # Nx_coarse[1], Ny_coarse[1] = (extent['patch_extents'][1][1]-extent['patch_extents'][1][0])[0:2]
    # Nx_fine, Ny_fine = (extent['patch_extents'][2][1]-extent['patch_extents'][2][0])[0:2]
    
    # if posX<extent['patch_extents'][0][-1][0]:
    #     k=0
    # else:
    #     k=1
    k = 0
    Nx_coarse, Ny_coarse = (extent['patch_extents'][k][1]-extent['patch_extents'][k][0])[0:2]+1
    posI = int((posX - extent['patch_extents'][k][-2][0] - dx_coarse/2) // dx_coarse)
    posJ = int((posY - extent['patch_extents'][k][-2][1] - dy_coarse/2) // dy_coarse)
    
    index = posJ*Nx_coarse + posI
    UU = f2['processor.00000']['level.00000']['patch.0000{}'.format(k)]['U_x']
    VV = f2['processor.00000']['level.00000']['patch.0000{}'.format(k)]['U_y']
    QU = np.array([[UU[index],UU[index+Nx_coarse]],[UU[index+1],UU[index+Nx_coarse+1]]])
    QV = np.array([[VV[index],VV[index+Nx_coarse]],[VV[index+1],VV[index+Nx_coarse+1]]])
    x1 = (posI+0.5)*dx_coarse + extent['patch_extents'][k][-2][0]
    x2 = (posI+1.5)*dx_coarse + extent['patch_extents'][k][-2][0]
    y1 = (posJ+0.5)*dy_coarse + extent['patch_extents'][k][-2][1]
    y2 = (posJ+1.5)*dy_coarse + extent['patch_extents'][k][-2][1]
    U = (np.array([[x2 - posX, posX - x1]]) @ QU @ np.array([[y2-posY],[posY-y1]])/dx_coarse/dy_coarse).squeeze()
    V = (np.array([[x2 - posX, posX - x1]]) @ QV @ np.array([[y2-posY],[posY-y1]])/dx_coarse/dy_coarse).squeeze()
    
    # Um = np.array(f2['processor.00000']['level.00000']['patch.00000']['U_x']).reshape(Ny_coarse,Nx_coarse)
    # Vm = np.array(f2['processor.00000']['level.00000']['patch.00000']['U_y']).reshape(Ny_coarse,Nx_coarse)
    
    # fig, ax = plt.subplots()
    # q = ax.quiver(Um[0:-1:8,0:-1:8], Vm[0:-1:8,0:-1:8])
    # ax.set_aspect('equal')
    # plt.show()
    return U,V 
def interp_vel(posX=0.0,posY=0.0,time=0.0,read_interval = 1,frame_rate = 1000,rootpath = './'):
    # dt = dump_interval/frame_rate
    frame = time*frame_rate
    
    frameUp = np.int64(np.ceil(frame/read_interval))*read_interval
    frameDown = np.int64(np.floor(frame/read_interval))*read_interval
    weightDown = (frameUp-frame)/read_interval
    weightUp = (frame-frameDown)/read_interval
    #########################################################
    """velocity for frameDown"""
    UDown, VDown = read_vel(posX, posY, frameDown, rootpath)
    #########################################################
    #########################################################
    if frameDown != frameUp:
        """velocity for frameUp"""
        UUp, VUp = read_vel(posX, posY, frameUp, rootpath)
        U_interp = UUp*weightUp+UDown*weightDown
        V_interp = VUp*weightUp+VDown*weightDown
    else:
        U_interp = UDown
        V_interp = VDown
    # print(U_interp,V_interp)
    
    # for i in range(len(f2['processor.00000']['level.00000']['patch.00000']['U.00'])):
    #     if (f2['processor.00000']['level.00000']['patch.00000']['U.01'][i]!= f2['processor.00000']['level.00000']['patch.00000']['U_y'][i]):
    #         print('a')
    return U_interp, V_interp
def interp_vel_from_matrix(Uf,Vf,xf,yf,Of = None,posX=0.0,posY=0.0,time=0.0,frame_rate = 1000,read_interval = 1):
    # dt = read_interval/frame_rate
    frame = time*frame_rate/read_interval
    frameUp = np.int64(np.ceil(frame))
    frameDown = np.int64(np.floor(frame))
    # if frame>6000:
    #     print(frame)
    if frame>Uf.shape[-1]-1:
        frameDown = Uf.shape[-1]-1
        frameUp = frameDown
        # print(frameUp)
        # print(frameDown)
        # print(frameUp==frameDown)
    weightDown = (frameUp-frame)
    weightUp = (frame-frameDown)
    
    indexX = np.searchsorted(xf,posX)
    indexY = np.searchsorted(yf,posY)
    # print(indexX,indexY)
    # print(posX,posY)
    dx = xf[1]-xf[0]
    dy = yf[1]-yf[0]
    xa = np.array([[xf[indexX] - posX, posX - xf[indexX-1]]])/dx
    ya = np.array([[yf[indexY] - posY] , [posY - yf[indexY-1]]])/dy
    #########################################################
    """velocity for frameDown"""
    QU = Uf[indexY-1:indexY+1,indexX-1:indexX+1,frameDown].T
    QV = Vf[indexY-1:indexY+1,indexX-1:indexX+1,frameDown].T
    UDown = (xa @ QU @ ya).squeeze()
    VDown = (xa @ QV @ ya).squeeze()
    if Of is not None:
        QO = Of[indexY-1:indexY+1,indexX-1:indexX+1,frameDown].T
        ODown = (xa @ QO @ ya).squeeze()
    # UDown, VDown = read_vel(posX, posY, frameDown)
    #########################################################
    
    #########################################################
    if frameDown != frameUp:
        """velocity for frameUp"""
        QU = Uf[indexY-1:indexY+1,indexX-1:indexX+1,frameUp].T
        QV = Vf[indexY-1:indexY+1,indexX-1:indexX+1,frameUp].T
        UUp = (xa @ QU @ ya).squeeze()
        VUp = (xa @ QV @ ya).squeeze()
        if Of is not None:
            QO = Of[indexY-1:indexY+1,indexX-1:indexX+1,frameUp].T
            OUp = (xa @ QO @ ya).squeeze()
            O_interp = OUp*weightUp + ODown*weightDown
        # UUp, VUp = read_vel(posX, posY, frameUp)
        U_interp = UUp*weightUp+UDown*weightDown
        V_interp = VUp*weightUp+VDown*weightDown
    else:
        U_interp = UDown+0
        V_interp = VDown+0
        if Of is not None:
            O_interp = ODown+0
    # print(U_interp,V_interp)
    
    # for i in range(len(f2['processor.00000']['level.00000']['patch.00000']['U.00'])):
    #     if (f2['processor.00000']['level.00000']['patch.00000']['U.01'][i]!= f2['processor.00000']['level.00000']['patch.00000']['U_y'][i]):
    #         print('a')
    if Of is not None:
        return U_interp, V_interp, O_interp/2
    else:
        return U_interp, V_interp
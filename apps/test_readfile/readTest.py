#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 15:47:00 2021

@author: yusheng
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
FILE = "./train/restarted_agent_00_net_"
ftype = np.float32
W    = np.fromfile(FILE +"weights.raw",     dtype=ftype)
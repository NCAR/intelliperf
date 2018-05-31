#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 11:04:31 2018

@author: uppala
"""
"""
Demo of table function to display a table within a plot.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


w, h = 3, 3;
Matrix = [[0 for x in range(w)] for y in range(h)] 
labelr = ['0','1','2']
labelc = ['name', 'S.No.','Count']

lightgrn = (0.5, 0.8, 0.5)
plt.table(cellText = Matrix,
          rowLabels=labelr,
          colLabels=labelc,
          rowColours=[lightgrn] * 16,
          colColours=[lightgrn] * 16,
          cellLoc='center',
          loc='upper left')
plt.axis('off')
plt.show()
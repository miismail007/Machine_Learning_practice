# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 18:22:54 2020

@author: Ismail
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle



filename = 'Regressor_model'
load_lr_model =pickle.load(open(filename, 'rb'))


yettopredict = np.array([[10.9]])
y_pred = load_lr_model.predict(yettopredict)
print(np.array(y_pred[0]))
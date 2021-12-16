# coding: utf-8
import matplotlib.pyplot as plt
import os
class MovAvg():
    def __init__(self, window_size=7):
        self.window_size = window_size
        self.data_queue = []
        self.sum=0

    def update(self, data):
        if len(self.data_queue) == self.window_size:
            self.sum=self.sum-self.data_queue[0]
            del self.data_queue[0]
        self.data_queue.append(data)
        self.sum=self.sum+data
        return self.sum/len(self.data_queue)


from pykalman import UnscentedKalmanFilter
import numpy as np
def f(state, noise):
    return state + np.sin(noise)
def g(state, noise):
    return state + np.cos(noise)

def UKFhandle(data):
    ukf = UnscentedKalmanFilter(f, g)
    data1=ukf.smooth(data)[0]
    #data1=Kalman1D(data,0.1)
    A, = plt.plot(data[20:], '-r', label='A', linewidth=5.0)
    B, = plt.plot(data1[20:], 'b-.', label='B', linewidth=5.0)
    legend = plt.legend(handles=[A, B])
    plt.show()


from pykalman import KalmanFilter
def Kalman1D(observations,damping=1):
    # To return the smoothed time series data
    observation_covariance = damping
    initial_value_guess = observations
    transition_matrix = 1
    transition_covariance = 0.1
    #initial_value_guess
    kf = KalmanFilter(
            initial_state_mean=initial_value_guess,
            initial_state_covariance=observation_covariance,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance,
            transition_matrices=transition_matrix
        )
    pred_state, state_cov = kf.smooth(observations)
    return pred_state


#UKFhandle(data[0])

from pykalman import KalmanFilter
import matplotlib.pyplot as plt

def kalmanFilter(data):
    kf = KalmanFilter(n_dim_obs = 1,
                    n_dim_state = 1,
                    initial_state_mean = -78,
                    initial_state_covariance = 20,
                    transition_matrices = [1],
                    transition_covariance = np.eye(1),
                    transition_offsets = None,
                    observation_matrices = [1],
                    observation_covariance = 10
                    )
    mean,cov = kf.filter(data)

    figsize = 9, 9
    figure, ax = plt.subplots(figsize=figsize)
    # 在同一幅图片上画两条折线
    A, = plt.plot(data[15:], '-r', label='A', linewidth=5.0)
    B, = plt.plot(mean[15:], 'b-.', label='B', linewidth=5.0)
    font1 = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 23,
            }
    legend = plt.legend(handles=[A, B], prop=font1)
    plt.show()

from scipy.signal import *
# 低通滤波器，数据预处理用
def filter_low(data):
    b, a = butter(8, 0.4, 'lowpass')  # 配置滤波器 8 表示滤波器的阶数
    afterData= filtfilt(b, a, data)
    figsize = 9, 9
    figure, ax = plt.subplots(figsize=figsize)
    # 在同一幅图片上画两条折线
    A, = plt.plot(data, '-r', label='A', linewidth=5.0)
    B, = plt.plot(afterData, 'b-.', label='B', linewidth=5.0)
    font1 = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 23,
            }
    legend = plt.legend(handles=[A, B], prop=font1)
    plt.show()
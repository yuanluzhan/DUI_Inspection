import numpy as np
from Scenario import *
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import os
import argparse

class PSO():

    def __init__(self,n_particles,max_iter,scenario):
        self.scenario = scenario
        v_high = 0.3
        self.n_particles = n_particles
        self.lb = 0
        self.ub = 1
        self.V = np.random.uniform(low=-v_high, high=v_high, size=(self.n_particles, self.scenario.roads))
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.n_particles, self.scenario.roads))
        for i in range(self.X.shape[0]):
            self.X[i,:] = self.X[i,:]* self.scenario.num_polices/np.sum(self.X[i,:])


        self.pbest_x = self.X.copy()  # personal best location of every particle in history
        self.pbest_y = np.array([[np.inf]] * self.n_particles)  # best image of every particle in history
        self.gbest_x = self.pbest_x.mean(axis=0).reshape(1, -1)  # global best location for all particles
        self.gbest_y = np.inf  # global best y for all particles
        self.gbest_y_hist = []  # gbest_y of every iteration

        self.w = 0.8  # inertia
        self.cp, self.cg = 0.5, 0.5
        # parameters to control personal best, global best respectively
        self.n_dim = self.scenario.roads  # dimension of particles, which is the number of variables of func
        self.max_iter = max_iter  # max iter
        self.record_mode = False


    def update_V(self):
        r1 = np.random.rand(self.n_particles, self.scenario.roads)
        r2 = np.random.rand(self.n_particles, self.scenario.roads)
        self.V = self.w * self.V + \
                 self.cp * r1 * (self.pbest_x - self.X) + \
                 self.cg * r2 * (self.gbest_x - self.X)

    def update_X(self):
        # print(np.sum(self.V))
        self.X = self.X + self.V
        # self.X = np.clip(self.X, 0, 1)
        # X: (n_roads, n_粒子）
        self.X = np.clip(self.X, 0, 1)
        for i in range(self.X.shape[0]):
            self.X[i, :] = self.X[i, :]*self.scenario.num_polices / (np.sum(self.X[i, :]))

    def cal_y(self):
        # calculate y for every x in X
        # self.Y = self.get_payoff(self.X).reshape(-1, 1)
        tmp = []
        for i in range(self.X.shape[0]):
            tmp.append([self.scenario.get_payoff(self.X[i,:])])
        self.Y = np.array(tmp)

        return self.Y

    def update_pbest(self):
        '''
        personal best
        :return:
        '''
        self.need_update = self.pbest_y > self.Y
        # self.need_update = self.need_update[1]
        # for idx, x in enumerate(self.X):
        #     if self.need_update[idx]:
        #         self.need_update[idx] = self.check_constraint(x)

        self.pbest_y = np.where(self.need_update, self.Y, self.pbest_y)
        self.pbest_x = np.where(self.need_update, self.X, self.pbest_x)

    def update_gbest(self):
        '''
        global best
        :return:
        '''
        idx_min = self.pbest_y.argmin()
        if self.gbest_y > self.pbest_y[idx_min]:
            self.gbest_x = self.X[idx_min, :].copy()
            self.gbest_y = self.pbest_y[idx_min]

    def recorder(self):
        if not self.record_mode:
            return
        self.record_value['X'].append(self.X)
        self.record_value['V'].append(self.V)
        self.record_value['Y'].append(self.Y)

    def run(self, precision=0.01):
        '''
        precision: None or float
            If precision is None, it will run the number of max_iter steps
            If precision is a float, the loop will stop if continuous N difference between pbest less than precision
        N: int
        '''
        c = 0
        for iter_num in range(self.max_iter):
            self.update_V()
            self.recorder()
            self.update_X()
            self.cal_y()
            self.update_pbest()
            self.update_gbest()
            if precision is not None:
                tor_iter = np.amax(self.pbest_y) - np.amin(self.pbest_y)
                if tor_iter < precision:
                    c = c + 1
                    if c > self.max_iter:
                        break
                else:
                    c = 0
            # if self.verbose:
            print('Iter: {}, Best fit: {} at {}'.format(iter_num, self.gbest_y, self.gbest_x))
            self.gbest_y_hist.append(self.gbest_y)
        self.best_x, self.best_y = self.gbest_x, self.gbest_y

        return self.best_x, self.best_y




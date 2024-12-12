import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from heapq import heappush, heappop
from itertools import count
import networkx as nx
from networkx.algorithms.shortest_paths.generic import _build_paths_from_predecessors
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import os
import argparse

class scenario():
    """"
    The DUI Driver scenario

    :parameter
    N : the number of roads
    M : the number of drivers
    D : the number of DUI drivers
    f : the penalty for being caught, the DUI driver's penalty is negative and law-abiding driver's penalty is zero
    g : the reward for escaping check, the DUI driver's reward is positive and law-abiding driver's penalty is zero
    t : the importance of each road

    """
    def __init__(self,information_strength,num_friends,num_polices,num_drivers, num_dui_drivers,known_strategy,map_rows,map_cols):
        # 初始化司机个数
        self.num_drivers = num_drivers
        self.num_DUI_drivers = num_dui_drivers
        self.drunk_index = random.sample(range(0, self.num_drivers), self.num_DUI_drivers)

        # 初始化司机的payoff
        self.reward_escaping = np.zeros(self.num_drivers)
        self.reward_escaping[self.drunk_index] = 1
        self.penalty_of_caught = np.zeros(self.num_drivers)
        self.penalty_of_caught[self.drunk_index] = -1
        self.information_strength = information_strength

        # 初始化警察个数
        self.num_polices = num_polices

        # The initial of the map
        self.map_rows = map_rows
        self.map_cols = map_cols
        self.map = self.mapinit(map_rows,map_cols)
        # self.map = self.mapinit_chiping()
        # self.roads = int(2 * self.map_rows * self.map_cols - self.map_rows - self.map_cols)

        # 知识共享机制初始化
        self.friends_para = num_friends
        self.num_friends = np.random.randint(0, num_friends, size=self.num_drivers)
        self.known_strategy = known_strategy


        self.record_mode = False
        self.record_path = '_f_'+str(num_friends) + '_i_' + str(self.information_strength) + '_p_' + str(num_polices) + '_dui_' +str(num_dui_drivers)

    def mapinit(self,m,n,del_node_list = [], del_edge_list = [],source_regions = []):
        "m: rows,   n: con"
        # TODO  地图的进一步简化
        a = nx.Graph()
        self.map_rows = m
        self.map_cols = n

        labels = {}
        for i in range(int(m*n)):
            a.add_node(i)
            labels[i] = str(i)
        pos = {}
        for i in range(n):
            for j in range(m):
                # print(i*n+j)
                pos[i*m+j]=np.array([-1+2*i/n,1-2*j/m])
        for i in range(n):
            for j in range(m-1):
                a.add_edge(int(i*m+j),int(i*m+1+j))
        for i in range(n-1):
            for j in range(m):
                a.add_edge(int(i*m+j),int(i*m+j+m))
        self.roads = len(a.edges)
        self.map_index = np.zeros((self.roads, self.roads))
        i = 0
        if len(del_node_list)>=1:
            for i in del_node_list:
                a.remove_node(i)
                labels.pop(i)
        if len(del_edge_list)>=1:
            for item in del_edge_list:
                a.remove_edge(item[0],item[1])
        self.roads = len(a.edges)
        for _ in a.edges:
            self.map_index[_[0],_[1]] = i
            self.map_index[_[1], _[0]] = i
            i = i+1
        self.pos = pos
        if len(source_regions)>0:
            self.source_regions = source_regions
        else:
            self.source_regions = a.nodes

        return a

    def belief(self,u,v,index):
        # return the brief probability
        k = int(self.map_index[u,v])
        return self.prior_knowledge[index, k]

    def best_path(
            self, sources,index,target, pred=None, paths =None
    ):
        """"
        find the  path with max(1-p1)(1-p2)(1-p3)
        """
        G = self.map
        G_succ = G._succ if G.is_directed() else G._adj
        paths = {source: [source] for source in sources}
        push = heappush
        pop = heappop
        dist = {}  # dictionary of final distances
        seen = {}
        # fringe is heapq with 3-tuples (distance,c,node)
        # use the count c to avoid comparing nodes (may not be able to)
        c = count()
        fringe = []
        # push(fringe, (1, next(c), source))

        for source in sources:
            if source not in G:
                raise nx.NodeNotFound(f"Source {source} not in G")
            seen[source] = 0
            push(fringe, (0, next(c), source))
        while fringe:
            (d, _, v) = pop(fringe)
            if v in dist:
                continue  # already searched this node.
            dist[v] = d
            if v == target:
                break

            for u, e in G_succ[v].items():
                cost = self.belief(u,v,index)
                # print(cost)
                if cost is None:
                    continue
                vu_dist = dist[v] + cost  -dist[v]*cost#1-(1-dist[v])*(1-cost)


                if u in dist:
                    u_dist = dist[u]
                    if vu_dist < u_dist:
                        raise ValueError("Contradictory paths found:", "negative weights?")
                    elif pred is not None and vu_dist == u_dist:
                        pred[u].append(v)
                elif u not in seen or vu_dist < seen[u]:
                    seen[u] = vu_dist
                    push(fringe, (vu_dist, next(c), u))
                    if paths is not None:
                        paths[u] = paths[v] + [u]
                    if pred is not None:
                        pred[u] = [v]
                elif vu_dist == seen[u]:
                    if pred is not None:
                        pred[u].append(v)

        return dist,paths

    def get_payoff(self,p):
        # p: the distribution of the polices
        # calculate the payoff of the defender

        pay_attackers = []
        pay_defenders = []
        p_caughts = []
        dists = []
        # save_paths = []

        i = 0
        for _ in self.map.edges:
            self.map.edges[_]['p'] = p[i]
            i = i + 1

        # 初始化每个司机
        # 先验知识
        self.prior_knowledge = np.ones((self.num_drivers, self.roads))
        if self.known_strategy == 1:
            for i in range(self.prior_knowledge.shape[0]):
                self.prior_knowledge[i,:] = p
        else:
            for i in range(self.prior_knowledge.shape[0]):
                self.prior_knowledge[i, :] = self.prior_knowledge[i, :] * self.num_polices / np.sum(
                    self.prior_knowledge[i, :])
        # 目的地和终点
        source_region = self.source_regions # 茌平的出发地
        map_nodes = list(self.map.nodes)

        source_region = map_nodes
        source_list = random.choices(source_region, k=self.num_drivers)

        target_list = []
        for i in range(self.num_drivers):
            source_tmp = source_list[i]
            target_tmp = random.sample(map_nodes, 1)
            while target_tmp == source_tmp:
                target_tmp = random.sample(map_nodes, 1)
            target_list.append(target_tmp[0])


        # Batch操作，每个batch10个or100个driver
        for i in range(self.num_drivers):
            # 找最短路径
            source = source_list[i]
            target = target_list[i]
            dist,paths = self.best_path(sources=[source], target=[target], index=i)
            dists.append(dist)
            p_caught = 0
            for k in range(len(paths[target])-1):
                p1 = self.map.edges[(paths[target][k],paths[target][k+1])]['p']
                p_caught = p_caught + p1 - p1*p_caught

            # 知识更新
            driver_index0 = np.random.randint(self.num_drivers, size=self.num_friends[i])
            driver_index = [x for x in driver_index0 if x>i]
            path_tmp = paths[target]
            road_num = list(range(0, self.prior_knowledge.shape[1], 1))
            road_index = []
            for tmp_index in range(len(path_tmp)-1):
                path_index = self.map_index[path_tmp[tmp_index],path_tmp[tmp_index+1]]
                road_index.append(int(path_index))
            tmp_knowledge = self.prior_knowledge[driver_index]
            gap = p - self.prior_knowledge[driver_index]
            road_zero = [x for x in road_num if x not in road_index]
            gap[:,road_zero]=0
            for index in range(tmp_knowledge.shape[0]):
                x = np.sum(tmp_knowledge[index][road_index])
                ax = np.sum(gap[index][road_index])+x
                b = (1-ax)/(1-x)
                tmp_knowledge[index][road_index]= gap[index][road_index]+tmp_knowledge[index][road_index]
                tmp_knowledge[index][road_zero] = tmp_knowledge[index][road_zero]*b
            self.prior_knowledge[driver_index] = self.prior_knowledge[driver_index] + gap*self.information_strength#random.uniform(0.001,0.1)
            pay_attacker = (1 - p_caught) * self.reward_escaping[i] + p_caught * self.penalty_of_caught[
                i]  # + self.num_friends[i]/100
            pay_defender = -p_caught * self.penalty_of_caught[i]

            pay_attackers.append(pay_attacker)
            p_caughts.append(p_caught)
            pay_defenders.append(pay_defender)
        self.pay_defenders = pay_defenders
        self.pay_attackers = pay_attackers
        self.p_caughts = p_caughts
        # self.save_paths = save_paths
        self.p = p

        return -np.sum(pay_defenders, axis=0)






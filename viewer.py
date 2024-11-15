import streamlit as st
import pandas as pd
import numpy as np
import os
import argparse
from heapq import heappush, heappop
from itertools import count
import matplotlib.pyplot as plt
import networkx as nx
import ast

# 设置页面标题
st.title('酒驾检查点推荐系统')
# 创建一个侧边栏
st.sidebar.header('地图信息初始化')

# 在侧边栏中添加滑动条用于选择数据数量
num_polices = st.sidebar.slider('检查点数量', min_value=1, max_value=10, value=3, step=1)
num_drivers = st.sidebar.slider('司机数量', min_value=100, max_value=10000, value=500, step=100)
num_duidrivers = st.sidebar.slider('酒驾司机数', min_value=10, max_value=1000, value=50, step=10)
n_rows = st.sidebar.slider('行数', min_value=3, max_value=20, value=5, step=1)
n_columns = st.sidebar.slider('列数', min_value=3, max_value=20, value=5, step=1)
knowledge_strength = st.sidebar.slider('知识传播强度', min_value=0, max_value=10, value=3, step=1)

# 输入酒驾司机出发的节点位置
source_region = st.sidebar.text_input("输入酒驾司机经常出发的节点位置", value="1")
del_node_list = st.sidebar.text_input("输入应该删除的节点编号", value="")
del_edge_list = st.sidebar.text_input("输入应该删除的边的节点对", value="")
iter_times = st.sidebar.slider('迭代次数', min_value=0, max_value=50, value=2, step=1)
# 创建一个场景对象
dui_object = scenario(knowledge_strength, 30, num_polices, num_drivers, num_duidrivers, 1, n_rows, n_columns)
# 计算并显示抽象地图
if len(del_node_list)>0:
    del_node_list = [int(x.strip()) for x in del_node_list.split(',')]
    
if len(del_edge_list)>0:
    split_res = ast.literal_eval(f"[{del_edge_list}]")
    del_edge_list = [tuple(item) if isinstance(item, tuple) else item for item in split_res[0]]
abstract_map = dui_object.mapinit(n_rows, n_columns,del_node_list,del_edge_list)


# 画出地图
fig, ax = plt.subplots(figsize=(10, 10))
array = [int(x.strip()) for x in source_region.split(',')]  # 将输入的源区域转换为整数列表

# 绘制地图和源区域
nx.draw(abstract_map, pos=dui_object.pos, ax=ax)  # 绘制整体地图
nx.draw(abstract_map, pos=dui_object.pos, ax=ax, node_size=500, node_color='skyblue', with_labels=True)
nx.draw_networkx_labels(abstract_map, pos=dui_object.pos, ax=ax, font_size=10, font_color='black') # 在图中显示节点编号
nx.draw(abstract_map, pos=dui_object.pos, nodelist=array, node_size=1000,node_color='r', ax=ax)  # 标记源区域（红色节点）






# 显示地图抽象图
st.write("地图抽象")
st.pyplot(fig)

# 设置大字体显示文字


# 可选：显示预期检查到的酒驾司机数
y,x = dui_object.run(max_iter=iter_times)

sorted_index = np.argsort(y)
st.write("预期检查到酒驾司机数", int(-x))
reco_edges = []
target_list = sorted_index[:num_polices+1]
for target in target_list:
    found = 0
    for i in range(len(dui_object.map_index)):
        for j in range(len(dui_object.map_index[0])):
            if dui_object.map_index[i][j] == target:
                reco_edges.append((i,j))
                found = 1
            if found == 1:
                break
        if found == 1:
            break
# 可选：显示推荐检查点设置
figr, axr = plt.subplots(figsize=(10, 10))

nx.draw(abstract_map, pos=dui_object.pos, ax=axr)  # 绘制整体地图
nx.draw(abstract_map, pos=dui_object.pos, ax=axr, node_size=500, node_color='skyblue', with_labels=True)
nx.draw_networkx_labels(abstract_map, pos=dui_object.pos, ax=axr, font_size=10, font_color='black') # 在图中显示节点编号
nx.draw(abstract_map, pos=dui_object.pos, nodelist=array, node_size=1000,node_color='r', ax=axr) 
nx.draw(abstract_map, pos=dui_object.pos, edgelist=reco_edges, width=10, edge_color='r', nodelist=array, node_color='r', ax=axr)

# nx.draw(abstract_map, pos=dui_object.pos, ax=axr)
# nx.draw(abstract_map, pos=dui_object.pos, edgelist=[(0,1)], width=10, edge_color='r', nodelist=array, node_color='r', ax=axr)
st.write("推荐检查点设置")
st.pyplot(figr)




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
        self.data_folder = 'data'+str(known_strategy)+'/'
        if not os.path.exists(self.data_folder):
            os.mkdir(self.data_folder)
        v_high = 0.3
        # init drivers
        self.num_drivers = num_drivers
        self.num_DUI_drivers = num_dui_drivers
        self.num_polices = num_polices
        self.drunk_index = random.sample(range(0, self.num_drivers), self.num_DUI_drivers)
        self.reward_escaping = np.zeros(self.num_drivers)
        self.reward_escaping[self.drunk_index] = 1
        self.penalty_of_caught = np.zeros(self.num_drivers)
        self.penalty_of_caught[self.drunk_index] = -1
        self.information_strength = information_strength

        # The initial of the map
        self.map_rows = map_rows
        self.map_cols = map_cols
        self.friends_para = num_friends
        self.num_friends = np.random.randint(0,num_friends, size=self.num_drivers)
        self.known_strategy = known_strategy
        self.map = self.mapinit(map_rows,map_cols)
        # self.map = self.mapinit_chiping()
        # self.roads = int(2 * self.map_rows * self.map_cols - self.map_rows - self.map_cols)

        self.n_particles = 10
        self.lb = 0
        self.ub = 1
        self.V = np.random.uniform(low=-v_high, high=v_high, size=(self.n_particles, self.roads))
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.n_particles, self.roads))
        for i in range(self.X.shape[0]):
            self.X[i,:] = self.X[i,:]* self.num_polices/np.sum(self.X[i,:])



        self.pbest_x = self.X.copy()  # personal best location of every particle in history
        self.pbest_y = np.array([[np.inf]] * self.n_particles)  # best image of every particle in history
        self.gbest_x = self.pbest_x.mean(axis=0).reshape(1, -1)  # global best location for all particles
        self.gbest_y = np.inf  # global best y for all particles
        self.gbest_y_hist = []  # gbest_y of every iteration

        self.w = 0.8  # inertia
        self.cp, self.cg = 0.5, 0.5
        # parameters to control personal best, global best respectively
        self.n_dim = self.roads  # dimension of particles, which is the number of variables of func
        self.max_iter = 10  # max iter
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

    def update_V(self):
        r1 = np.random.rand(self.n_particles, self.roads)
        r2 = np.random.rand(self.n_particles, self.roads)
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
            self.X[i, :] = self.X[i, :]*self.num_polices / (np.sum(self.X[i, :]))

    def cal_y(self):
        # calculate y for every x in X
        # self.Y = self.get_payoff(self.X).reshape(-1, 1)
        tmp = []
        for i in range(self.X.shape[0]):
            tmp.append([self.get_payoff(self.X[i,:])])
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

    def run(self, max_iter=None, precision=0.01):
        '''
        precision: None or float
            If precision is None, it will run the number of max_iter steps
            If precision is a float, the loop will stop if continuous N difference between pbest less than precision
        N: int
        '''
        self.max_iter = max_iter
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

    def save(self):
        # it is related to the distribution of the drunk drivers.
        # tmp = self.p

        anyone = self.get_payoff(self.p)
        strategy = np.array(self.p)
        pay_defenders = np.array(self.pay_defenders)
        prior_konwledge = np.array(self.prior_knowledge)
        pay_attackers = np.array(self.pay_attackers)
        data_folder = self.data_folder



        np.save(data_folder+'p_caught'+self.record_path,np.array(self.p_caughts))
        np.save(data_folder+'pay_defender'+self.record_path,pay_defenders)
        np.save(data_folder+'strategy'+self.record_path, strategy)
        np.save(data_folder+'prior_konwledge'+self.record_path,prior_konwledge)
        np.save(data_folder+'pay_attacker'+self.record_path, pay_attackers)
        # np.save(data_folder+'paths'+self.record_path,self.save_paths)


    def test_diff(self,p):
        anyone = self.get_payoff(p)
        pay_defenders = np.array(self.pay_defenders)
        prior_konwledge = np.array(self.prior_knowledge)
        pay_attackers = np.array(self.pay_attackers)
        strategy = np.array(self.p)
        print(np.sum(pay_defenders))


    def test2(self):
        p_caught = np.load('data/'+'p_caught'+self.record_path+'.npy')
        pay_defender = np.load('data/'+'pay_defender'+self.record_path+'.npy')
        strategy =  np.load('data/'+'strategy' +self.record_path+'.npy')
        prior_knowledge = np.load('data/'+'prior_konwledge'+self.record_path+'.npy')
        pay_attacker =  np.load('data/'+'pay_attacker'+self.record_path+'.npy')
        gap = []
        payoff = []
        p = []
        attacker = []

        print(prior_knowledge.shape)
        for i in range(p_caught.shape[0]):
            # print(strategy)
            # print(prior_knowledge.shape)
            # print(prior_knowledge[i])
            tmp = np.sum(abs(strategy-prior_knowledge[i]))
            if pay_defender[i]!=0:
                gap.append(tmp)
                payoff.append(pay_defender[i])
                attacker.append(pay_attacker[i])
                p.append(p_caught[i])
        x = np.arange(0,len(gap))
        # plt.plot(x,gap)
        #
        # plt.show()
        trace1 = go.Scatter(x=x,y=gap,mode='lines',name='gap')
        trace2 = go.Scatter(x=x,y=payoff,mode='lines',name='pay_defender')
        trace3 = go.Scatter(x=x,y=p,mode='lines',name='p')
        fig = make_subplots(rows=3,  # 将画布分为两行
                            cols=1,  # 将画布分为两列
                            subplot_titles=["gap"+str(self.information_strength),
                                            "pay_defender",
                                            "p"
                                            ],
                            # 子图的标题
                            x_title="x轴标题",
                            y_title="y轴标题"
                            )
        fig.add_trace(trace1,1,1)
        fig.add_trace(trace2,2,1)
        fig.add_trace(trace3, 3, 1)
        #pio.show(fig)

        pio.write_image(fig,'figures/'+self.record_path+'.png',format='png')
import numpy as np

from Scenario import scenario
from Optimization import PSO

# 简单环境中
information_strength =0.3
num_friends = 30
num_polices = 2
num_drivers = 1000
num_dui_drivers = 100
known_strategy = 0
map_rows = 3
map_cols = 3
easy_env = scenario(information_strength, num_friends, num_polices, num_drivers,num_dui_drivers, known_strategy, map_rows, map_cols)

# 需要的是不同策略的验证结果

# 我们的策略
train_1 = PSO(n_particles=2,max_iter=3,scenario=easy_env)
train_1.run()
# 保存在当前环境下的策略

# random策略
random_strategy = np.random()


# Stackelberg策略，这个也要训练一下


# 当前策略
common_strategy = np.random()
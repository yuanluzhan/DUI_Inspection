import numpy as np

from Scenario import scenario
from Optimization import PSO

# 简单环境中
information_strength =0.3
num_friends = 30
num_polices = 3
num_drivers = 10000
num_dui_drivers = 1000
known_strategy = 0
map_rows = 3
map_cols = 3
# test_env = scenario(information_strength = information_strength,
#                     num_friends = num_friends,
#                     num_polices = num_polices,
#                     num_drivers = num_drivers,
#                     num_dui_drivers = num_dui_drivers,
#                     known_strategy = known_strategy,
#                     map_rows = map_rows,
#                     map_cols = map_cols,
#                     env_para="chiping")

# 需要的是不同策略的验证结果

# 我们的策略

def train_and_save(information_strength, num_friends):
    train_env = scenario(information_strength=information_strength,
                         num_friends=num_friends,
                         num_polices=num_polices,
                         num_drivers=num_drivers,
                         num_dui_drivers=num_dui_drivers,
                         known_strategy=0,
                         map_rows=map_rows,
                         map_cols=map_cols,
                         env_para="simple")
    print(information_strength, num_friends)
    train_1 = PSO(n_particles=50, max_iter=30, scenario=train_env)
    strategy, result = train_1.run()

    np.save("strategies/OPRADI_" + str(information_strength) + '_' + str(num_friends) + ".npy", strategy)
    train_env = scenario(information_strength=information_strength,
                         num_friends=num_friends,
                         num_polices=num_polices,
                         num_drivers=num_drivers,
                         num_dui_drivers=num_dui_drivers,
                         known_strategy=1,
                         map_rows=map_rows,
                         map_cols=map_cols,
                         env_para="simple")
    print(information_strength, num_friends)
    train_1 = PSO(n_particles=50, max_iter=30, scenario=train_env)
    strategy, result = train_1.run()
    np.save("strategies/Stackelberg_" + str(information_strength) + '_' + str(num_friends) + ".npy", strategy)
# for information_strength in [0.4]:
#     for num_friends in [30]:
#         train_and_save(information_strength, num_friends)
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor() as executor:
    for information_strength in [0.2,0.4,0.6,0.8,1.0]:
        for num_friends in [10,30,50,100]:
            executor.submit(train_and_save, information_strength, num_friends)
# for information_strength in [0.2,0.4,0.6,0.8,1.0]:
#     for num_friends in [10,30,50,100]:
#         train_env = scenario(information_strength = information_strength,
#                             num_friends = num_friends,
#                             num_polices = num_polices,
#                             num_drivers = num_drivers,
#                             num_dui_drivers = num_dui_drivers,
#                             known_strategy = known_strategy,
#                             map_rows = map_rows,
#                             map_cols = map_cols)
#         train_1 = PSO(n_particles=200,max_iter=20,scenario=train_env)
#         strategy, results = train_1.run()
#         print(information_strength,num_friends)
#         np.save("strategies/OPRADI_"+str(information_strength)+'_'+str(num_friends)+".npy", strategy)
# # 保存策略




# 保存在当前环境下的策略

# random策略
# random_strategy = np.random(100,100)


# Stackelberg策略，这个也要训练一下


# 当前策略
# common_strategy = np.random(100,100)
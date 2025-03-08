import numpy as np

from Scenario import scenario

import os

def test_save(information_strength, num_friends):
    # 测试环境的参数
    information_strength = information_strength
    num_friends = num_friends
    num_polices = 3
    num_drivers = 10000
    num_dui_drivers = 1000
    known_strategy = 0
    map_rows = 3
    map_cols = 3
    test_env = scenario(information_strength=information_strength,
                        num_friends=num_friends,
                        num_polices=num_polices,
                        num_drivers=num_drivers,
                        num_dui_drivers=num_dui_drivers,
                        known_strategy=known_strategy,
                        map_rows=map_rows,
                        map_cols=map_cols,
                        env_para="simple")
    # 读取策略
    num_friends = num_friends
    information_strength = information_strength
    OPRADI_path = "strategies/OPRADI_" + str(information_strength) + "_" + str(num_friends) + ".npy"
    Stackelberg_path = "strategies/Stackelberg_" + str(information_strength) + "_" + str(num_friends) + ".npy"
    OPRADI_startegy = np.load(OPRADI_path)
    print(OPRADI_startegy.sum())
    Stackelberg_startegy = np.load(Stackelberg_path)
    indices = np.random.choice(test_env.roads, 3, replace=False)


    # 存储要测试的信息
    pay_defenders_OPRADI = []
    pay_defenders_Stackelberg = []
    pay_defenders_commmon = []
    pay_defenders_random = []
    # 测试
    for i in range(100):
        test_env.get_payoff(OPRADI_startegy)
        pay_defenders_OPRADI.append(test_env.pay_defenders)
        test_env.get_payoff(Stackelberg_startegy)
        pay_defenders_Stackelberg.append(test_env.pay_defenders)


        random_strategy = np.random.rand(test_env.roads)
        random_strategy = num_polices*random_strategy / np.sum(random_strategy)
        test_env.get_payoff(random_strategy)
        pay_defenders_random.append(test_env.pay_defenders)

        commmon_starategy = np.zeros(test_env.roads)
        commmon_starategy[indices] = 1
        test_env.get_payoff(commmon_starategy)
        pay_defenders_commmon.append(test_env.pay_defenders)

    results_folder = "results/" + str(information_strength)+"_" + str(num_friends)+"/"
    if not os.path.exists(results_folder):
        # 如果不存在，则创建results文件夹
        os.mkdir(results_folder)
    np.save(results_folder+"pay_defenders_OPRADI.npy", pay_defenders_OPRADI)
    np.save(results_folder+"pay_defenders_Stackelberg.npy", pay_defenders_Stackelberg)
    np.save(results_folder+"pay_defenders_common.npy", pay_defenders_commmon)
    np.save(results_folder+"pay_defenders_random.npy", pay_defenders_random)


for information_strength in [0.2,0.4,0.6,0.8,1.0]:
    for num_friends in [10,30,50,100]:
        print(information_strength,num_friends)
        test_save(information_strength,num_friends)
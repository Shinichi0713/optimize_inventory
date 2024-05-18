# Author: yoshinaga
import numpy as np
import matplotlib.pyplot as plt
import torch

import agent_dqn
import environment
import default_setting      # set for running environment


# create artifical dataset
def create_dataset():
    demand_hist = []
    np.random.seed(0)
    for i in range(52):
        # 月～木
        for j in range(4):
            random_demand = np.random.normal(3, 1.5)
            if random_demand < 0:
                random_demand = 0
            random_demand = np.round(random_demand)
            demand_hist.append(random_demand)
        # 金
        random_demand = np.random.normal(6, 1)
        if random_demand < 0:
            random_demand = 0
        random_demand = np.round(random_demand)
        demand_hist.append(random_demand)
        # 土, 日
        for j in range(2):
            random_demand = np.random.normal(12, 2)
            if random_demand < 0:
                random_demand = 0
            random_demand = np.round(random_demand)
            demand_hist.append(random_demand)
    # plt.hist(demand_hist)
    # plt.show()
    return demand_hist


# create user demand dataset
def create_user_demand_dataset():
    # preparation of test dataset
    demand_test = []
    num_sample = 40
    for k in range(100, 100 + num_sample):
        np.random.seed(k)
        demand_future = []
        for i in range(52):
            for j in range(4):
                random_demand = np.random.normal(3, 1.5)
                if random_demand < 0:
                    random_demand = 0
                random_demand = np.round(random_demand)
                demand_future.append(random_demand)
            random_demand = np.random.normal(6, 1)
            if random_demand < 0:
                random_demand = 0
            random_demand = np.round(random_demand)
            demand_future.append(random_demand)
            for j in range(2):
                random_demand = np.random.normal(12, 2)
                if random_demand < 0:
                    random_demand = 0
                random_demand = np.round(random_demand)
                demand_future.append(random_demand)
        demand_test.append(demand_future)
    return demand_test


# train_dqn model
def train_agent_dqn():
    demand_list = create_dataset()

    env = environment.InvOptEnv(demand_list)
    agent = agent_dqn.Agent(state_size=7, action_size=21, seed=0)
    agent.qnetwork_target.train()
    n_episodes = 1000
    max_t = 10000
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.995

    scores = []
    eps = eps_start

    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            action = decide_capacity_supplier(action)
            next_state, reward, done = env.step(action)
            agent.step(state, action, reward, next_state, done)
            # above step decides whether we will train(learn) the network
            # actor (local_qnetwork) or we will fill the replay buffer
            # if len replay buffer is equal to the batch size then we will
            # train the network or otherwise we will add experience tuple in our 
            # replay buffer.
            state = next_state
            score += reward
            if done:
                print('episode' + str(i_episode) + ':', score)
                scores.append(score)
                break
        eps = max(eps * eps_decay, eps_end)     # decrease the epsilon
    
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Reward')
    plt.xlabel('Epsiode #')
    plt.show()

    agent.qnetwork_local.save_to_state_dict()


# compare the performance of the trained model with the classic policy
def compare_performance():
    demand_test = create_user_demand_dataset()
    model = agent_dqn.QNetwork(state_size=7, action_size=21, seed=0)
    model.eval()
    profit_RL = []
    actions_list = []
    invs_list = []
    shortage_lists = []
    for demand in demand_test:
        env = environment.InvOptEnv(demand)
        env.reset()
        profit = 0
        actions = []
        invs = []
        done = False
        state = env.state
        day_current = 0
        while not done:
            state = torch.from_numpy(state).float().unsqueeze(0).to(model.device)
            with torch.no_grad():
                action_values = model(state)
            action = decide_capacity_supplier(np.argmax(action_values.cpu().data.numpy()))
            actions.append(action)

            next_state, reward, done = env.step(action)
            state = next_state
            invs.append(env.inv_level)
            profit += reward
            day_current += 1
        
        # if the process is over , store the histories of actions and inventories
        # 見たいもの：在庫、需要、発注の変化量
        actions_list.append(actions)
        invs_list.append(invs)
        profit_RL.append(profit)
        shortage_lists.append(env.count_shortage)
    
    save_histories(actions_list, 'actions_list_RL.csv')
    save_histories(invs_list, 'invs_list_RL.csv')
    save_histories(demand_test, 'demand_test.csv')
    save_histories(profit_RL, 'profit_RL.csv')
    save_histories(shortage_lists, 'shortage_lists_RL.csv')

    RL_mean = np.mean(profit_RL)
    print('RL_mean:', RL_mean)

    profit_sS = []
    actions_list = []
    invs_list = []
    shortage_lists = []
    for demand in demand_test:
        total_profit, count_shortage, order_arrival_list, inv_list = profit_calculation_sS(15, 32, demand)
        profit_sS.append(total_profit)
        shortage_lists.append(count_shortage)
        actions_list.append(order_arrival_list)
        invs_list.append(inv_list)

    save_histories(actions_list, 'actions_list_classic.csv')
    save_histories(invs_list, 'invs_list_classic.csv')
    save_histories(profit_sS, 'profit_classic.csv')
    save_histories(shortage_lists, 'shortage_lists_classic.csv')

    sS_mean = np.mean(profit_sS)
    print('sS_mean:', sS_mean)


def decide_capacity_supplier(amount_arrange):
    capacity_supplier = 14
    return min(capacity_supplier, amount_arrange)


# 2次元配列の内容をcsvに保存する
def save_histories(datas_input, file_name):
    import csv

    def check_dimension(arr):
        for item in arr:
            if isinstance(item, list):
                return 2  # リスト内にリストがある場合は2次元配列
        return 1  # それ以外は1次元配列
    dim = check_dimension(datas_input)
    if dim == 1:
        datas_input = [[item] for item in datas_input]
    
    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(datas_input)


# クラシックの手法の発注を行う関数
def profit_calculation_sS(s, S, demand_records):
    total_profit = 0
    inv_level = 25      # inventory on hand, use this to calculate inventory costs
    lead_time = 2
    capacity = 50
    holding_cost = 3
    fixed_order_cost = 50
    variable_order_cost = 10
    unit_price = 30
    order_arrival_list = []
    oerder_histories = []
    inv_list = []
    count_shortage = 0
    # make simulation of the inventory system
    for current_period in range(len(demand_records)):
        inv_pos = inv_level
        if len(order_arrival_list) > 0:
            for i in range(len(order_arrival_list)):
                inv_pos += order_arrival_list[i][1]
        if inv_pos <= s:
            order_quantity = decide_capacity_supplier(min(20, S - inv_pos))
            order_arrival_list.append([current_period + lead_time, order_quantity])
            y = 1
        else:
            order_quantity = 0
            y = 0
        if len(order_arrival_list) > 0:
            if current_period == order_arrival_list[0][0]:
                inv_level = min(capacity, inv_level + order_arrival_list[0][1])
                order_arrival_list.pop(0)
        demand = demand_records[current_period]
        
        count_shortage += 1 if inv_level < demand else 0
        # if exist inventory, sell the inventory
        units_sold = demand if demand <= inv_level else inv_level
        # this equation is the same as the one in the DQN agent
        profit = units_sold * unit_price - holding_cost * inv_level - y * fixed_order_cost - order_quantity * variable_order_cost
        # update inventory amount
        inv_level = max(0, inv_level - demand)
        inv_list.append(inv_level)
        oerder_histories.append(order_quantity)
        total_profit += profit
    print('count_shortage:', count_shortage)
    return total_profit, count_shortage, oerder_histories, inv_list


# とりあえず一番良い発注点sと、補充する量Sを確認→s=15, S=32がベスト
def check_best_sS():
    demand_hist = create_dataset()
    s_S_list = []
    for S in range(1, 61):  # give a little room to allow S to exceed the capacity 
        for s in range(0, S):
            s_S_list.append([s, S])  
            
    profit_sS_list = []
    for sS in s_S_list:
        profit_sS_list.append(profit_calculation_sS(sS[0], sS[1], demand_hist))

    best_sS_profit = np.max(profit_sS_list)
    best_sS = s_S_list[np.argmax(profit_sS_list)]
    print('best sS:', best_sS)


if __name__ == '__main__':
    print('start')
    # train_agent_dqn()
    compare_performance()

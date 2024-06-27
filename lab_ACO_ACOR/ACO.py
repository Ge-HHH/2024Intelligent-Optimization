import random
import numpy as np
import datamaker

import numpy as np
import random

from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import matplotlib.pyplot as plt

class ACO_VRPTW:
    def __init__(self, data, num_ants=10, alpha=1.0, beta=1.0, evaporation_rate=0.5, iterations=50):
        """
        参数:
        data : dict - 包含配送中心、客户和车辆装载量的数据。
        num_ants : int - 蚁群的大小。
        alpha : float - 信息素的重要性。
        beta : float - 启发式信息（如距离的倒数）的重要性。
        evaporation_rate : float - 信息素挥发率。
        iterations : int - 算法运行的迭代次数。
        """
        self.data = data
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.iterations = iterations
        self.depot = data['depot']
        self.customers = data['customers']
        self.vehicle_capacity = data['Q']
        self.num_customers = len(self.customers)
        self.his_cost=[]
        # 包括配送中心，信息素矩阵和可见性矩阵的大小是 num_customers + 1
        self.pheromone_levels = np.ones((self.num_customers + 1, self.num_customers + 1))
        self.visibility = self._calculate_visibility()
        self.distances = self._calculate_distances()
    
    def _calculate_distances(self):
        all_locations = [self.depot['location']] + [c['location'] for c in self.customers]
        distances = np.zeros((self.num_customers + 1, self.num_customers + 1))
        for i in range(self.num_customers + 1):
            for j in range(self.num_customers + 1):
                distances[i][j] = np.hypot(all_locations[i][0] - all_locations[j][0], all_locations[i][1] - all_locations[j][1])
        return distances

    def _calculate_visibility(self):
        """ 计算启发式信息，即可见性矩阵，通常为距离的倒数。 """
        visibility = np.zeros((self.num_customers + 1, self.num_customers + 1))
        locations = [self.depot['location']] + [c['location'] for c in self.customers]
        
        for i in range(self.num_customers + 1):
            for j in range(self.num_customers + 1):
                if i != j:
                    dist = np.hypot(locations[i][0] - locations[j][0], locations[i][1] - locations[j][1])
                    visibility[i][j] = 1.0 / (dist + 1e-10)  # 避免除以零
        return visibility

    def _update_pheromone(self, all_routes):
        """ 更新信息素矩阵。 """
        self.pheromone_levels *= (1 - self.evaporation_rate)  # 信息素蒸发
        for route in all_routes:
            for i in range(len(route) - 1):
                self.pheromone_levels[route[i]][route[i+1]] += 1.0 / len(route)  # 路径越短，增加的信息素越多

    def solve(self):
        best_route = None
        best_cost = float('inf')

        for iter in range(self.iterations):
            all_routes = []
            for ant in range(self.num_ants):
                route = self._construct_solution()
                all_routes.append(route)
                cost = self._calculate_route_cost(route)
                if cost < best_cost:
                    best_cost = cost
                    best_route = route
            self._update_pheromone(all_routes)
            costs=[self._calculate_route_cost(route) for route in all_routes]
            print(f'Iteration {iter + 1}/{self.iterations} - Best cost: {best_cost:.2f} - Mean cost: {np.mean(costs):.2f}')
            self.his_cost.append((best_cost,np.mean(costs)))

        return best_route, best_cost

    def solve_multi_p(self):
        """使用多线程运行 ACO 算法解决 VRPTW。"""
        best_route = None
        best_cost = float('inf')

        with ThreadPoolExecutor(max_workers=self.num_ants) as executor:
            for iteration in range(self.iterations):
                # 提交所有蚂蚁的路径构建任务到线程池
                futures = [executor.submit(self._construct_solution) for _ in range(self.num_ants)]
                all_routes = []
                for future in as_completed(futures):
                    route = future.result()
                    all_routes.append(route)
                    cost = self._calculate_route_cost(route)
                    if cost < best_cost:
                        best_cost = cost
                        best_route = route
                
                # 在所有蚂蚁完成后更新信息素
                self._update_pheromone(all_routes)
                costs = [self._calculate_route_cost(route) for route in all_routes]
                print(f'Iteration {iteration + 1}/{self.iterations} - Best cost: {best_cost:.2f} - Mean cost: {np.mean(costs):.2f}')
                self.his_cost.append((best_cost, np.mean(costs)))

        return best_route, best_cost

    def _construct_solution(self):
        """ 构建蚂蚁的解决方案，包括验证载重约束。 """
        route = [0]  # 初始化路线，始终从配送中心开始
        load = 0
        visited = set()
        # visited.add(0)  # 配送中心视为已访问
        current_time = 0
        while len(visited) < self.num_customers :
            current = route[-1]
            next_city = self._select_next_city(current, visited, load,current_time)
            if next_city == 0:
                load = 0  # 返回配送中心卸货
                current_time = 0
            else:
                load += self.customers[next_city - 1]['demand']
                current_time = current_time + self.distances[current][next_city] + self.customers[next_city - 1]['service_time']
                visited.add(next_city)
            route.append(next_city)
            if next_city != 0:
                visited.add(next_city)

        return route
    
    def Eta(self, current,next, load, current_time):
        if (current==next):
            return 0
        if (next==0):
            return 1/self.distances[current][next]
        if load+self.customers[next-1]['demand']>self.vehicle_capacity and current_time+self.distances[current][next]>self.customers[next-1]['time_window'][1]:
            return 0
        
        dis=max(self.distances[current][next],self.customers[next-1]['time_window'][0]-current_time)

        return 1/(dis+1e-6)
        


    def _select_next_city(self, current, visited, load,current_time):
        """ 根据概率选择下一个城市，并考虑载重约束。 """
        probabilities = []
        for j in range(0,self.num_customers + 1):
            if j not in visited:
                # if (j==87):
                #     print('here')
                # if (j==24):
                #     print('here')
                prob = (self.pheromone_levels[current][j] ** self.alpha) * \
                       (self.Eta(current,j,load,current_time) ** self.beta)
                probabilities.append(prob)
            else:
                probabilities.append(0)

        # 归一化概率
        probabilities = np.array(probabilities)
        if np.sum(probabilities) > 0:
            probabilities /= np.sum(probabilities)
            next_city = np.random.choice(self.num_customers + 1, 1, p=probabilities)[0]
            return next_city
        else:
            return 0

    def _calculate_route_cost(self, route):
        """ 计算给定路线的成本，即路线长度。 """
        cost = 0
        locations = [self.depot['location']] + [c['location'] for c in self.customers]
        for i in range(len(route) - 1):
            cost += np.hypot(locations[route[i]][0] - locations[route[i+1]][0],
                             locations[route[i]][1] - locations[route[i+1]][1])
        if route[-1] != 0:
            cost += np.hypot(locations[route[-1]][0] - locations[0][0],
                             locations[route[-1]][1] - locations[0][1])
        return cost
    
    def distance(self, a, b):
        return np.hypot(a[0]-b[0],a[1]-b[1])

def eva_rate_influence(data):
    eva_rates=np.linspace(0.1,0.9,10)
    writer=open("eva_rate_influence.txt","w")
    diff_means=[]
    for (i,eva_rate) in enumerate(eva_rates):
        aco=ACO_VRPTW(data, num_ants=20, alpha=1.0, beta=1.0, evaporation_rate=eva_rate, iterations=50)
        best_route, best_cost = aco.solve()
        diffs=[abs(i[0]-i[1]) for i in aco.his_cost]
        diff_mean=np.mean(diffs)
        writer.write(f'{eva_rate} {best_cost} {diff_mean}\n')
        diff_means.append(diff_mean)
    
    plt.figure(figsize=(16,10))
    plt.plot(eva_rates,diff_means)
    plt.show()
def regular_test(data):
    # data=datamaker.load_vrptw_data('vrptw_data_1.json')
    aco = ACO_VRPTW(data, num_ants=20, alpha=3 ,beta=2, evaporation_rate=0.1, iterations=50)
    t=time.time()
    # best_route, best_cost = aco.solve_multi_p()
    best_route, best_cost = aco.solve()
    t=time.time()-t
    
    print(f'Cost: {best_cost:.2f} - Time: {t:.2f} s')
    #visualize the best route
    import matplotlib.pyplot as plt
    depot = data['depot']
    customers = data['customers']
    depot_x, depot_y = depot['location']
    customer_x = [c['location'][0] for c in customers]
    customer_y = [c['location'][1] for c in customers]
    plt.plot(depot_x, depot_y, 'ro')
    plt.scatter(customer_x, customer_y)

    with open("ACO_result.txt","w") as f:
        for i in best_route:
            f.write(str(i)+' ')
        f.write('\n')
        f.write(str(len(aco.his_cost))+ '\n')
        for i in aco.his_cost:
            f.write(str(i[0])+' '+str(i[1])+'\n')
    #route arrows and number
    print(best_route)
    all_locations = [(depot_x, depot_y)] + [(customer_x[i], customer_y[i]) for i in range(len(customers))]
    for i in range(len(best_route) - 1):
        start = all_locations[best_route[i]]
        end = all_locations[best_route[i + 1]]
        plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1], head_width=1, head_length=1, fc='k', ec='k')
        plt.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2, str(i + 1), fontsize=8)
    plt.arrow(all_locations[best_route[-1]][0], all_locations[best_route[-1]][1], depot_x - all_locations[best_route[-1]][0], depot_y - all_locations[best_route[-1]][1], head_width=1, head_length=1, fc='k', ec='k') 
    plt.text((all_locations[best_route[-1]][0] + depot_x) / 2, (all_locations[best_route[-1]][1] + depot_y) / 2, str(len(best_route)), fontsize=8) 
    plt.show()

    #plot the cost
    plt.plot([i[0] for i in aco.his_cost],label='best cost')
    plt.plot([i[1] for i in aco.his_cost],label='mean cost')
    plt.legend()
    plt.show()

    #plot diff
    diffs=[abs(i[0]-i[1]) for i in aco.his_cost]
    plt.plot(diffs,label='diff mean')
    plt.legend()
    plt.show()

def alpha_beta_influence(data):
    alphas=np.linspace(1,5,10)
    betas=np.linspace(1,5,10)
    writer=open("alpha_beta_influence.txt","w")
    diff_means=np.zeros((10,10))
    best_costs=np.zeros((10,10))
    mean_costs=np.zeros((10,10))
    mean_best_costs=np.zeros((10,10))
    for (i,alpha) in enumerate(alphas):
        for (j,beta) in enumerate(betas):
            aco=ACO_VRPTW(data, num_ants=20, alpha=alpha, beta=beta, evaporation_rate=0.3, iterations=50)
            best_route, best_cost = aco.solve()
            diffs=[abs(i[0]-i[1]) for i in aco.his_cost]
            diff_mean=np.mean(diffs)
            diff_means[i][j]=diff_mean
            best_costs[i][j]=best_cost
            mean_best_costs[i][j]=np.mean([i[0] for i in aco.his_cost])
            mean_costs[i][j]=np.mean([i[1] for i in aco.his_cost])
            writer.write(f'{alpha} {beta} {best_cost} {mean_best_costs[i][j]} {mean_costs[i][j]} {diff_mean}\n')
    
    
    #3d plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(alphas, betas)
    ax.plot_surface(X, Y, best_costs, cmap='viridis',alpha=0.8)
    ax.set_xlabel('alpha')
    ax.set_ylabel('beta')
    ax.set_zlabel('best cost')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(alphas, betas)
    ax.plot_surface(X, Y, mean_costs, cmap='viridis',alpha=0.8)
    ax.set_xlabel('alpha')
    ax.set_ylabel('beta')
    ax.set_zlabel('mean cost')
    plt.show()

    plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(alphas, betas)
    ax.plot_surface(X, Y, mean_best_costs, cmap='viridis',alpha=0.8)
    ax.set_xlabel('alpha')
    ax.set_ylabel('beta')
    ax.set_zlabel('mean best cost')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(alphas, betas)
    ax.plot_surface(X, Y, diff_means, cmap='viridis',alpha=0.8)
    ax.set_xlabel('alpha')
    ax.set_ylabel('beta')
    ax.set_zlabel('diff mean')
    plt.show()

def read_alpha_beta_influence():
    alphas=[]
    betas=[]
    best_costs=[]
    mean_costs=[]
    diff_means=[]
    mean_best_costs=[]
    with open("alpha_beta_influence_2.txt","r") as f:
        lines=f.readlines()
        for (i,line) in enumerate(lines):
            line=line.split()
            alphas.append(float(line[0]))
            betas.append(float(line[1]))
            best_costs.append(float(line[2]))
            mean_best_costs.append(float(line[3]))
            mean_costs.append(float(line[4]))
            diff_means.append(float(line[5]))

    best_costs=np.array(best_costs).reshape((10,10))
    mean_costs=np.array(mean_costs).reshape((10,10))
    diff_means=np.array(diff_means).reshape((10,10))
    mean_best_costs=np.array(mean_best_costs).reshape((10,10))
    alphas=np.array(alphas).reshape((10,10))
    betas=np.array(betas).reshape((10,10))
    X,Y=alphas,betas
    # X,Y=np.array(alphas),np.array(betas)
    # best_costs=np.array(best_costs).reshape((10,10))
    # mean_costs=np.array(mean_costs)
    # diff_means=np.array(diff_means)
    #3d plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # X, Y = np.meshgrid(alphas, betas)
    ax.plot_surface(X, Y, best_costs, cmap='viridis',alpha=0.8)
    ax.set_xlabel('alpha')
    ax.set_ylabel('beta')
    ax.set_zlabel('best cost')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # X, Y = np.meshgrid(alphas, betas)
    ax.plot_surface(X, Y, mean_costs, cmap='viridis',alpha=0.8)
    ax.set_xlabel('alpha')
    ax.set_ylabel('beta')
    ax.set_zlabel('mean cost')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # X, Y = np.meshgrid(alphas, betas)
    ax.plot_surface(X, Y, diff_means, cmap='viridis',alpha=0.8)
    ax.set_xlabel('alpha')
    ax.set_ylabel('beta')
    ax.set_zlabel('diff mean')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # X, Y = np.meshgrid(alphas, betas)
    ax.plot_surface(X, Y, mean_best_costs, cmap='viridis',alpha=0.8)
    ax.set_xlabel('alpha')
    ax.set_ylabel('beta')
    ax.set_zlabel('mean best cost')
    plt.show()



if __name__ == '__main__':
    data=datamaker.read_from_txt('./data/C101.txt')
    # eva_rate_influence(data)
    regular_test(data)
    # alpha_beta_influence(data)
    # read_alpha_beta_influence()
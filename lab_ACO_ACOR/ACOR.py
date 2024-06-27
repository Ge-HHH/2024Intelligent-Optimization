import numpy as np
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

def objective_function(x):
    return -((6.452 * (x[0] + 0.125 * x[1]) * (np.cos(x[0]) - np.cos(2 * x[1])) ** 2) / np.sqrt(0.8 + (x[0] - 4.2) ** 2 + 2 * (x[1] - 7) ** 2)+3.226*x[1])

def objective_function1(x):
    return (20+x[0]**2+x[1]**2-10*(np.cos(2*np.pi*x[0])+np.cos(2*np.pi*x[1])))

def acor(objective_function, bounds, n_ants=100, max_iter=100, rho=0.1, sigma=0.1):
    dim = len(bounds)
    ants = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_ants, dim))
    best_solution = None
    best_fitness = np.inf
    history = []
    for iteration in range(max_iter):
        # fitness = np.apply_along_axis(objective_function, 1, ants)
        fitness = [objective_function(ant) for ant in ants]
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_fitness:
            best_fitness = fitness[best_idx]
            best_solution = ants[best_idx]
        mean_fitness = np.mean(fitness)
        history.append((best_fitness, mean_fitness))

        sorted_indices = np.argsort(fitness)
        sorted_ants = ants[sorted_indices]

        weights = np.exp(((1-np.arange(n_ants)) / (rho * n_ants))**2)
        weights /= np.sum(weights)

        new_ants = np.zeros_like(ants)
        for i in range(n_ants):
            idx = np.random.choice(np.arange(n_ants), p=weights)
            new_ants[i] = np.random.normal(sorted_ants[idx], sigma)

        new_ants = np.clip(new_ants, bounds[:, 0], bounds[:, 1])
        #将ants和new_ants合并
        tmp = np.vstack([ants, new_ants])
        #计算fitness
        tmp_fitness = [objective_function(ant) for ant in tmp]
        #按照fitness排序
        tmp_sorted_indices = np.argsort(tmp_fitness)
        tmp_sorted = tmp[tmp_sorted_indices[:n_ants]]
        ants = tmp_sorted
        print(f"Iteration {iteration+1}/{max_iter}: Best Fitness = {best_fitness}, Best Solution = {best_solution}, Mean Fitness = {mean_fitness}")

    return best_solution, best_fitness, np.array(history)

def vis():
    bounds1=np.array([[0,10],[0,10]])
    X=np.linspace(bounds1[0,0],bounds1[0,1],100)
    Y=np.linspace(bounds1[1,0],bounds1[1,1],100)
    X,Y=np.meshgrid(X,Y)
    Z=-objective_function([X,Y])
    fig=plt.figure(figsize=(20,10))
    ax1=fig.add_subplot(1,2,1,projection='3d')
    ax1.plot_surface(X,Y,Z,cmap='viridis')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Function 1')

    ax2=fig.add_subplot(1,2,2,projection='3d')
    bounds2=np.array([[-5,5],[-5,5]])
    X=np.linspace(bounds2[0,0],bounds2[0,1],100)
    Y=np.linspace(bounds2[1,0],bounds2[1,1],100)
    X,Y=np.meshgrid(X,Y)
    Z=objective_function1([X,Y])
    ax2.plot_surface(X,Y,Z,cmap='viridis')
    ax2.set_xlabel('X1')
    ax2.set_ylabel('X2')
    ax2.set_zlabel('Z')
    ax2.set_title('Function 2')
    plt.show()

    
if __name__ == '__main__':
    # vis()
    bounds = np.array([[0, 10], [0, 10]])  # 示例变量范围
    bounds1 = np.array([[-5, 5], [-5, 5]])  # 示例变量范围

    # means=[]
    # rhos=[0.5,1,2,5,10]

    # for rho in rhos:
    #     best_solution, best_fitness,his = acor(objective_function, bounds,n_ants=100,max_iter=100,rho=rho,sigma=1)
    #     print("Best Solution:", best_solution)
    #     print("Best Fitness:", best_fitness)
    #     means.append(his[:,1])
    
    # fig=plt.figure(figsize=(10,5))
    # x=np.arange(100)
    # for i in range(len(rhos)):
    #     plt.plot(x,means[i],label=f'rho={rhos[i]}')
    # plt.xlabel('Iteration')
    # plt.ylabel('Mean Fitness')
    # plt.legend()
    # plt.show()

    # means=[]
    # sigmas=[0.1,0.5,1,2,5]

    # for sigma in sigmas:
    #     best_solution, best_fitness,his = acor(objective_function, bounds,n_ants=100,max_iter=100,rho=2,sigma=sigma)
    #     print("Best Solution:", best_solution)
    #     print("Best Fitness:", best_fitness)
    #     means.append(his[:,1])
    
    # fig=plt.figure(figsize=(10,8))
    # x=np.arange(100)
    # for i in range(len(sigmas)):
    #     plt.plot(x,means[i],label=f'sigma={sigmas[i]}')
    # plt.xlabel('Iteration')
    # plt.ylabel('Mean Fitness')
    # plt.legend()
    # plt.show()


    # bounds = np.array([[-5, 5], [-5, 5]])  # 示例变量范围
    # best_solution, best_fitness,his = acor(objective_function1, bounds,n_ants=100,max_iter=100,rho=2,sigma=1)
    # print("Best Solution:", best_solution)
    # print("Best Fitness:", best_fitness)

    rhos=[0.5,1,2,5,10]
    sigmas=[0.1,0.5,1,2,5]
    means=[]
    stds=[]
    means1=[]
    stds1=[]
    # for rho in rhos:
    #     bests=[]
    #     bests1=[]
    #     for i in range(10):
    #         best_solution, best_fitness,his = acor(objective_function, bounds,n_ants=100,max_iter=100,rho=rho,sigma=1)
    #         bests.append(best_fitness)
    #     for i in range(10):
    #         best_solution, best_fitness,his = acor(objective_function1, bounds1,n_ants=100,max_iter=100,rho=rho,sigma=1)
    #         bests1.append(best_fitness)
    #     means.append(np.mean(bests))
    #     stds.append(np.std(bests))
    #     means1.append(np.mean(bests1))
    #     stds1.append(np.std(bests1))
    
    # with open('rst_rho.txt','w') as f:
    #     for i in range(len(rhos)):
    #         f.write(f'rho={rhos[i]}:%.3f±%.3f\n'%(means[i],stds[i]))
    #     for i in range(len(rhos)):
    #         f.write(f'rho={rhos[i]}:%.3f±%.3f\n'%(means1[i],stds1[i]))

    for sigma in sigmas:
        bests=[]
        bests1=[]
        for i in range(10):
            best_solution, best_fitness,his = acor(objective_function, bounds,n_ants=100,max_iter=100,rho=2,sigma=sigma)
            bests.append(best_fitness)
        for i in range(10):
            best_solution, best_fitness,his = acor(objective_function1, bounds1,n_ants=100,max_iter=100,rho=2,sigma=sigma)
            bests1.append(best_fitness)
        means.append(np.mean(bests))
        stds.append(np.std(bests))
        means1.append(np.mean(bests1))
        stds1.append(np.std(bests1))

    with open('rst_sigma.txt','w') as f:
        for i in range(len(sigmas)):
            f.write(f'sigma={sigmas[i]}:%.3f±%.3f\n'%(means[i],stds[i]))
        for i in range(len(sigmas)):
            f.write(f'sigma={sigmas[i]}:%.3f±%.3f\n'%(means1[i],stds1[i]))
    


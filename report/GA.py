import random
import numpy as np
import datamaker

class GeneticAlgorithm:
    def __init__(self, data, population_size=100, generations=300, mutation_rate=0.01, crossover_rate=0.7, 
                 greedy_init=False, crossover_type='order'):
        self.data = data
        self.depot = data['depot']
        for i in range(len(data['customers'])):
            data['customers'][i]['id']=i+1
        self.customers = data['customers']
        self.vehicle_capacity = data['Q']
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = self.initialize_population(greedy=greedy_init)
        self.his_cost=[]
        self.crossover_type = crossover_type
    
    @staticmethod
    def distance(location1, location2):
        return ((location1[0] - location2[0]) ** 2 + (location1[1] - location2[1]) ** 2) ** 0.5

    def initialize_population(self, greedy=False):
        population = []
        if greedy:
            for _ in range(self.population_size):
                population.append(self.greedy_init())
        else:
            for _ in range(self.population_size):
                individual = list(self.customers)
                random.shuffle(individual)
                population.append(individual)
        return population
    
    def greedy_init(self):
        individual =[]
        current_load = 0
        current_location = self.depot['location']
        current_time = 0
        visited = set()
        for _ in range(len(self.customers)):
            probs=[]
            for customer in self.customers:
                if customer['id'] in visited:
                    probs.append(0)
                    continue
                if current_load + customer['demand'] > self.vehicle_capacity or current_time + self.distance(current_location, customer['location']) > customer['time_window'][1]:
                    dis=self.distance(current_location, self.depot['location'])+self.distance(self.depot['location'],customer['location'])
                else:
                    dis=self.distance(current_location, customer['location'])
                probs.append(1.0/dis+1e-6)
            probs=np.array(probs)
            probs=probs/np.sum(probs)
            next_customer = np.random.choice(self.customers, p=probs)
            if current_load + next_customer['demand'] > self.vehicle_capacity or current_time + self.distance(current_location, next_customer['location']) > next_customer['time_window'][1]:
                current_location = next_customer['location']
                current_load = next_customer['demand']
                current_time = max(self.distance(self.depot['location'],next_customer['location']),next_customer['time_window'][0]) + next_customer['service_time']
                visited.add(next_customer['id'])
                individual.append(next_customer)
            else:
                current_location = next_customer['location']
                current_load += next_customer['demand']
                current_time = max(current_time + self.distance(current_location, next_customer['location']), next_customer['time_window'][0]) + next_customer['service_time']
                individual.append(next_customer)
                visited.add(next_customer['id'])
        return individual

    def route_distance(self, individual):
        total_distance = 0
        current_load = 0
        current_location = self.depot['location']
        current_time = 0
        for customer in individual:
            if current_load + customer['demand'] > self.vehicle_capacity or current_time + self.distance(current_location, customer['location']) > customer['time_window'][1]:
                total_distance += self.distance(current_location, self.depot['location'])
                current_location = self.depot['location']
                current_load = 0
                current_time = 0
            total_distance += self.distance(current_location, customer['location'])
            current_location = customer['location']
            current_load += customer['demand']
            current_time = max(current_time + self.distance(current_location, customer['location']), customer['time_window'][0]) + customer['service_time']
        total_distance += self.distance(current_location, self.depot['location'])
        return total_distance

    def fitness(self, individual):
        return 1 / self.route_distance(individual)
    
    def dp_fitness(self,individual):
        n=len(individual)
    def select_parents(self,fitnesses=None):
        def tournament_selection(fits):
            tournament_size = 5
            contenders = random.sample(range(len(self.population)), tournament_size)
            winner = max(contenders, key=lambda x: fits[x])
            return self.population[winner]

        parent1 = tournament_selection(fitnesses)
        parent2 = tournament_selection(fitnesses)
        return parent1, parent2
    #Order crossover
    def order_crossover(self, parent1, parent2):
        start, end = sorted(random.sample(range(len(parent1)), 2))
        offspring = [None] * len(parent1)
        offspring[start:end+1] = parent1[start:end+1]
        current_pos = (end + 1) % len(parent1)
        for gene in parent2:
            if gene not in offspring:
                offspring[current_pos] = gene
                current_pos = (current_pos + 1) % len(parent1)
        return offspring
    def vsubroute_crossover(self, parent1, parent2):
        def split_list(lst, delimiter):
            result = []
            current = []
            
            for item in lst:
                if item == delimiter:
                    if len(current) ==0:
                        continue
                    result.append(current)
                    current = []
                else:
                    current.append(item)
            if (len(current)!=0):
                 result.append(current)
            return result
        route1=split_list(self.individual_2_route(parent1),0)
        route2=split_list(self.individual_2_route(parent2),0)
        # offspring=[]
        customers=[]
        for i in random.sample(range(len(route1)),len(route1)//2):
            customers.extend(route1[i])
            # offspring.append(route1[i])
        for i in range(len(route2)):
            for j in route2[i]:
                if j not in customers:
                    customers.append(j)
        rst=[self.customers[i-1] for i in customers]
        return rst

    def crossover(self, parent1, parent2):
        if self.crossover_type == 'order':
            return self.order_crossover(parent1, parent2)
        elif self.crossover_type == 'subroute':
            return self.subroute_crossover(parent1, parent2)
        else:
            raise ValueError("Crossover type not supported")

    def mutate(self, individual):
        # Implement a mutation method, such as swap mutation
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

    def evolve(self):
        new_population = []
        fitnesses = [self.fitness(individual) for individual in self.population]
        for _ in range(self.population_size):
            parent1, parent2 = self.select_parents(fitnesses=fitnesses)
            if random.random() < self.crossover_rate:
                offspring = self.crossover(parent1, parent2)
            else:
                offspring = random.choice([parent1, parent2])
            self.mutate(offspring)
            new_population.append(offspring)
        new_fitnesses = [self.fitness(individual) for individual in new_population]
        new_pop=list(zip(new_population,new_fitnesses))
        new_pop.extend(list(zip(self.population,fitnesses)))
        new_pop=sorted(new_pop,key=lambda x:-x[1])[:self.population_size]
        self.population = [x[0] for x in new_pop]

    def run(self):
        for _ in range(self.generations):
            self.evolve()
            fits=[self.route_distance(individual) for individual in self.population]
            best_solution = min(range(len(fits)), key=lambda x: fits[x])
            best_cost = fits[best_solution]
            mean_cost = np.mean(fits)
            best_solution = self.population[best_solution]
            self.his_cost.append((best_cost,mean_cost))
            print(f"Iter {_+1}/{self.generations} Best Cost:", best_cost)
        
        return best_solution

    def individual_2_route(self, individual):
        total_distance = 0
        current_load = 0
        current_location = self.depot['location']
        current_time = 0
        route = [0]
        for customer in individual:
            if current_load + customer['demand'] > self.vehicle_capacity or current_time + self.distance(current_location, customer['location']) > customer['time_window'][1]:
                total_distance += self.distance(current_location, self.depot['location'])
                current_location = self.depot['location']
                current_load = 0
                current_time = 0
                route.append(0)
            total_distance += self.distance(current_location, customer['location'])
            current_location = customer['location']
            current_load += customer['demand']
            current_time = max(current_time + self.distance(current_location, customer['location']), customer['time_window'][0]) + customer['service_time']
            route.append(customer['id'])
        total_distance += self.distance(current_location, self.depot['location'])
        route.append(0)
        return route
    
    
    def visialize(self, route):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        locations = [self.depot['location']] + [c['location'] for c in self.customers]
        x = [loc[0] for loc in locations]
        y = [loc[1] for loc in locations]
        plt.plot(x[0], y[0], 'ro')
        plt.scatter(x[1:], y[1:])
        for i in range(len(route) - 1):
            plt.arrow(x[route[i]], y[route[i]], x[route[i+1]]-x[route[i]], y[route[i+1]]-y[route[i]], head_width=1, head_length=1, fc='r', ec='r')
            plt.text((x[route[i]]+x[route[i+1]])/2, (y[route[i]]+y[route[i+1]])/2, str(i+1))

        plt.show()

def distance(loc1, loc2):
    return ((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2) ** 0.5


# Example usage
# Assuming `customers` is a list of dictionaries with keys 'location' and 'demand'
# Assuming `depot` is a location tuple, e.g., (0, 0)
# Vehicle capacity as an integer
data=datamaker.read_from_txt('./data/C101.txt')
genetic_algo = GeneticAlgorithm(data, population_size=100, generations=900, mutation_rate=0.03, crossover_rate=0.6, greedy_init=True, crossover_type='order')
best_solution = genetic_algo.run()
route = genetic_algo.individual_2_route(best_solution)
print(route)
print("Total distance:", genetic_algo.fitness(best_solution))

# with open('GA_result.txt','w') as f:
#     for i in route:
#         f.write(str(i)+' ')
#     f.write('\n')
#     f.write(str(len(genetic_algo.his_cost))+ '\n')
#     for i in genetic_algo.his_cost:
#         f.write(str(i[0])+' '+str(i[1])+'\n')


genetic_algo.visialize(route)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
plt.plot([x[0] for x in genetic_algo.his_cost],label='Best Cost')
plt.plot([x[1] for x in genetic_algo.his_cost],label='Mean Cost')
plt.legend()
plt.show()




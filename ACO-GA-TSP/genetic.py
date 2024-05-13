import copy
import numpy as np
import random
import csv
import math

def calc_distancia(cidade1, cidade2):
    """Calcula distância entre duas cidades

    Keyword arguments:
    cidade1 -- dict com coordenadas [x,y] da cidade
    cidade2 -- dict com coordenadas [x,y] da cidade
    """
    return math.sqrt((cidade1[1] - cidade2[1])**2 +
                     (cidade1[2] - cidade2[2])**2)

# def parse_cities(data_size):
#     data_size += 1
#     counter = 0
#     matrix = [[0 for _ in range(data_size)] for _ in range(data_size)]

#     with open('distances.csv', encoding="utf8") as file:
#         read = csv.reader(file)
#         next(read)
#         src_id = 0
#         dest_id = 0
#         for row in read:
#             for i in range(data_size + 1):
#                 if i == 0:
#                     continue
#                 else:
#                     matrix[src_id][dest_id] = float(row[i])
#                 dest_id += 1
#             dest_id = 0
#             src_id += 1
#             counter += 1
#             if counter == data_size:
#                 break

#     return matrix

def parse_cities(data_size):
    cities = []

    with open('./data/chn31.tsp', 'r') as file:
        for line in file:
            parts = line.strip().split()
            city_id = parts[0]
            x = float(parts[1])
            y = float(parts[2])
            cities.append((city_id, x, y))
            
    distance_matrix = [[0] * data_size for _ in range(data_size)]
    
    for i in range(data_size):
        for j in range(data_size):
            distance_matrix[i][j] = calc_distancia(cities[i], cities[j])
    
    return distance_matrix


matrix = parse_cities(30)
num_cities = len(matrix)
neighbour_rate = num_cities // 8


def evaluate_solution(path):
    time_taken = 0
    for i in range(len(path) - 1):
        time_taken += matrix[path[i]][path[i+1]]

    return time_taken

def generate_random_solution():
    cities = list(range(1, num_cities))  # cities from 1 to len(matrix) - 1
    random.shuffle(cities)
    return [0] + cities

def get_closest_city(current_city, available_cities):
    heuristic_list = [(x, matrix[current_city][x]) for x in available_cities]
    heuristic_list = sorted(heuristic_list, key=lambda k: k[1])

    return heuristic_list[0][0]

def generate_greedy_solution():
    sol = [0]

    all_cities = list(range(1, num_cities))

    while len(sol) < num_cities:
        next_city = get_closest_city(sol[-1], all_cities)
        sol.append(next_city)
        all_cities.remove(next_city)

    return sol

def generate_population(population_size):
    solutions = []
    for i in range(population_size):
        solutions.append(generate_random_solution())
    return solutions

def tournament_selection(population, tournament_size):
    pop_copy = copy.deepcopy(population)
    best_sol = pop_copy[0]
    best_score = evaluate_solution(pop_copy[0])

    for i in range(tournament_size):
        idx = np.random.randint(0, len(pop_copy))
        score = evaluate_solution(pop_copy[idx])
        if score > best_score:
            best_score = score
            best_sol = pop_copy[idx]

        del pop_copy[idx]

    return best_sol


def generate_crossovers(parent1, parent2):
    first_cross = copy.deepcopy(parent1)
    second_cross = copy.deepcopy(parent2)

    random_idx = random.randint(0, num_cities - neighbour_rate)  # Adjusted range

    # Perform crossover
    first_cross[random_idx:random_idx + neighbour_rate], second_cross[random_idx:random_idx + neighbour_rate] = \
        second_cross[random_idx:random_idx + neighbour_rate], first_cross[random_idx:random_idx + neighbour_rate]

    return fix_duplicate_cities(first_cross), fix_duplicate_cities(second_cross)


def generate_neighbor(solution):
    neighbor_solution = solution.copy()
    neighbor_solution = neighbor_solution[1:]
    for _ in range(neighbour_rate):
        i, j = random.sample(range(len(neighbor_solution)), 2)
        neighbor_solution[i], neighbor_solution[j] = neighbor_solution[j], neighbor_solution[i]

    neighbor_solution.insert(0, 0)
    return neighbor_solution

def fix_duplicate_cities(path):
    unique_cities = set()
    fixed_path = []
    for city in path:
        if city not in unique_cities:
            unique_cities.add(city)
            fixed_path.append(city)
    # Ensure all cities are visited
    for city in range(len(path)):
        if city not in fixed_path:
            fixed_path.append(city)
    return fixed_path
def roulette_select(population):
    score_sum = sum([evaluate_solution(solution) for solution in population])
    selection_probs = [evaluate_solution(solution) / score_sum for solution in population]

    return population[np.random.choice(len(population),p = selection_probs)]

def remove_least_fit(population, population_size):
    population = sorted(population, key=lambda x: evaluate_solution(x))
    population = population[0:population_size]
    return population


def genetic_algorithm(num_iterations, population_size, reproduction_rate):
    population = generate_population(population_size)

    best_sol = min(population, key=lambda x: evaluate_solution(x))
    best_score = evaluate_solution(best_sol)
    current_iteration = 0

    # print(f"\n\nInitial Solution: {best_sol} \nScore: {best_score}\n")

    while current_iteration < num_iterations:
        first_offspring = best_sol
        second_offspring = roulette_select(population)

        cross_over1, cross_over2 = generate_crossovers(first_offspring, second_offspring)
        if random.randint(0, 10) < 5:
            mutation = generate_neighbor(first_offspring)
            population.append(mutation)
            mutation = generate_neighbor(cross_over1)
            population.append(mutation)
            mutation = generate_neighbor(cross_over2)
            population.append(mutation)

        else:
            population.append(cross_over1)
            population.append(cross_over2)


        for i in range(int(reproduction_rate * population_size)):
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            child_1, child_2 = generate_crossovers(parent1, parent2)

            if random.randint(0, 10) < 5:
                mutation = generate_neighbor(child_1)
                population.append(mutation)
                mutation = generate_neighbor(child_2)
                population.append(mutation)

            population.append(child_1)
            population.append(child_2)

        temp_score = best_score

        if best_score > evaluate_solution(min(population, key=lambda x: evaluate_solution(x))):
            best_sol = min(population, key=lambda x: evaluate_solution(x))
            best_score = evaluate_solution(best_sol)

            # print(
                # f"Iteration {current_iteration}: upgrade \n\t\tScore: {temp_score} -> {best_score}\n")

        population = remove_least_fit(population, population_size)
        current_iteration += 1

    print(f"custo total: {best_score}, caminho: {best_sol}")

genetic_algorithm(10000, 5, 0.25)

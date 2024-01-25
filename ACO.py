import xml.etree.ElementTree as ET
import numpy as np

file_path = 'brazil58.xml'

# Parse the XML
tree = ET.parse(file_path)
root = tree.getroot()

nodes = root.findall(".graph/vertex")

cost_matrix = np.zeros((len(nodes), len(nodes)))
i = 0
for node in nodes:

    for edge in node:
        cost_matrix[i][int(edge.text)] = float(edge.attrib["cost"])
    
    i += 1
# Displaying the distance matrix
print(cost_matrix)

# Random Pheromone Initialisation
def initialize_pheromones(num_cities):
    return np.random.rand(num_cities, num_cities) * 0.1


def ant_path_generation_with_1_over_d(num_ants, pheromone_matrix, cost_matrix, alpha, beta):
    num_cities = len(cost_matrix)
    # List to store generated paths for all ants
    paths = []  

    # Iterate through each ant
    for ant in range(num_ants):  
        visited = [False] * num_cities  
        path = []  
        # Choose a random starting city for this ant
        current_city = np.random.randint(0, num_cities)  
        visited[current_city] = True  
        # Add city to the path
        path.append(current_city)  

        for _ in range(num_cities - 1):  
            # List to store probabilities for choosing the next city
            probabilities = []  

            # Calculate probabilities for unvisited neighboring cities
            for city in range(num_cities):
                if not visited[city] and city != current_city:
                    pheromone = pheromone_matrix[current_city][city]  
                    distance = cost_matrix[current_city][city]  

                    # Calculate inverse distance 
                    inverse_distance = 1 / max(distance, 0.0001)

                    # Combine pheromone and inverse distance information
                    combined_prob = (pheromone ** alpha) * (inverse_distance ** beta)
                    probabilities.append((city, combined_prob))  

            total_probabilities = sum(prob[1] for prob in probabilities)  

            # If total probabilities are invalid or very small, reassign probabilities based on distances
            if total_probabilities <= 0:
                remaining_unvisited = [city for city in range(num_cities) if not visited[city]]
                remaining_distances = [1 / max(cost_matrix[current_city][city], 0.0001) for city in remaining_unvisited]
                total_remaining_distances = sum(remaining_distances)

                # Assign probabilities in proportion to the distances for unvisited cities
                probabilities = [(remaining_unvisited[i], remaining_distances[i] / total_remaining_distances) for i in range(len(remaining_unvisited))]
            else:
                # Normalise probabilities to ensure they sum up to 1
                probabilities = [(city, prob / total_probabilities) for city, prob in probabilities]

            # Choose the next city based on probabilities
            next_city = np.random.choice([city_prob[0] for city_prob in probabilities],
                                         p=[city_prob[1] for city_prob in probabilities])

            # Update ant's path and visited cities
            path.append(next_city)
            visited[next_city] = True
            current_city = next_city

        paths.append(path)  

    return paths 



# Code for using Q/d heuristic 
def ant_path_generation_with_Q_over_d(num_ants, pheromone_matrix, cost_matrix, alpha, beta):
    num_cities = len(cost_matrix)
    paths = []

    for ant in range(num_ants):
        visited = [False] * num_cities
        path = []
        current_city = np.random.randint(0, num_cities)
        visited[current_city] = True
        path.append(current_city)

        for _ in range(num_cities - 1):
            probabilities = []

            for city in range(num_cities):
                if not visited[city] and city != current_city:
                    pheromone = pheromone_matrix[current_city][city]
                    distance = cost_matrix[current_city][city]
                    
                    # Calculate inverse distance 
                    inverse_distance = 1 / max(distance, 0.0001)  
                    Q_over_d = pheromone * inverse_distance

                    combined_prob = (Q_over_d ** beta)
                    probabilities.append((city, combined_prob))

            total_probabilities = sum(prob[1] for prob in probabilities)

            # Filter out non-positive probabilities and normalise if valid
            valid_probabilities = [(city, prob / total_probabilities) for city, prob in probabilities if prob > 0]
            total_valid_probabilities = sum(prob[1] for prob in valid_probabilities)

            if not np.isclose(total_valid_probabilities, 1.0):  
                # Adjust probabilities to ensure they sum up to 1
                valid_probabilities = [(city, prob / total_valid_probabilities) for city, prob in valid_probabilities]

            # Ensure valid probabilities exist before making a choice
            if valid_probabilities:
                next_city = np.random.choice([city_prob[0] for city_prob in valid_probabilities],
                                             p=[city_prob[1] for city_prob in valid_probabilities])
            else:
                # If no valid probabilities, choose a random unvisited city
                next_city = np.random.choice([city for city in range(num_cities) if not visited[city]])

            path.append(next_city)
            visited[next_city] = True
            current_city = next_city

        paths.append(path)

    return paths


# Pheromone Update
def update_pheromones(pheromone_matrix, paths, cost_matrix, pheromone_deposit):
    num_cities = len(cost_matrix)
    # Finding where ants have been
    for path in paths:
        total_cost = sum(cost_matrix[path[i - 1]][path[i]] for i in range(num_cities))
        for i in range(num_cities - 1):
            pheromone_matrix[path[i]][path[i + 1]] += pheromone_deposit / total_cost
        pheromone_matrix[path[-1]][path[0]] += pheromone_deposit / total_cost
    # Updating pheromone matrix
    return pheromone_matrix

# Pheromone Evaporation
def evaporate_pheromones(pheromone_matrix, evaporation_rate):
    new_pheromones = pheromone_matrix * (1 - evaporation_rate)
    if np.any(new_pheromones < 0):
        # Ensures pheromones always exist
        raise ValueError("Pheromones have become negative. Adjust the evaporation rate.")
    return new_pheromones

def ACO_algorithm(num_ants, num_iterations, cost_matrix, alpha, beta, evaporation_rate, Q, use_elitist_ants=False):
    num_cities = len(cost_matrix)
    pheromone_matrix = initialize_pheromones(num_cities)

    # Creating initial variable values
    best_path = None
    best_cost = float('inf')
    average_cost = 0
    worst_cost = float('-inf')
    worst_path = None

    # List to store paths of elitist ants
    elitist_paths = []  

    # Pheromone update after every iteration
    for iteration in range(num_iterations):
        ant_paths = ant_path_generation_with_1_over_d(num_ants, pheromone_matrix, cost_matrix, alpha, beta)
        
        # Save best paths if elitist ants are enabled
        if use_elitist_ants:
            elitist_paths.extend(sorted(ant_paths, key=lambda path: sum(cost_matrix[path[i - 1]][path[i]] for i in range(num_cities)))[:num_ants // 10])
        
        # Changing values of pheromones on the paths
        pheromone_matrix = update_pheromones(pheromone_matrix, ant_paths, cost_matrix, Q)
        pheromone_matrix = evaporate_pheromones(pheromone_matrix, evaporation_rate)

        # Find the best path in this iteration
        iteration_best_path = min(ant_paths, key=lambda path: sum(cost_matrix[path[i - 1]][path[i]] for i in range(num_cities)))
        iteration_best_cost = sum(cost_matrix[iteration_best_path[i - 1]][iteration_best_path[i]] for i in range(num_cities))

        if iteration_best_cost < best_cost:
            best_cost = iteration_best_cost
            best_path = iteration_best_path

        # Calculate average cost
        iteration_average_cost = sum(sum(cost_matrix[path[i - 1]][path[i]] for i in range(num_cities)) for path in ant_paths) / num_ants
        average_cost += iteration_average_cost

        # Update worst path and cost if found in this iteration
        iteration_worst_path = max(ant_paths, key=lambda path: sum(cost_matrix[path[i - 1]][path[i]] for i in range(num_cities)))
        iteration_worst_cost = sum(cost_matrix[iteration_worst_path[i - 1]][iteration_worst_path[i]] for i in range(num_cities))
        if iteration_worst_cost > worst_cost:
            worst_cost = iteration_worst_cost
            worst_path = iteration_worst_path

    # Include elitist ants' paths in the calculation if they were used
    if use_elitist_ants:
        ant_paths.extend(elitist_paths)

    average_cost /= num_iterations

    return best_path, best_cost, average_cost, worst_path, worst_cost





# Parameters
num_ants = 10
# Termination criteria
num_iterations = 1000
# Impact of pheromone trails on the ant's decision
alpha = 1.0
# Effect of heuristic information
beta = 2.0
# Pheromone evaporation
evaporation_rate = 0.5
# Amount of pheromone to deposit on the edges of the paths (Q)
pheromone_deposit = 1

# Elitest ant switch
use_elitist = False
best_path, best_cost, average_cost, worst_path, worst_cost = ACO_algorithm(num_ants, num_iterations, cost_matrix, alpha, beta, evaporation_rate, pheromone_deposit)

print("Best path:", best_path)
print("Best cost:", best_cost)
print("Average cost:", average_cost)
print("Worst path:", worst_path)
print("Worst cost:", worst_cost)





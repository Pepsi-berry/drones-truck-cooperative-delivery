import random
from math import exp, sqrt
from copy import copy
from .drone_assignment_routing_heuristic import drone_assignment_routing

# give a truck route and return three neighbors of the given route
def neighborhood_search_sa(solution_0, nodes, distances):
    flat_nodes = [
        n 
        for i in range(1, 3)
        for n in nodes[i]
    ]
    solution_1 = copy(solution_0)
    
    # generate two random indices to be swapped
    i_stop = random.randint(1, len(solution_1) - 2) # - 2?
    i_location = random.randint(0, len(flat_nodes) - 1) # - 1?
    
    # node_truck should be included in truck route
    if solution_1[i_stop] in nodes[1]:
        j_stop = random.randint(1, len(solution_1) - 2)
        # swap i and j in truck route
        solution_1[i_stop], solution_1[j_stop] = solution_1[j_stop], solution_1[i_stop]
    else:
        if flat_nodes[i_location] in solution_1: 
            j_stop = solution_1.index(flat_nodes[i_location])
            solution_1[i_stop], solution_1[j_stop] = solution_1[j_stop], solution_1[i_stop]
        else:
            solution_1[i_stop] = flat_nodes[i_location]
    
    solution_2 = [nodes[0]]
    tmp_solution = copy(solution_1[1:-1])
    # print(solution_2, tmp_solution)
    # print(distances)
    
    for i in range(len(solution_1) - 2):
        j_star = -1
        j_value = 2**24
        for j in tmp_solution:
            if distances[solution_2[i]][j] < j_value: 
                j_star = j
                j_value = distances[solution_2[i]][j]
        # print(j_star)
        solution_2.append(j_star)
        tmp_solution.remove(j_star)
    
    solution_2.append(nodes[-1])
    
    # solution_3 == reversed solution_2
    solution_3 = solution_2[::-1]
    solution_3[0], solution_3[-1] = solution_3[-1], solution_3[0]
    
    return solution_1, solution_2, solution_3
    

def compute_solution_makespan(drone_assign_routing, lower_solution, not_found, travel_time_truck, travel_time_uav):
    if not_found:
        return 1e16
    
    # print(travel_time_truck, travel_time_uav)
    # print(drone_assign_routing)
    makespan = 0
    for i in range(len(lower_solution) - 1):
        makespan += max([travel_time_truck[lower_solution[i]]] + [travel_time_uav[sortie] for sortie in drone_assign_routing if sortie[0] == lower_solution[i]])
        
    return makespan


# the initial solution contains customer node only
# random policy
# locations will be needed when optimize generating policy
def generate_initial_routing(nodes, locations, S_hat):
    if S_hat < len(nodes[1]):
        raise ValueError("The truck route must contains all the truck customer locations, Check the S_hat setting.")
    # to ensure that the initial route contains the truck customer locations
    initial_route = copy(nodes[1])
    # add the insufficient locations from non-truck-must customer locations
    initial_route += random.sample(nodes[2], S_hat - len(nodes[1]))
    random.shuffle(initial_route)
    
    initial_route = [nodes[0]] + initial_route + [nodes[-1]]
    
    return initial_route
    

# in SA phase, there is no flexible location exist in truck route
def assignment_routing_SA(nodes, 
                          locations, 
                          distances_truck, 
                          distances_uav, 
                          uavs, 
                          uav_velocity, 
                          uav_range, 
                          truck_velocity, 
                          S_max, 
                          T_max, 
                          T_min, 
                          K, 
                          iter_max):
    # travel_times_t
    # parameters initialization
    T_curr = T_max

    curr_solution = generate_initial_routing(nodes, locations, S_max)
    best_solution = copy(curr_solution)
    # print(best_solution)
    curr_value = compute_solution_makespan(*(drone_assignment_routing(best_solution, 
                                                                      nodes, 
                                                                      distances_truck, 
                                                                      truck_velocity, 
                                                                      distances_uav, 
                                                                      uavs, 
                                                                      uav_velocity, 
                                                                      uav_range)))
    best_value = curr_value
    
    # SA main iteration
    while T_curr > T_min:
        for _ in range(iter_max):
            solution_prime = None
            value_prime = 1e20
            
            # find the optimal solution neighbor
            for neigh in neighborhood_search_sa(curr_solution, nodes, distances_truck):
                neigh_value = compute_solution_makespan(*(drone_assignment_routing(neigh, 
                                                                                   nodes, 
                                                                                   distances_truck, 
                                                                                   truck_velocity, 
                                                                                   distances_uav, 
                                                                                   uavs, 
                                                                                   uav_velocity, 
                                                                                   uav_range)))
                if neigh_value < value_prime:
                    solution_prime = neigh
                    value_prime = neigh_value
            
            if value_prime < best_value: 
                best_solution = solution_prime
                best_value = value_prime
            
            if value_prime <= curr_value:
                curr_solution = solution_prime
                curr_value = value_prime
            # prevent local optimal solution
            else:
                rm = random.random()
                # if rm <= exp((((value_prime - curr_value) / value_prime) * 100) / T_curr): 
                # there is a suspected error in the formula of the paper here
                if rm <= exp((((curr_value - value_prime) / value_prime) * 100) / T_curr): 
                    curr_solution = solution_prime
                    curr_value = value_prime
        
        T_curr *= K # cooling down
        
    nodes_U_truck = [node for node in nodes[2] if node in best_solution]
    nodes_U_drone = list( set(nodes[2]) - set(nodes_U_truck) )
    
    return best_solution, best_value, nodes_U_truck, nodes_U_drone


if __name__ == "__main__":
    # parameters initialization
    num_customer = 20
    num_customer_truck = int(num_customer * 0.2)
    num_flexible_location = num_customer * 1
    
    uavs = [0, 0, 1, 1] # uav_no -> uav_type
    uav_velocity = [12, 12] # uav_type -> uav_v
    uav_range = [1000, 1000] # uav_type -> uav_r
    
    truck_velocity = 8
    
    nodes = [
        0, 
        list(range(1, num_customer_truck + 1)), 
        list(range(num_customer_truck + 1, num_customer + 1)), 
        list(range(num_customer + 1, num_customer + num_flexible_location + 1)), 
        num_customer + num_flexible_location + 1
    ]
    locations = [(500, 500)] + [(random.randint(0, 1_000), random.randint(0, 1_000)) for _ in range(num_customer + num_flexible_location)] + [(500, 500)]
    
    # distance matrix calculation
    # respectively manhattan distance and euclidean distance
    distances_truck = [
        [sum(abs(loc[k] - lo[k]) for k in range(2)) for lo in locations]
        for loc in locations
    ]
    distances_uav = {
        i_loc: [int(sqrt(sum((locations[i_loc][k] - lo[k])**2 for k in range(2)))) for lo in locations]
        for i_loc in nodes[2]
    }
    
    S_hat = int(num_customer * 0.5)
    T_max = 660
    T_min = 1e-2
    K = 0.94
    iter_max = 22
    
    # list_r = [nodes[0]] + random.sample(range(1, num_customer + 1), S_hat) + [nodes[-1]]
    # print(list_r)
    
    # print(neighborhood_search_sa(list_r, nodes, distances_truck))
    
    B_star, v_star, n_u_0, n_u_1 = assignment_routing_SA(nodes, 
                                                         locations, 
                                                         distances_truck, 
                                                         distances_uav, 
                                                         uavs, 
                                                         uav_velocity, 
                                                         uav_range, 
                                                         truck_velocity, 
                                                         S_hat, T_max, T_min, K, iter_max)
    
    print(B_star, v_star, n_u_0, n_u_1)
    
    
    # import matplotlib.pyplot as plt
    # route = [locations[i] for i in B_star]
    # x_values, y_values = zip(*locations[:num_customer + 1])
    # x_r, y_r = zip(*route)

    # plt.figure(figsize=(8, 6))

    # plt.scatter(x_values, y_values, color='blue')
    # plt.plot(x_r, y_r, color='red', marker='o', label='B Path')

    # for i, (x, y) in enumerate(locations[:num_customer + 1]):
    #     plt.text(x, y, f'{i}', fontsize=12, ha='right')

    # plt.xlabel('X')
    # plt.ylabel('Y')

    # plt.grid(True)

    # plt.show()
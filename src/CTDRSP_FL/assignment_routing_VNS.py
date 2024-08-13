import random
from copy import copy
from .drone_assignment_routing_heuristic import drone_assignment_routing
from .assignment_routing_SA import compute_solution_makespan

# give a truck route and return three neighbors of the given route()
# specifiable search space for vns
def neighborhood_search_vns(solution_0, nodes, distances, searching_space=[1, 3]):
    searching_nodes = [
        n 
        for i in range(searching_space[0], searching_space[1])
        for n in nodes[i]
    ]
    solution_1 = copy(solution_0)
    
    # generate two random indices to be swapped
    i_stop = random.randint(1, len(solution_1) - 2)
    i_location = random.randint(0, len(searching_nodes) - 1)
    
    # node_truck should be included in truck route
    if solution_1[i_stop] in nodes[1]:
        j_stop = random.randint(1, len(solution_1) - 2)
        # swap i and j in truck route
        solution_1[i_stop], solution_1[j_stop] = solution_1[j_stop], solution_1[i_stop]
    else:
        if searching_nodes[i_location] in solution_1: 
            j_stop = solution_1.index(searching_nodes[i_location])
            solution_1[i_stop], solution_1[j_stop] = solution_1[j_stop], solution_1[i_stop]
        else:
            solution_1[i_stop] = searching_nodes[i_location]
    
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


# extend the neighborhood searching space to flexible customer locations in vns
# input the solution obtained in SA phase and return furtherly optimized solution
def assignment_routing_VNS(SA_solution, 
                           SA_value, 
                           nodes, 
                           distances_truck, 
                           truck_velocity, 
                           distances_uav, 
                           uavs, 
                           uav_velocity, 
                           uav_range, 
                           iter_max_inner, iter_max_outer):
    best_solution = copy(SA_solution)
    best_value = copy(SA_value)
    
    for _ in range(iter_max_outer):
        # get the searching neighborhood in outer iteration
        neighborhood = neighborhood_search_vns(best_solution, nodes, distances_truck, searching_space=[1, 4])[0]
        for _ in range(iter_max_inner):
            # find the best solution and its value among the 6 neighbors
            solution_prime = None
            value_prime = 1e20
            
            for nei in neighborhood_search_vns(neighborhood, nodes, distances_truck) + neighborhood_search_vns(neighborhood, nodes, distances_truck, searching_space=[3, 4]):
                nei_value = compute_solution_makespan(*(drone_assignment_routing(
                    nei, 
                    nodes, 
                    distances_truck, 
                    truck_velocity, 
                    distances_uav, 
                    uavs, 
                    uav_velocity, 
                    uav_range
                )))
                if nei_value < value_prime:
                    solution_prime = nei
                    value_prime = nei_value
            
            if value_prime <= best_value:
                best_solution = solution_prime
                best_value = value_prime
                
    nodes_U_drone = list( set(nodes[2]) - set(best_solution) )
    nodes_U_truck = list( set(nodes[2]) - set(nodes_U_drone) )
        
    return best_solution, best_value, nodes_U_truck, nodes_U_drone


if __name__ == "__main__":
    # parameters initialization
    num_customer = 10
    num_customer_truck = int(num_customer * 0.2)
    num_flexible_location = num_customer * 1
    
    iter_max_inner = 30
    iter_max_outer = 15
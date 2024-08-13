import random
from copy import copy
from math import sqrt

def compute_travel_time(solution, feasible_sorties, truck_distances, uav_distances, truck_velocity, uavs, uav_velocity):
    # truck travel time from a stop in current solution(except returning depot) to next stop
    truck_travel_time = {
        solution[i_stop]: truck_distances[solution[i_stop]][solution[i_stop + 1]] / truck_velocity
        for i_stop in range(len(solution) - 1)
    }
    
    # sortie -> travel_time
    uav_travel_time = {
        (i_stop, cust, uav_no): (uav_distances[cust][i_stop] + uav_distances[cust][i_stop + 1]) / uav_velocity[uavs[uav_no]]
        for i_stop, cust, uav_no in feasible_sorties
    }
    # uav_travel_time = [
    #     (uav_distances[cust][i_stop] + uav_distances[cust][i_stop + 1]) / uav_velocity[uavs[uav_no]]
    #     for i_stop in range(len(solution) - 1)
    #     for cust in nodes_U_drone
    #     for uav_no in range(len(uavs))
    # ]
    
    return truck_travel_time, uav_travel_time


# consider only the uav_range constraint without other factor, etc. uav capacity.
def build_feasible_sorties(nodes_U_drone, truck_route, distances, uavs, uav_range):
    return [
        (truck_route[i_start], cust, uav_no) 
        for cust in nodes_U_drone
        for i_start in range(len(truck_route) - 1) # returning point could not launch uav
        for uav_no in range(len(uavs))
        if distances[cust][truck_route[i_start]] + distances[cust][truck_route[i_start + 1]] < uav_range[uavs[uav_no]]
    ]


def remove_corresponding_sorties(selected_sortie, feasible_sorties):
    return [
        st for st in feasible_sorties
        if st[1] != selected_sortie[1] and ((st[0], st[-1]) != (selected_sortie[0], selected_sortie[-1]))
    ]


def drone_assignment_routing(lower_solution, nodes, truck_distances, truck_velocity, uav_distances, uavs, uav_velocity, uav_range):
    # get the needed parameters
    nodes_U_drone = list( set(nodes[2]) - set(lower_solution) )
    # print(nodes_U_drone)
    
    feasible_sorties = build_feasible_sorties(
        nodes_U_drone, 
        lower_solution, 
        uav_distances, 
        uavs, 
        uav_range)
    # print(feasible_sorties)
    truck_travel_time, uav_travel_time = compute_travel_time(
        lower_solution, 
        feasible_sorties, 
        truck_distances, 
        uav_distances, 
        truck_velocity, 
        uavs, 
        uav_velocity)
    
    drone_assignment = []
    remaining_customer_locations = copy(nodes_U_drone)
    not_found = False
    
    # assign sorties without truck waiting
    while feasible_sorties:
        no_zero_delay_sortie = True
        for st in feasible_sorties:
            if truck_travel_time[st[0]] >= uav_travel_time[st]:
                no_zero_delay_sortie = False
                zero_delay_sorties = [
                    zds for zds in feasible_sorties if zds[1] == st[1] and truck_travel_time[zds[0]] >= uav_travel_time[zds]
                ]
                
                if len(zero_delay_sorties) == 1:
                    # only one sortie without truck stop
                    # add to result and remove corresponding sorties
                    drone_assignment.append(zero_delay_sorties[0])
                    feasible_sorties = remove_corresponding_sorties(zero_delay_sorties[0], feasible_sorties)
                    # remove assigned customer location
                    remaining_customer_locations.remove(zero_delay_sorties[0][1])
                else:
                    # find the sortie such that the time delay between the truck and drone arrival at truck stop is minimized
                    # print("List: ", [truck_travel_time[zds[0]] - uav_travel_time[zds] for zds in zero_delay_sorties])
                    least_delayed_sortie = min(zero_delay_sorties, key=lambda zd_sortie: truck_travel_time[zd_sortie[0]] - uav_travel_time[zd_sortie])
                    # print(least_delayed_sortie, ": ", truck_travel_time[least_delayed_sortie[0]] - uav_travel_time[least_delayed_sortie])
                    # then add to result and remove corresponding sorties
                    drone_assignment.append(least_delayed_sortie)
                    feasible_sorties = remove_corresponding_sorties(least_delayed_sortie, feasible_sorties)
                    # remove assigned customer location
                    remaining_customer_locations.remove(least_delayed_sortie[1])
                    
                break
        
        if no_zero_delay_sortie:
            break
    
    # assign the rest of the customer locations
    for remaining_customer in remaining_customer_locations:
        delayed_sorties = [
            zds for zds in feasible_sorties if zds[1] == remaining_customer
        ]
        
        if len(delayed_sorties) == 0:
            not_found = True
            break
        elif len(delayed_sorties) == 1:
            # only one feasible sortie
            # add to result and remove corresponding sorties
            drone_assignment.append(delayed_sorties[0])
            feasible_sorties = remove_corresponding_sorties(delayed_sorties[0], feasible_sorties)
        else:
            # find the sortie such that the time delay between the truck and drone arrival at truck stop is minimized
            least_delayed_sortie = min(delayed_sorties, key=lambda d_sortie: uav_travel_time[d_sortie] - truck_travel_time[d_sortie[0]])
            # then add to result and remove corresponding sorties
            drone_assignment.append(least_delayed_sortie)
            feasible_sorties = remove_corresponding_sorties(least_delayed_sortie, feasible_sorties)
    
    return drone_assignment, lower_solution, not_found, truck_travel_time, uav_travel_time


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


if __name__ == "__main__":
    # parameters initialization
    uavs = [0, 0, 1, 1] # uav_no -> uav_type
    uav_velocity = [12, 12] # uav_type -> uav_v
    uav_range = [1000, 1000] # uav_type -> uav_r
    
    truck_velocity = 8
    
    num_customer = 10
    num_customer_truck = int(num_customer * 0.2)
    num_flexible_location = num_customer * 1
    
    nodes = [
        0, 
        list(range(1, num_customer_truck + 1)), 
        list(range(num_customer_truck + 1, num_customer + 1)), 
        list(range(num_customer + 1, num_customer + num_flexible_location + 1)), 
        num_customer + num_flexible_location + 1
    ]
    locations = [(500, 500)] + [(random.randint(0, 1_000), random.randint(0, 1_000)) for _ in range(num_customer + num_flexible_location)] + [(500, 500)]
    # locations = [(500, 500), (841, 873), (736, 129), (366, 671), (833, 882), (814, 526), (663, 653), (228, 720), (667, 21), (913, 783), (999, 502), (770, 170), (240, 207), (5, 563), (445, 272), (184, 266), (113, 112), (983, 299), (712, 63), (215, 536), (961, 697), (500, 500)]
    distances_truck = [
        [sum(abs(loc[k] - lo[k]) for k in range(2)) for lo in locations]
        for loc in locations
    ]
    distances_uav = {
        i_loc: [int(sqrt(sum((locations[i_loc][k] - lo[k])**2 for k in range(2)))) for lo in locations]
        for i_loc in nodes[2]
    }
    l_solution = generate_initial_routing(nodes, locations, S_hat=5)
    # l_solution = [0, 8, 5, 2, 3, 1, 21]
    print(locations)
    
    uav_assign, l_solution, n_f, _, _ = drone_assignment_routing(l_solution, nodes, distances_truck, truck_velocity, distances_uav, uavs, uav_velocity, uav_range)
    print(l_solution, uav_assign, n_f)
    
    
    import matplotlib.pyplot as plt
    route = [locations[i] for i in l_solution]
    x_values, y_values = zip(*locations[:num_customer + 1])
    x_r, y_r = zip(*route)

    plt.figure(figsize=(8, 6))

    plt.scatter(x_values, y_values, color='blue')
    plt.plot(x_r, y_r, color='red', marker='o', label='B Path')

    for i, (x, y) in enumerate(locations[:num_customer + 1]):
        plt.text(x, y, f'{i}', fontsize=12, ha='right')

    plt.xlabel('X')
    plt.ylabel('Y')

    # 显示网格
    plt.grid(True)

    # 显示图形
    plt.show()
    
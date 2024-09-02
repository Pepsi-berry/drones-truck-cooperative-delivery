import gurobipy as gp
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import GRB
from copy import copy


def k_means_cluster(nodes, locations, uav_range, SIZE, K_min, K_max, G_max, max_it_outer=400, max_iter=200):
    # initaliza parameters
    K_curr = K_min
    depot = locations[nodes[0]]
    points = np.array(locations[nodes[2]])
    num_points = len(nodes[2])
    
    # calculate only one cluster except depot as the iteration initialization
    focal_points = np.array([depot] + [(np.random.randint(0, SIZE), np.random.randint(0, SIZE)) for _ in range(K_curr)])
    distances = np.linalg.norm(points[:, np.newaxis] - focal_points, axis=2)
    assign = np.argmin(distances, axis=1)
    
    it = 0
    while not all(distances[cust][assign[cust]] <= uav_range for cust in range(num_points)) or not np.all(np.bincount(assign) < G_max):
        it += 1
        if it >= max_it_outer:
            return focal_points, assign
        # return when constraints are met
        if K_curr < K_max:
            K_curr += 1
        else:
            K_curr = K_min
        
        # K_curr += 1
        ite = 1
        # randomly generate initial clustering focal points
        focal_points = np.array([depot] + [(np.random.randint(0, SIZE), np.random.randint(0, SIZE)) for _ in range(K_curr)])
        new_focal_points = SIZE - focal_points
        
        while ite <= max_iter and not np.all(focal_points == new_focal_points):
            focal_points = copy(new_focal_points)
            distances = np.linalg.norm(points[:, np.newaxis] - focal_points, axis=2)
            assign = np.argmin(distances, axis=1)
            
            ite += 1
            
            # update focal points
            for i in range(K_curr):
                if np.any(assign == i + 1):
                    new_focal_points[i + 1] = points[assign == i + 1].mean(axis=0)
        
        focal_points = copy(new_focal_points)
        # print(focal_points, assign)
    
    # print("Not Found Feasible Warm Start")
    return focal_points, assign


def solve_base_TSP(locations):
    dist_matrix = np.sqrt(np.sum((locations[:, np.newaxis] - locations[np.newaxis, :]) ** 2, axis=2))
    n = locations.shape[0]
    
    model = gp.Model("TSP")
    model.setParam('OutputFlag', 0) # eliminate the gurobi logs output to console

    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
    u = model.addVars(n, vtype=GRB.CONTINUOUS, name="u")

    # objective setting
    model.setObjective(gp.quicksum(dist_matrix[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j), GRB.MINIMIZE)

    # add constraints
    for j in range(n):
        model.addConstr(gp.quicksum(x[i, j] for i in range(n) if i != j) == 1, name=f"enter_{j}")

    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n) if j != i) == 1, name=f"leave_{i}")

    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                model.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1, name=f"mtz_{i}_{j}")

    # solve
    model.optimize()
    
    # if model.status == GRB.OPTIMAL:
    #     print("Optimal tour found:")
    #     for i in range(n):
    #         for j in range(n):
    #             if x[i, j].X > 0.5:
    #                 print(f"City {i} -> City {j}")
    #     print(f"Minimum tour length: {model.ObjVal}")
    # else:
    #     print("No optimal solution found.")
        
    return np.array([[x[i, j].X for j in range(n)] for i in range(n)])


def solve_warm_start_solution(nodes, locations, uav_range, SIZE, K_min, K_max, G_max, max_iter=200):
    K_min_prime = K_min
    # garentee a feasible solution for a feasible MIP cluster number
    for _ in range(200):
        # get k_means initial cluster
        focal_points, labels = k_means_cluster(
            nodes, locations, uav_range, SIZE, K_min_prime, K_max, G_max, max_iter
        )
        # print(focal_points)
        # print(labels)
        locations_D = locations[nodes[2]]
        locations_T = locations[nodes[1]]
        
        assignments_T = np.zeros([locations_T.shape[0], focal_points.shape[0]])
        assignments_D = np.zeros([locations_D.shape[0], focal_points.shape[0]])
        
        assignments_D[np.arange(locations_D.shape[0]), labels] = 1
        
        # calculate the distance from the l in l_D to the initial cluster focal points, 
        dist_to_center = np.linalg.norm(locations_T[:, np.newaxis] - focal_points, axis=2)
        # get the index array in descending order of distance
        sorted_dist_indices = np.array(np.unravel_index(np.argsort(dist_to_center, axis=None), dist_to_center.shape)).T
        
        # transfer the cluster focal points to the truck points considering the range constraint
        T_focals = set()
        replaced_focals = [ 0 ]
        for i_hat, k_hat in sorted_dist_indices:
            # do the same for focal point k?
            if i_hat not in T_focals and k_hat not in replaced_focals:
                cluster_points = locations_D[labels == k_hat]
                dist_to_T = np.linalg.norm(cluster_points[:, np.newaxis] - locations_T[i_hat], axis=1)
                
                if np.all(dist_to_T < uav_range):
                    # print("move from ", focal_points[k_hat], " to ", locations_T[i_hat])
                    focal_points[k_hat] = locations_T[i_hat]
                    assignments_T[i_hat, k_hat] = 1
                    # prevent multiple cluster center coordinate points from appearing with the same coordinates
                    T_focals.add(i_hat)
                    replaced_focals.append(k_hat)
        
        if len(T_focals) == len(nodes[1]):
            # print("cluster num: ", focal_points.shape[0])
            break
        # else:
        #     K_min_prime = np.random.randint(K_min, K_max)
    
    assignments = np.concatenate((assignments_T, assignments_D), axis=0)
    tsp_route = solve_base_TSP(focal_points)
    
    # get the assign and return
    # print(focal_points)
    return focal_points, assignments, tsp_route


def solve_JOCR_U(nodes, locations, uav_range, uav_velocity, truck_velocity, SIZE, K_min, K_max, G_max, max_iter=200):
    # prepare the parameters
    # locations_T = locations[nodes[1]]
    # locations_D = locations[nodes[2]]
    A = locations[nodes[1] + nodes[2]][:, 0]
    B = locations[nodes[1] + nodes[2]][:, 1]
    num_customers = len(nodes[1] + nodes[2])
    F_l = [0] * len(nodes[1]) + [uav_range] * len(nodes[2])
    # linearization parameter
    P = 6
    theta = 0.2831
    M = 2**16
    
    K_min_p = K_min
    for _ in range(20):
        # get warm start solution
        focal_points_ws, assignments_ws, route_ws = solve_warm_start_solution(
            nodes, locations, uav_range, SIZE, K_min_p, K_max, G_max, max_iter
        )
        num_clusters = route_ws.shape[0]
        # print(num_clusters)
        
        clusters = np.arange(num_clusters)
        
        # to modify route as 0 -> 1 -> 2 -> N
        rearrange_idx = [ 0 ]
        curr_idx = 0
        for _ in range(num_clusters - 1):
            curr_idx = np.argmax(route_ws[curr_idx])
            rearrange_idx.append(curr_idx)
        
        # rearrange warm start solution
        focal_points_ws_rearrange = focal_points_ws[rearrange_idx]
        assignments_ws_rearrange = assignments_ws[:, rearrange_idx]
        route_ws_rearrange = np.zeros_like(route_ws)
        for i in range(num_clusters):
            route_ws_rearrange[i, (i + 1) % num_clusters] = 1
        
        model = gp.Model("JOCR_U")
        model.setParam('OutputFlag', 0) # eliminate the gurobi logs output to console

        # add variables
        # focal point coordinations, x axis and y axis(a, b)
        a = model.addVars(num_clusters, lb=0, ub=SIZE, vtype=GRB.CONTINUOUS, name="a")
        b = model.addVars(num_clusters, lb=0, ub=SIZE, vtype=GRB.CONTINUOUS, name="b")
        # Euclidean distance
        d_E = model.addVars(num_customers, num_clusters, vtype=GRB.CONTINUOUS, name="d_E")
        d_E_x = model.addVars(num_customers, num_clusters, vtype=GRB.CONTINUOUS, name="d_E_x")
        d_E_y = model.addVars(num_customers, num_clusters, vtype=GRB.CONTINUOUS, name="d_E_y")
        # truck distance
        d_R = model.addVars(num_clusters, num_clusters, vtype=GRB.CONTINUOUS, name="d_R")
        d_R_x = model.addVars(num_clusters, num_clusters, vtype=GRB.CONTINUOUS, name="d_R_x")
        d_R_y = model.addVars(num_clusters, num_clusters, vtype=GRB.CONTINUOUS, name="d_R_y")
        # customer assignment
        x = model.addVars(num_customers, num_clusters, vtype=GRB.BINARY, name="x")
        # drone customer assignment
        q = model.addVars(num_customers, num_clusters, vtype=GRB.BINARY, name="q")
        
        # # truck route between clusters(fixed for linearization objective func)
        # y = model.addVars(num_clusters, num_clusters, vtype=GRB.BINARY, name="y")
        
        # max linearization t variable for min_time objective
        t = model.addVars(num_clusters, lb=0, vtype=GRB.CONTINUOUS, name="t")
        # num uav limit
        g = model.addVar(vtype=GRB.INTEGER, name="g")
        # u = model.addVars(n, vtype=GRB.CONTINUOUS, name="u")

        # objective setting (min makespan)
        model.setObjective(
            sum(t) + gp.quicksum(route_ws_rearrange[k_1, k_2] * (d_R[k_1, k_2] / truck_velocity) 
                                        for k_1 in clusters
                                        for k_2 in clusters), 
            GRB.MINIMIZE)

        # add constraints
        for l in range(num_customers):
            model.addConstr(gp.quicksum(x[l, k] for k in range(num_clusters)) == 1, name=f"assign_{l}")
        
        for l in range(num_customers):
            for k in range(num_clusters):
                model.addConstr(d_E_x[l, k] >= A[l] - a[k], name=f"d_E_x_greater_{l}_{k}")
        for l in range(num_customers):
            for k in range(num_clusters):
                model.addConstr(d_E_x[l, k] >= a[k] - A[l], name=f"d_E_x_less_{l}_{k}")
        for l in range(num_customers):
            for k in range(num_clusters):
                model.addConstr(d_E_y[l, k] >= B[l] - b[k], name=f"d_E_y_greater_{l}_{k}")
        for l in range(num_customers):
            for k in range(num_clusters):
                model.addConstr(d_E_y[l, k] >= b[k] - B[l], name=f"d_E_y_less_{l}_{k}")
        for l in range(num_customers):
            for k in range(num_clusters):
                for p in range(P):
                    model.addConstr(d_E[l, k] >= d_E_x[l, k] * np.cos(p*theta) + d_E_y[l, k] * np.sin(p*theta) - (1 - x[l, k]) * M, 
                                    name=f"d_E_{l}_{k}_{p}")
        
        for l in range(num_customers):
            for k in range(num_clusters):
                model.addConstr(d_E[l, k] <= F_l[l] * q[l, k], name=f"truck_customer_visit_{l}_{k}")
                
        for k in range(num_clusters):
            model.addConstr(gp.quicksum(q[l, k] for l in range(num_customers)) <= g, name=f"cluster_limit_{k}")
        model.addConstr(g <= G_max, name=f"num_uav_limit")
        
        for k in range(num_clusters):
            for k_p in range(num_clusters):
                model.addConstr(d_R_x[k, k_p] >= a[k] - a[k_p], name=f"d_R_x_greater_{k}_{k_p}")
        for k in range(num_clusters):
            for k_p in range(num_clusters):
                model.addConstr(d_R_x[k, k_p] >= a[k_p] - a[k], name=f"d_R_x_less_{k}_{k_p}")
        for k in range(num_clusters):
            for k_p in range(num_clusters):
                model.addConstr(d_R_y[k, k_p] >= b[k] - b[k_p], name=f"d_R_y_greater_{k}_{k_p}")
        for k in range(num_clusters):
            for k_p in range(num_clusters):
                model.addConstr(d_R_y[k, k_p] >= b[k_p] - b[k], name=f"d_R_y_less_{k}_{k_p}")
        for k in range(num_clusters):
            for k_p in range(num_clusters):
                model.addConstr(d_R[k, k_p] == d_R_x[k, k_p] + d_R_y[k, k_p], name=f"d_R_{k}_{k_p}")
                
        # 0 here means cluster[0]
        model.addConstr(a[0] == locations[0][0], name=f"depot_cluster_a")
        model.addConstr(b[0] == locations[0][1], name=f"depot_cluster_b")
        
        # service time is ignored here
        for l in range(num_customers):
            for k in range(num_clusters):
                model.addConstr(t[k] >= 2 * (d_E[l, k] / uav_velocity), name=f"uav_time_{l}_{k}")

        # for i in range(n):
        #     model.addConstr(gp.quicksum(x[i, j] for j in range(n) if j != i) == 1, name=f"leave_{i}")

        # for i in range(1, n):
        #     for j in range(1, n):
        #         if i != j:
        #             model.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1, name=f"mtz_{i}_{j}")

        # set warm start
        for k in range(num_clusters): 
            a[k].Start = focal_points_ws_rearrange[k][0]
            b[k].Start = focal_points_ws_rearrange[k][1]
        
        for l in range(num_customers):
            for k in range(num_clusters):
                x[l, k].Start = assignments_ws_rearrange[l, k]
        
        # solve
        model.update()
        model.setParam('TimeLimit', 120)
        model.optimize()
        
        # post process
        if model.SolCount > 0:
            # model.write("model.lp")
            focal_points_star = [(a[i].X, b[i].X) for i in clusters]
            assignments_star = np.array([
                [ x[l, k].X for k in range(num_clusters) ] 
                for l in range(num_customers)
            ])
            labels_star = [ np.argmax(assignments_star[l]) for l in range(num_customers) ]
            
            # print(focal_points_star, labels_star)
            return True, focal_points_star, labels_star
        else:
            K_min_p = num_clusters + 1
            # model.computeIIS()
            # model.write("model.ilp")
            
            # print("Not Found Feasible Solution")
            # return False, focal_points_ws_rearrange, assignments_ws
        
    print("Not Found Feasible Solution")
    return False, focal_points_ws_rearrange, assignments_ws


# reason for infeasible: max number of cluster insuitable
# (i.e. Assuming that there are 4 l_T, when solving the warm start solution, 
# the number of clusters that may be obtained is 5——any number less than K_max 
# (the initial cluster of k-means does not consider the constraint that l_T needs to be the 
# focal point), because when solving the MIP model, 
# the number of clusters by which The model is defined is obtained directly according to the warm start. 
# Except for l_T, there is only one focal point left. Since the customer points are randomly arranged, 
# there is a high possibility that no feasible solution can be obtained.)
if __name__ == "__main__":
    # parameters initialization
    num_customer = 20
    num_customer_truck = int(num_customer * 0.2)
    
    uavs = [0, 0, 1, 1] # uav_no -> uav_type
    uav_velocity = [12, 12] # uav_type -> uav_v
    uav_range = [1_000, 1_000] # uav_type -> uav_r
    
    truck_velocity = 8
    
    nodes = [
        0, 
        list(range(1, num_customer_truck + 1)), 
        list(range(num_customer_truck + 1, num_customer + 1)), 
    ]
    SIZE = 5_000
    # locations = np.array([(SIZE / 2,SIZE / 2)] + [(np.random.randint(0, SIZE), np.random.randint(0, SIZE)) for _ in range(num_customer)])
    # locations = np.array([[500.0, 500.0], [235.0, 86.0], [671.0, 897.0], [25.0, 672.0], [225.0, 614.0], [180.0, 847.0], [563.0, 184.0], [594.0, 879.0], [199.0, 947.0], [578.0, 621.0], [352.0, 904.0], [375.0, 770.0], [513.0, 693.0], [838.0, 810.0], [308.0, 331.0], [778.0, 977.0], [111.0, 628.0], [195.0, 180.0], [837.0, 254.0], [844.0, 638.0], [981.0, 768.0]])
    # print(locations.tolist())
    
    G_max = 6
    coef_k = 7
    K_min = int(num_customer / G_max) + 1
    K_max = int(num_customer / G_max) + coef_k
    max_iter = 200
    
    cnt = 0
    for _ in range(1):
        locations = np.array([(SIZE / 2,SIZE / 2)] + [(np.random.randint(0, SIZE), np.random.randint(0, SIZE)) for _ in range(num_customer)])
        found_solution, fp, ass = solve_JOCR_U(nodes, locations, uav_range[0], uav_velocity[0], truck_velocity, SIZE, K_min, K_max, G_max, max_iter)
        # if found_solution == True:
        #     cnt += 1
    # print(cnt)
    print(fp, ass)
    
    # solve_base_TSP(np.random.randint(0, 20, [5, 2]))
    # print(locations[nodes[1]])
    # fp, ass, route = solve_warm_start_solution(nodes, locations, uav_range[0], SIZE, K_max, G_max, max_iter)
    # print(fp, ass, route)
    # rearrange_idx = [ 0 ]
    # curr_idx = 0
    # for _ in range(route.shape[0] - 1):
    #     curr_idx = np.argmax(route[curr_idx])
    #     rearrange_idx.append(curr_idx)
    # # rearrange_idx = [ np.argmax(route[:, column_i]) for column_i in range(1, route.shape[1]) ]
    # # rearrange_idx.append(np.argmax(route[:, 0]))
    # print(rearrange_idx)
    # fp_rearrange = fp[rearrange_idx]
    # ass_rearrange = ass[:, rearrange_idx]
    # print(fp_rearrange, ass_rearrange)
    
    # for i in range(50):
    # fp, ass = k_means_cluster(nodes, locations, uav_range[0], SIZE, K_max, G_max, max_iter)
    # print(fp, ass)
    # print(locations[nodes[1]])
    
    if found_solution:
        cust = nodes[1] + nodes[2]
        fp = np.array(fp)
        plt.scatter(fp[:, 0], fp[:, 1], c=np.arange(fp.shape[0]))
        for i, (x, y) in enumerate(fp):
            plt.text(x, y, f'{i}', fontsize=12, ha='right')
        plt.scatter(locations[cust][:, 0], locations[cust][:, 1], c=ass)
        plt.show()
        
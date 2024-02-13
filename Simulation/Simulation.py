# %%
import networkx as nx
import pandas as pd
import numpy as np
import pulp
import itertools
import os
import pickle

from tqdm import tqdm

df = pd.read_csv("dataset/finished_dataset.csv")
with open('graph_from_routes_wd.pkl', 'rb') as f:
    graph_from_routes = pickle.load(f)


# %%
df.head()

# %%
graph = graph_from_routes.copy()

# %%
import math
edge_info = {}
l = 50 #km/h#/ 3600#m/h
# Iterate over edges and add distances to the dictionary
for u, v, d in graph.edges(data=True):
    edge_info[(u, v)] = [
        d["length"],
        d["length"] / d["maxspeed"] if not math.isnan(d["maxspeed"]) else d["length"]/l,
        10
    ]

# %%
edge_info = {}
l = 50 #km/h#/ 3600#m/h
# Iterate over edges and add distances to the dictionary
for u, v, d in graph.edges(data=True):
    edge_info[(u, v)] = [
        d["length"]/10,
        d["length"]/500,
        50
    ]

# %%


# %%
edge_info

# %%
def distance_matrix_from_graph(G):
    """
    Create a distance matrix from a networkx graph.
    """
    num_nodes = len(G.nodes)
    distance_matrix = [[0] * num_nodes for _ in range(num_nodes)]
    for u in range(num_nodes):
        for v in range(num_nodes):
            if u != v:
                distance_matrix[u][v] = nx.shortest_path_length(G, source=u, target=v, weight='weight')
    return distance_matrix



# %%
from enum import Enum
class Charger(Enum):
    SLOW = (0.1,00.6)
    MEDIUM = (0.1,00.8)
    FAST = (0.2,00.8)

# %%


class Depot():
    def __init__(self, node_id, type = Charger.SLOW, vcapacity = 50, vehicles_now = 25) -> None:
        self.id = node_id
        self.type = type
        self.r, _ = type.value
        self.vcapacity_ = vcapacity
        self.vehicles_now = vehicles_now


# %%
class Vehicle():
    def __init__(self, id, pcapacity=4,gcapacity = 10, 
                 R_ = 1.5, 
                 ctype = Charger.SLOW,
                 start_depot = 0, 
                 end_depot = 0, 
                 Q = 100,
                 tau = None,
                 ):
        self.id = id
        self.pcapacity_ = pcapacity
        self.gcapacity_ = gcapacity
        self.R_ = R_
        self.charge = 100
        self.start_depot = start_depot
        self.end_depot = end_depot
        self.requests_ ={}
        _,self.theta = ctype.value
        self.Q = Q
        self.tau = tau if tau is not None else 20/(self.R_*self.theta)
    def update_requests(self, id, start, dest):
        self.requests_[id] = (start,dest)
    def update_requests(self, request):
        self.requests_[request.id] = (request.start,request.dest)
    def update_charge(self,distance, speed = 50):
        self.charge -=  distance/speed * self.R_


# %%
class Request():
    def __init__(self, id, people, start, end) -> None:
        self.id = id
        self.people_ = people
        self.start = start
        self.end = end 


# %%
def convert_reqs_to_obj(requests, required_info = ["id", "passenger_count", "pickup_graph_node","dropoff_graph_node"]):
    reqs_obj = {}
    for node, item in requests.items():
        amount = item["amount"]
        temp_reqs = []
        for k in item["internal_ids"]:
            temp_reqs += [
                        Request( *[item[x][k] for x in required_info])
                        ]
        reqs_obj[node] = temp_reqs
    
    return reqs_obj

# %%
def convert_reqs_to_list(requests, required_info = ["id", "passenger_count", "pickup_graph_node","dropoff_graph_node"]):
    temp_reqs = []
    for node, item in requests.items():
        amount = item["amount"]
        for k in item["internal_ids"]:
            temp_reqs += [
                        Request( *[item[x][k] for x in required_info])
                        ]
    return temp_reqs

# %%
def get_requests(dataframe, required_info = ["amount","internal_ids", "id", "passenger_count", "pickup_graph_node","dropoff_graph_node"]):
    requests = {}
    pick_up_nodes = np.unique(dataframe["pickup_graph_node"])
    
    for node in pick_up_nodes:
        temp_dict = dataframe[dataframe["pickup_graph_node"]==node].to_dict()
        temp_dict["amount"] = len(temp_dict["id"])
        temp_dict["internal_ids"] = list(temp_dict["id"].keys())
        requests[node] = {x:temp_dict[x] for x in required_info}
    reqs_obj = convert_reqs_to_obj(requests)
    reqs_list = convert_reqs_to_list(requests)
    return requests, reqs_obj, reqs_list

# %%
def initialize_depots(graph, vehicles_amount = [20]*5,type = [Charger.FAST]*5 ):
    depots = []
    idx = 0
    for node_id, data in graph.nodes(data = True):
        if data["depot"]:
            depots += [Depot(node_id, type = type[idx], vehicles_now = vehicles_amount[idx])]
            idx += 1
    return depots

# %%
def initialize_vehicles_naive(depots):
    vehicles = []
    # Calculate the total number of vehicles at the beginning
    n_vehicles = sum(d.vehicles_now for d in depots)
    vehicle_id_counter = 0  # Start assigning IDs from 1
    for i, d in enumerate(depots):
        start_depot_id = d.id
        end_depot_id = d.id #depots[(i + 1) % len(depots)].id  # Ensures start and end depots are different
        for _ in range(d.vehicles_now):
            vehicles.append(Vehicle(id=vehicle_id_counter,
                                    pcapacity=5,
                                    gcapacity=10,
                                    R_=1,
                                    ctype = d.type,
                                    start_depot=start_depot_id,
                                    end_depot=end_depot_id))
            vehicle_id_counter += 1  # Increment the ID counter
    return vehicles, n_vehicles


# %%
import random
def initialize_vehicles_random(depots):
    vehicles = []
    n_vehicles = sum(d.vehicles_now for d in depots)
    vehicle_id_counter = 0
    for i, d in enumerate(depots):
        start_depot_id = d.id
        # Choose a random end depot that is different from the start depot
        end_depot = random.choice([depot for depot in depots if depot.id != start_depot_id])
        end_depot_id = end_depot.id
        for _ in range(d.vehicles_now):
            vehicles.append(Vehicle(id=vehicle_id_counter,
                                    pcapacity=5,
                                    gcapacity=10,
                                    R_=1,
                                    ctype = d.type,
                                    start_depot=start_depot_id,
                                    end_depot=end_depot_id))
            vehicle_id_counter += 1
    return vehicles, n_vehicles




def initialize_vehicles_with_specific_start_end(depots, end_depots = {42432963:42430044,42430044: 6177439750, 6223571524:42432963, 3786901738:42432963, 6177439750:6223571524}):
    vehicles = []
    n_vehicles = sum(d.vehicles_now for d in depots)
    vehicle_id_counter = 0
    for i, d in enumerate(depots):
        start_depot_id = d.id
        end_depot_id = end_depots[start_depot_id]
        for _ in range(d.vehicles_now):
            vehicles.append(Vehicle(id=vehicle_id_counter,
                                    pcapacity=5,
                                    gcapacity=10,
                                    R_=1,
                                    ctype = d.type,
                                    start_depot=start_depot_id,
                                    end_depot=end_depot_id))
            vehicle_id_counter += 1
    return vehicles, n_vehicles


# %%
depots = initialize_depots(graph, vehicles_amount = [1]*5)
vehicles, _ = initialize_vehicles_naive(depots)

# %%
depot_ids = [d.id for d in depots]  # Generate the list of depot ids

# Replace values with NaN where dropoff_graph_node is in the list of depot ids
df.loc[df['dropoff_graph_node'].isin(depot_ids), 'dropoff_graph_node'] = np.nan
mode_value = df['dropoff_graph_node'].mode()[0]
df['dropoff_graph_node'].fillna(mode_value, inplace=True)
df.loc[df['pickup_graph_node'].isin(depot_ids), 'pickup_graph_node'] = np.nan
mode_value = df['pickup_graph_node'].mode()[0]
df['pickup_graph_node'].fillna(mode_value, inplace=True)

# %%
def clean_requests(requests):

    tot = 0
    for r in requests:
        if r.people_ >4:
            requests.remove(r)
            continue
    
        for d in depots:
            if r.end == d.id or r.start == d.id:
                requests.remove(r)
                continue
        tot = r.people_ + tot

    return requests, tot

# %%
_, _, requests = get_requests(df[:1])
requests, tot = clean_requests(requests)


# %%
edges_to_remove = [(u, v) for u, v in graph.edges() if u == v]
graph.remove_edges_from(edges_to_remove)

# %%
import math
def create_problem2(graph,depots, vehicles, requests,edge_info=edge_info, name="VehicleRoutingProblem", subsets=[]):
    problem = pulp.LpProblem(name, pulp.LpMaximize)
    #Routing variable
    V = {}
    for a in vehicles:
        for u in graph.nodes():
            for v in graph.nodes():
                if graph.has_edge(u, v):
                    V[a.id,u, v]= pulp.LpVariable(f"V{a.id}{u}{v}", cat='Binary') 
    
    #Request Assigning variable
    x = {}
    for a in vehicles:
        for r in requests:
            x[a.id,r.id]= pulp.LpVariable(f"x{a.id},{r.id}", cat='Binary') 

    s = {}
    for a in vehicles:
        for v in graph.nodes():
                s[a.id,v]= pulp.LpVariable(f"s{a.id},{v}") 

    y = {}
    N = len(requests)+200100
    for a in vehicles:
            y[a.id]= pulp.LpVariable(f"y{a.id}", cat='Binary') 
            problem += N*y[a.id]>= (pulp.lpSum(x[a.id, r.id] for r in requests)) 



    #(4.24)
    for u,v in graph.edges():
         problem += pulp.lpSum(V[a.id,u,v] for a in vehicles) <= edge_info[u,v][2]

    #(4.27)
    for a in vehicles:
        problem += pulp.lpSum(V[a.id,u,v]*a.R_*edge_info[u,v][1]for u,v in graph.edges() ) <= a.charge


    #(4.5)
    for a in vehicles:
            problem += pulp.lpSum(x[a.id,r.id]*r.people_ for r in requests) <= a.pcapacity_

    #(4.25)
    M = 50000000
    for a in vehicles:
        for i in graph.nodes():
                for j in graph.nodes():
                    if j != a.start_depot and graph.has_edge(i, j):
                        problem +=s[a.id,i] + edge_info[(i,j)][1] - M*(1 - V[a.id, i, j]) <=s[a.id,j]


    #(4.9)
    for r in requests:
        problem += pulp.lpSum(x[a.id,r.id] for a in vehicles) <= 1
    
    #for a in vehicles:
    #    problem += pulp.lpSum(x[a.id,r.id] for r in requests) == len(requests)

    (4.16)
    for a in vehicles:
        temp_list = list(graph.nodes()).copy()
        temp_list.remove(a.start_depot)
        if a.start_depot != a.end_depot:
            temp_list.remove(a.end_depot)
        no_end = list(graph.nodes()).copy()
        no_end.remove(a.end_depot)
        no_start = list(graph.nodes()).copy()
        no_start.remove(a.start_depot)
        
        for v in temp_list:
            problem += pulp.lpSum(V[a.id,u, v] for u in no_end if graph.has_edge(u, v)) \
                    - pulp.lpSum(V[a.id,v, w] for w in no_start if graph.has_edge(v, w)) == 0
    
    #(4.32) - (4.33)
    for a in vehicles:
        temp_list = list(graph.nodes()).copy()
        temp_list.remove(a.start_depot)
        if a.start_depot != a.end_depot:
            temp_list.remove(a.end_depot)
        no_end = list(graph.nodes()).copy()
        no_end.remove(a.end_depot)
        no_start = list(graph.nodes()).copy()
        no_start.remove(a.start_depot)
        if a.start_depot == a.end_depot:
            continue
        #problem +=   (pulp.lpSum(V[a.id, v, a.start_depot] for v in no_start if graph.has_edge( v,a.start_depot)) ) == 0
        #problem +=   (pulp.lpSum(V[a.id, a.end_depot,  v] for v in no_end if graph.has_edge( a.end_depot, v)) ) == 0
#       
        problem +=   (pulp.lpSum(V[a.id, v, a.end_depot] for v in no_start if graph.has_edge( v,a.end_depot)) ) == y[a.id]
        #for r in requests:
        problem +=   (pulp.lpSum(V[a.id, a.start_depot,  v] for v in no_end if graph.has_edge( a.start_depot, v)) ) ==y[a.id]


    #(4.30) - (4.31)
    for a in vehicles:
        for r in requests:
                problem += pulp.lpSum(V[a.id,  v,r.start] for v in graph.nodes() if graph.has_edge( v,r.start)) >= x[a.id, r.id]
                problem += pulp.lpSum(V[a.id,  v,r.end] for v in graph.nodes() if graph.has_edge( v,r.end)) >= x[a.id, r.id]
    

             
    
    #problem+=pulp.lpSum(x[a.id,r.id] for a in vehicles for r in requests)

    #problem += pulp.lpSum(y[a.id,r.start]  for a in vehicles for r in requests)
    problem += pulp.lpSum(x[a.id,r.id] for a in vehicles for r in requests)
    #problem += pulp.lpSum(-V[a.id,u, v]*edge_info[(u,v)][1] for a in vehicles for v in graph.nodes() for u in graph.nodes() if graph.has_edge(u, v))

    return problem, V, x,y

def check_road(id, depot_id, V):
    raw_road = {}
    lrr = 0
    for u, v in graph.edges():
        if pulp.value(V[id,u, v]) == 1:
            raw_road[u] = v 
            lrr+=1
    n = depot_id
    final_road = {}
    print(lrr,raw_road)
    print( depot_id)
    for i in range(lrr):
        print(i+1,n)
        final_road[n] = raw_road[n]
        n = raw_road[n]

    print(len(final_road.keys()))

    print(final_road)
    return final_road
        

# %%
#node = 1
#final_road = check_road(node, vehicles[node].start_depot)

# %%
def choose_requests(requests,graph, amount, from_index):
    picked = []
    idx = from_index
    while len(picked) != amount:
        r = requests[idx]
        idx+=1
        if idx > len(requests):
            return -1
        if r.end not in graph.nodes():
            print("r.end not in nodes")
            continue
        if r.start not in graph.nodes():
            print("Req not in nodes")
            continue
        if r.start not in graph.nodes():
            print("Req not in nodes")
            continue
        picked += [r]

    return picked

# %%
def update_vehicles(vehicles, graph, V, edges_info):
    for a in vehicles:
        new_charge = a.charge
        for u,v in graph.edges():
            try:
                new_charge -= pulp.value(V[a.id,u,v])*a.R_*edges_info[u,v][1]  
            except:
                pass
        if new_charge != a.charge:
                old = a.end_depot 
                a.end_depot = a.start_depot
                a.start_depot = old
                a.charge = new_charge
        
        #print(f"Vehicle {a.id} soc {a.charge}")
            #a.charge += timing_charge
    

# %%
def update_depots(depots, vehicles, timing_charge = "fill"):
    #for d in depots:
    #    print(f"{d.id} # vehicles: {d.vehicles_now}")
    #print()
    for d in depots:
        vehicles_now = 0
        for a in vehicles:
            if a.start_depot == d.id:
                vehicles_now +=1
                if timing_charge == "fill":
                    new_charge = a.Q
                else:
                    b = a.Q*a.theta / a.R_
                    if timing_charge <= b:
                        new_charge = a.charge + a.R_* timing_charge
                    if timing_charge > b: 
                        new_charge = a.charge + a.R_*b
                        i_t = a.R_*np.exp(-(timing_charge-b)/a.tau)
                        new_charge+= a.Q - a.Q*i_t*(1-a.theta)/a.R_

                a.charge = new_charge if new_charge< 100 else 100
        d.vehicles_now = vehicles_now
    #    print(f"{d.id} # vehicles: {d.vehicles_now}")
    
    #for a in vehicles:
        #print(f"Vehicle {a.id} soc: {a.charge}")
        #print("after:", d.vehicles_now)
        

# %%

def check_and_create_folder(folder_path, name_fold = "simu"):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        #return folder_path

    # Look for folders starting with "simu_"
    existing_folders = [name for name in os.listdir(folder_path) if name.startswith(f"{name_fold}_")]
    #print(f"{name}_")
    if not existing_folders:
        new_folder_name = f"{name_fold}_0001"
    else:
        # Sort the existing folders to find the highest numbered folder
        existing_folders.sort()
        last_folder_number = int(existing_folders[-1][len(name_fold)+1:])  # Extract the number from the folder name 
        #print(os.path.join(folder_path, existing_folders[-1]), len(os.listdir(os.path.join(folder_path, existing_folders[-1]))))   
        new_folder_number = last_folder_number  if  len(os.listdir(os.path.join(folder_path, existing_folders[-1]))) ==0 else last_folder_number +1
        #print(new_folder_number)
        new_folder_name = f"{name_fold}_{new_folder_number:04d}"

    new_folder_path = os.path.join(folder_path, new_folder_name)
    os.makedirs(new_folder_path, exist_ok=True)

    return new_folder_path

# %%
def save_info(V,x,requests, graph, vehicles, old_vehicles, old_depots,edges_info, folder_path, ite,name_road = 'road_info.csv', name_vehicles = 'vehicle_info.csv', name_edge_info = "edge_info.csv", name_request_info = "requests_info.csv"):
    road_info = {}
    for u, v in graph.edges():
        edge_used = []
        for a in vehicles:
            edge_used.append(pulp.value(V[a.id,u,v]))
        road_info[f"{u}-{v}"] = edge_used

    df = pd.DataFrame.from_dict(road_info, orient='index', columns=[a.id for a in vehicles]).reset_index().rename(columns={'index': 'edge'})
    df.to_csv(os.path.join(folder_path, f"{ite:04d}_{name_road}"))


    vehicle_info = {}
    for a, a_old, (start_depot, end_depot) in zip(vehicles, old_vehicles, old_depots):
        r = sum(pulp.value(x[a.id,r.id]) for r in requests)
        vehicle_info[a.id] = [
                                a_old.charge,
                                a.charge,
                                start_depot,
                                a.start_depot,
                                end_depot,
                                a.end_depot,
                                r
                            ]

    pd.DataFrame.from_dict(vehicle_info, orient='index', columns=[
                                                    "old_charge",
                                                    "new_charge",
                                                    "old_startdepot",
                                                    "new_startdepot",
                                                    "old_enddepot",
                                                    "new_enddepot",
                                                    "requests_done"
                                                ]).reset_index().rename(columns={'index': 'id'}).to_csv(os.path.join(folder_path, f"{ite:04d}_{name_vehicles}"))
    

    new_edges_info = {}
    for u,v in edges_info.keys():
        new_edges_info[f"{u}-{v}"]  = edges_info[(u,v)]


    pd.DataFrame.from_dict(new_edges_info, orient='index', columns=[
                                                    "distance",
                                                    "time",
                                                    "capacity"
                                                ]).reset_index().rename(columns={'index': 'edge'}).to_csv(os.path.join(folder_path, f"{ite:04d}_{name_edge_info}"))


    requests_info = {}
    for a in vehicles:
        requests_info[a.id] = []
        for r in requests:
            requests_info[a.id] += [pulp.value(x[a.id,r.id])]
    
    pd.DataFrame.from_dict(requests_info, orient='index', columns=[
                                                    r.id for r in requests
                                                ]).reset_index().rename(columns={'index': 'edge'}).to_csv(os.path.join(folder_path, f"{ite:04d}_{name_request_info}"))

    

# %%

def insert_random_end_point(requests, graph):
    for r in requests:
        r.end = random.choice([x for x in graph.nodes() if x != r.start])
    return requests

import copy
def rec_horizon_problem(df, 
                        graph, requests, depots, vehicles,edges_info, 
                        solver,
                        req_per_i = 3, charging_time = 40, iterations = 30, capacitissimo=0):
    folder_path = check_and_create_folder("./results_random_depot_correct")
    
    already_done = 0
    res = -1
    
    
    if not isinstance(req_per_i, list):
        req_per_i = [req_per_i]*iterations

    progress_bar = tqdm(range(iterations),total=iterations, position = 0)

    for i in progress_bar:
        picked_requests = choose_requests(requests,graph, req_per_i[i], i+already_done)
        if picked_requests==-1:
            print("Available requests are over :)")
            return 
        picked_requests = insert_random_end_point(picked_requests, graph)
        already_done += len(picked_requests)
        
        problem, V, x, y= create_problem2(graph,depots, vehicles, picked_requests)
        tqdm.write("Problem Created")
        #res = problem.solve(pulp.PULP_CBC_CMD( msg=0,timeLimit=60))
        res = problem.solve(solver=solver)
        if res == 1:
            old_depots = [(a.start_depot, a.end_depot)for a in vehicles]
            update_vehicles(vehicles, graph, V, edges_info)
            old_vehicles = copy.deepcopy(vehicles)
            #print()
            update_depots(depots, vehicles, charging_time)
            save_info(V,x, picked_requests,graph, vehicles,old_vehicles, old_depots, edges_info, folder_path, ite= i+1)
            #print()


            #already_done += requests_completed_successfully
        
        progress_bar.set_description(f"Old iteration {res}, {charging_time}- {capacitissimo}")
        
        
        #print("-----------------------------------------------------")

# %%


# %%

_, _, requests = get_requests(df)
requests, tot = clean_requests(requests)


solver= pulp.GUROBI_CMD(msg=1, options=[('Threads', 8)])
for capacitissimo, charging_time in zip([ 40, 20, 10, 10 ], [ 20,10,20,10]):
    print(capacitissimo, charging_time)
    depots = initialize_depots(graph, vehicles_amount = [3,6,8,4,3])
    vehicles, _ = initialize_vehicles_with_specific_start_end(depots)
    # Iterate over edges and add distances to the dictionary
    edge_info = {}
    for u, v, d in graph.edges(data=True):
        edge_info[(u, v)] = [
            d["length"]/10,
            d["length"]/500,
            capacitissimo
        ]
    rec_horizon_problem(df,
                         graph, requests, depots, vehicles, edge_info,
                         solver= solver, 
                         req_per_i=50, charging_time = charging_time, capacitissimo = capacitissimo)


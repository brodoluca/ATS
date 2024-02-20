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
df_filter = pd.read_csv("requests_done_simulation.csv")
filter_ids = df_filter["id"].tolist()
df = df[df["id"].isin(filter_ids)]
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
        50
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
    def __init__(self, id, people, start, end,  py,px, dy,dx) -> None:
        self.id = id
        self.people_ = people
        self.start = start
        self.end = end 
        self.px = px
        self.py=py
        self.dx = dx
        self.dy = dy


# %%
def convert_reqs_to_obj(requests, required_info = ["id", "passenger_count", "pickup_graph_node","dropoff_graph_node","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]):
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
def convert_reqs_to_list(requests, required_info = ["id", "passenger_count", "pickup_graph_node","dropoff_graph_node","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]):
    temp_reqs = []
    for node, item in requests.items():
        amount = item["amount"]
        for k in item["internal_ids"]:
            temp_reqs += [
                        Request( *[item[x][k] for x in required_info])
                        ]
    return temp_reqs

# %%
def get_requests(dataframe, required_info = ["amount","internal_ids", "id", "passenger_count", "pickup_graph_node","dropoff_graph_node","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]):
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
p, _, requests = get_requests(df[:20])

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
        end_depot_id =depots[(i + 1) % len(depots)].id  # Ensures start and end depots are different
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

# %%
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
#vehicles, _ = initialize_vehicles_naive(depots)

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
edges_to_remove = [(u, v) for u, v in graph.edges() if u == v]
graph.remove_edges_from(edges_to_remove)

# %%
def check_road(id, depot_id, V,s):
    raw_road = {}
    times = {}
    arriving_times = {}
    lrr = 0
    for u, v in graph.edges():
        if pulp.value(V[id,u, v]) >0.5:
            raw_road[u] = v 
            if s != None:
                times[u] = pulp.value(s[id,u])
                arriving_times[v] = pulp.value(s[id,v])
            lrr+=1
    n = depot_id
    final_road = {}
    if s != None:
        print(times)
        print(arriving_times)
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
                new_charge -= int(pulp.value(V[a.id,u,v]) > 0.5)*a.R_*edges_info[u,v][1]  
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
def save_info(V,x,requests, graph, vehicles, old_vehicles, old_depots,edges_info, folder_path, ite, b = None,name_road = 'road_info.csv', name_vehicles = 'vehicle_info.csv', name_edge_info = "edge_info.csv", name_request_info = "requests_info.csv"):
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

    if b == None:
        return
    
    end_depots_info = {}
    for a in vehicles:
        end_depots_info[a.id] = []
        for d in depots:
            if d.id == a.start_depot:
                continue
            end_depots_info[a.id] += [pulp.value(b[a.id,d.id])]
    
    pd.DataFrame.from_dict(end_depots_info, orient='index', columns=[
                                                    d.id for d in depots
                                                ]).reset_index().rename(columns={'index': 'edge'}).to_csv(os.path.join(folder_path, f"{ite:04d}_end_depots_info.csv"))

    

# %%
def insert_random_end_point(requests, graph):
    for r in requests:
        r.end = random.choice([x for x in graph.nodes() if x != r.start])
    return requests
    

# %%
import math
def create_problem_w_rebalancing(graph,depots, vehicles, requests,edge_info=edge_info, name="VehicleRoutingProblem", subsets=[]):
    problem = pulp.LpProblem(name, pulp.LpMinimize)
    #Routing variable
    V = {}
    for a in vehicles:
        for u in graph.nodes():
            for v in graph.nodes():
                if graph.has_edge(u, v):
                    V[a.id,u, v]= pulp.LpVariable(f"V{a.id}{u}{v}", cat='Binary') 
    
    #Check IF a request is resolved
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
    #is a depot assigned to the vehicle
    b = {}
    for a in vehicles:
        for d in depots:
            b[a.id, d.id]= pulp.LpVariable(f"b{a.id},{d.id}", cat='Binary') 


    #for a in vehicles:
    #    problem+=pulp.lpSum(b[a.id, d.id] for d in depots) == y[a.id]
        


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
    M = 34
    for a in vehicles:
        for i in graph.nodes():
            for j in graph.nodes():
                if graph.has_edge(i, j):
                    problem +=s[a.id,i] + edge_info[(i,j)][1] - M*(1 - V[a.id, i, j]) <=s[a.id,j]
#
    

    starts = [r.start for r in requests]
    ends = [r.end for r in requests]
    for a in vehicles:
        for v in graph.nodes():
            if v in starts: #or v in ends:
                continue
            problem += s[a.id,v] >=0
            problem += s[a.id,v] <= 30
    
    for r in starts:
        problem += s[a.id,r] <=20
        problem += s[a.id,r] >=10
    

    #(4.9)
    for r in requests:
        problem += pulp.lpSum(x[a.id,r.id] for a in vehicles) == 1

    #(4.16)
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
            problem += pulp.lpSum(V[a.id,u, v] for u in  graph.nodes()  if graph.has_edge(u, v)) \
                    - pulp.lpSum(V[a.id,v, w] for w in  graph.nodes()  if graph.has_edge(v, w)) == 0


    d_ids = [d.id for d in depots]


    
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
        problem +=   (pulp.lpSum(V[a.id, v, a.end_depot] for v in graph.nodes() if graph.has_edge( v,a.end_depot)) ) == 1#1y[a.id]
        #for r in requests:
        #problem +=   (pulp.lpSum(V[a.id, v,a.start_depot] for v in graph.nodes() if graph.has_edge( v,a.start_depot)) ) ==0
        problem +=   (pulp.lpSum(V[a.id, a.start_depot,  v] for v in graph.nodes() if graph.has_edge( a.start_depot, v)) ) ==1#y[a.id]


    #(4.30) - (4.31)
    #for a in vehicles:
    #    for r in requests:
                #problem += pulp.lpSum(V[a.id,  v,r.start] for v in graph.nodes() if graph.has_edge( v,r.start)) >= x[a.id, r.id]
    #            problem += pulp.lpSum(V[a.id,  v,r.end] for v in graph.nodes() if graph.has_edge( v,r.end)) >= x[a.id, r.id]
    for r in requests:
        for a in vehicles:
            no_end = list(graph.nodes()).copy()
            no_end.remove(a.end_depot)
            problem += pulp.lpSum(V[a.id,  v,r.start] for v in graph.nodes() if graph.has_edge( v,r.start) ) >= x[a.id, r.id]
            #problem += pulp.lpSum(V[a.id,  r.start,v] for v in graph.nodes() if graph.has_edge( r.start,v) ) >= x[a.id, r.id]
        
        #for a in vehicles:
        #    problem += pulp.lpSum(V[a.id,  v,r.start] for v in graph.nodes() if graph.has_edge( v,r.start)) >= x[a.id, r.id]
             
    
    #problem+=pulp.lpSum(x[a.id,r.id] for a in vehicles for r in requests)

    #problem += pulp.lpSum(y[a.id,r.start]  for a in vehicles for r in requests)
    problem += pulp.lpSum(-x[a.id,r.id] for a in vehicles for r in requests)
    #problem += pulp.lpSum(V[a.id,u, v]*edge_info[(u,v)][1] for a in vehicles for v in graph.nodes() for u in graph.nodes() if graph.has_edge(u, v))

    return problem, V, x,y,b,s

# %%
p, _, requests = get_requests(df)
requests, tot = clean_requests(requests)
#random.shuffle(requests)
depots = initialize_depots(graph, vehicles_amount = [3,6,8,4,3])

vehicles, _ = initialize_vehicles_with_specific_start_end(depots)


# %%
for d in depots:
    print(d.id, d.vehicles_now)

# %%
for a in vehicles:
    print(a.start_depot)

# %%
def test_problem(graph, vehicles, requests,edge_info=edge_info, name="VehicleRoutingProblem", link_capacity=50):
    problem = pulp.LpProblem(name, pulp.LpMinimize)
    #Routing variable
    V = {}
    for a in vehicles:
        for u in graph.nodes():
            for v in graph.nodes():
                if graph.has_edge(u, v):
                    V[a.id,u, v]= pulp.LpVariable(f"V{a.id}{u}{v}", cat='Binary') 
    
   
    x = {}
    for a in vehicles:
        for r in requests:
            x[a.id,r.id]= pulp.LpVariable(f"x{a.id},{r.id}", cat='Binary') 
    done = []
    #for r in requests:
    #    if r in done:
    #        continue
    #    for rr in requests:
    #        if rr in done:
    #            continue
    #        if rr == r:
    #            continue
    #        if r.end == rr.start and r.start == rr.end:
    #            done += [r,rr]
    #            for a in vehicles:
    #                problem += x[a.id,r.id] + x[a.id,rr.id]==1

    y = {}
    N = len(requests)+200100
    for a in vehicles:
            y[a.id]= pulp.LpVariable(f"y{a.id}", cat='Binary') 
            problem += N*y[a.id]>= (pulp.lpSum(x[a.id, r.id] for r in requests)) 

    for a in vehicles:
        for node in graph.nodes():
            if node != a.start_depot and node != a.end_depot:
                problem += pulp.lpSum(V[a.id, i, node] for i in graph.nodes() if graph.has_edge(i, node)) == \
                        pulp.lpSum(V[a.id, node, j] for j in graph.nodes() if graph.has_edge(node, j))


    capacity_starts = {r.start:a.pcapacity_ for r in requests}

    #(4.9)
    for r in requests:
        problem += pulp.lpSum(x[a.id,r.id] for a in vehicles) <= 1

     #(4.24)
    for u,v in graph.edges():
         problem += pulp.lpSum(V[a.id,u,v] for a in vehicles) <= link_capacity

    #(4.27)
    for a in vehicles:
        problem += pulp.lpSum(V[a.id,u,v]*a.R_*edge_info[u,v][1]for u,v in graph.edges() ) <= a.charge


    #(4.5)
    for a in vehicles:
            problem += pulp.lpSum(x[a.id,r.id]*r.people_ for r in requests) <= a.pcapacity_


    """Overall Time Spent"""
    t = {}
    for a in vehicles:
        for v in graph.nodes():
            t[a.id, v] = pulp.LpVariable(f"t{a.id},{v}")
            problem += t[a.id, v] <= 500#a.pcapacity_ + pulp.lpSum(V[a.id, i, j] for i, j in graph.edges())
            problem += t[a.id, v] >= 0
    problem += t[a.id, a.start_depot] == 0


    
    for r in requests:
        for a in vehicles:
            problem +=t[a.id, r.end] -  t[a.id, r.start] >= x[a.id,r.id]-1
    


    starts_ends = {r.start:r.end for r in requests}
    M = len(graph.nodes()) + sum([edge_info[(i,j)][1] for i,j in graph.edges()])
    for a in vehicles:
        for i, j in graph.edges():
            if i == a.start_depot or j == a.start_depot:
                continue
            if j in list(starts_ends.keys()) and i == starts_ends[j]:
                continue
            d_i = edge_info[(i,j)][1]
            problem += t[a.id, i] + d_i <= t[a.id, j] + M * (1 - V[a.id, i, j])


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
        problem +=   (pulp.lpSum(V[a.id, v, a.end_depot] for v in graph.nodes() if graph.has_edge( v,a.end_depot)) ) == y[a.id]
        problem +=   (pulp.lpSum(V[a.id, a.start_depot,  v] for v in graph.nodes() if graph.has_edge( a.start_depot, v)) ) ==y[a.id]


    for r in requests:
        for a in vehicles:
            no_end = list(graph.nodes()).copy()
            no_end.remove(a.end_depot)
            problem += pulp.lpSum(V[a.id,  v,r.start] for v in graph.nodes() if graph.has_edge( v,r.start) ) >= x[a.id,r.id]
            problem += pulp.lpSum(V[a.id,  r.end,v] for v in graph.nodes() if graph.has_edge( r.end,v) ) >= x[a.id,r.id]

    #problem += pulp.lpSum(V[a.id,u, v]*edge_info[(u,v)][0] for a in vehicles for u,v in graph.edges())
    problem += pulp.lpSum(-x[a.id,r.id] for a in vehicles for r in requests) #+ pulp.lpSum(V[a.id,u,v]*a.R_*edge_info[u,v][1]for u,v in graph.edges() for a in vehicles ) 
    #problem += pulp.lpSum(V[a.id,u, v]*edge_info[(u,v)][1] for a in vehicles for v in graph.nodes() for u in graph.nodes() if graph.has_edge(u, v))
    return problem, V, x,t,None,None
    

# %%
test_vehicles, _ = initialize_vehicles_with_specific_start_end(depots)

# %%
import copy
def rec_horizon_problem(df, graph,requests,vehicles,edges_info, solver, link_capacity=50, req_per_i = 3, charging_time = 20, iterations = 1, test = True):
    if not test:
        folder_path = check_and_create_folder("./ti_prego_dio")
    already_done = 0
    res = -1
    if not isinstance(req_per_i, list):
        req_per_i = [req_per_i]*iterations

    progress_bar = tqdm(range(iterations),total=iterations)

    for i in progress_bar:
        picked_requests = choose_requests(requests,graph, req_per_i[i], i+already_done)
        if picked_requests==-1:
            print("Available requests are over :)")
            return 
        already_done += len(picked_requests)
        
        problem, V, x, t, b,s= test_problem(graph, edge_info=edge_info,vehicles = vehicles, requests=picked_requests, link_capacity=link_capacity)
        #res = problem.solve(pulp.PULP_CBC_CMD( msg=0,timeLimit=60))
        
        res = problem.solve(solver)
        if res == 1:
            old_depots = [(a.start_depot, a.end_depot)for a in vehicles]
#           
            h = 0    
            for a in vehicles:
                for r in picked_requests:
                    if pulp.value(x[a.id, r.id])>0.5:
                        h+=1
                        print( a.id, a.end_depot, r.start, r.end)
                        check_road(a.id, a.start_depot, V, t)
            print(h)

            update_vehicles(vehicles, graph, V, edges_info)
            old_vehicles = copy.deepcopy(vehicles)
            #print()
            update_depots(depots, vehicles, charging_time)
            if not test:
                save_info(V,x, picked_requests,graph, vehicles,old_vehicles, old_depots, edges_info, folder_path, ite= i+1, b = b)
            #print()
            
                        
        
        progress_bar.set_description(f"Old iteration {res}")
        
        
        #print("-----------------------------------------------------")

# %%
rebalancing_areas_df = pd.read_csv("dataset/rebalancing_areas.csv")

# %%
mid_points = [(float(point.split()[0][1:]), float(point.split()[1][:-1]))for  point in list(rebalancing_areas_df["Contraced Points"])[:-1]]
depots_to_area = { depot:area for area,depot in enumerate(list(rebalancing_areas_df["depots"])[:-1])}

# %%
def rebalance_depots(depots, new_requests,mid_points,depots_to_area):
    problem = pulp.LpProblem("Rebalancing", pulp.LpMinimize)
    vehicles_amount_per_depot = {}
    area_capacity = {}
    #rows_with_item = df[df['A'] == item]
    for r in new_requests:
        area = np.argmin(np.linalg.norm(np.array(mid_points) - [r.py,r.px], axis=1))
        if area not in area_capacity.keys():
            area_capacity[area] = r.people_
        else:
            area_capacity[area] += r.people_
        #print([r.py,r.px])
        #print(area_capacity)

    b = {}
    for a in vehicles:
        for d in depots:
            b[a.id,d.id]= pulp.LpVariable(f"b{a.id},{d.id}", cat='Binary') 
    


    for a in vehicles:
        problem += pulp.lpSum(b[a.id,d.id] for d in depots) == 1
    
    for d in depots:
        area = depots_to_area[d.id]
        problem += pulp.lpSum(a.pcapacity_*b[a.id,d.id] for a in vehicles) >= area_capacity[area] if area in area_capacity.keys() else 0

    #We want to minimize vehicles moving, hence the difference between the old and new depot
    #Or maximize the number of depots keeping their vehicles
    
    problem += pulp.lpSum(-b[a.id,a.start_depot] for a in vehicles)

    ret = problem.solve(pulp.PULP_CBC_CMD( msg=0,timeLimit=60))
    print(area_capacity)
    print(ret)
    for a in vehicles:
        for d in depots:
            if pulp.value(b[a.id, d.id])>0.5:
                print(a.id, a.start_depot, d.id, )
    return vehicles_amount_per_depot
    

# %%


# %%


# %%


# %%
depots_to_area.keys()

# %%
def test_problem_with_rebalancing(graph, vehicles, requests,rebalancing_picked_requests,
                                  mid_points,depots_to_area,
                                  edge_info=edge_info, name="VehicleRoutingProblem", link_capacity=50):
    problem = pulp.LpProblem(name, pulp.LpMaximize)

    


    #Routing variable
    V = {}
    for a in vehicles:
        for u in graph.nodes():
            for v in graph.nodes():
                if graph.has_edge(u, v):
                    V[a.id,u, v]= pulp.LpVariable(f"V{a.id}{u}{v}", cat='Binary') 
    
   
    x = {}
    for a in vehicles:
        for r in requests:
            x[a.id,r.id]= pulp.LpVariable(f"x{a.id},{r.id}", cat='Binary') 


    y = {}
    N = len(requests)+200100
    for a in vehicles:
            y[a.id]= pulp.LpVariable(f"y{a.id}", cat='Binary') 
            problem += N*y[a.id]>= (pulp.lpSum(x[a.id, r.id] for r in requests)) 

    for a in vehicles:
        for node in graph.nodes():
            if node in depots_to_area.keys():
                continue
            if node != a.start_depot: #and node != a.end_depot:  The end depot must be taken care differently!!
                problem += pulp.lpSum(V[a.id, i, node] for i in graph.nodes() if graph.has_edge(i, node)) == \
                        pulp.lpSum(V[a.id, node, j] for j in graph.nodes() if graph.has_edge(node, j))


    capacity_starts = {r.start:a.pcapacity_ for r in requests}

    #(4.9)
    for r in requests:
        problem += pulp.lpSum(x[a.id,r.id] for a in vehicles) <= 1

     #(4.24)
    for u,v in graph.edges():
         problem += pulp.lpSum(V[a.id,u,v] for a in vehicles) <= link_capacity

    #(4.27)
    for a in vehicles:
        problem += pulp.lpSum(V[a.id,u,v]*a.R_*edge_info[u,v][1]for u,v in graph.edges() ) <= a.charge


    #(4.5)
    for a in vehicles:
            problem += pulp.lpSum(x[a.id,r.id]*r.people_ for r in requests) <= a.pcapacity_


    """Overall Time Spent"""
    t = {}
    for a in vehicles:
        for v in graph.nodes():
            t[a.id, v] = pulp.LpVariable(f"t{a.id},{v}")
            problem += t[a.id, v] <= 500#a.pcapacity_ + pulp.lpSum(V[a.id, i, j] for i, j in graph.edges())
            problem += t[a.id, v] >= 0
    problem += t[a.id, a.start_depot] == 0


    
    for r in requests:
        for a in vehicles:
            problem +=t[a.id, r.end] -  t[a.id, r.start] >= x[a.id,r.id]-1
    


    starts_ends = {r.start:r.end for r in requests}
    M = len(graph.nodes()) + sum([edge_info[(i,j)][1] for i,j in graph.edges()])
    for a in vehicles:
        for i, j in graph.edges():
            if i == a.start_depot or j == a.start_depot:
                continue
            if j in list(starts_ends.keys()) and i == starts_ends[j]:
                continue
            d_i = edge_info[(i,j)][1]
            problem += t[a.id, i] + d_i <= t[a.id, j] + M * (1 - V[a.id, i, j])


    for r in requests:
        for a in vehicles:
            problem += pulp.lpSum(V[a.id,  v,r.start] for v in graph.nodes() if graph.has_edge( v,r.start) ) >= x[a.id,r.id]
            problem += pulp.lpSum(V[a.id,  r.end,v] for v in graph.nodes() if graph.has_edge( r.end,v) ) >= x[a.id,r.id]

    for a in vehicles:
        temp_list = list(graph.nodes()).copy()
        temp_list.remove(a.start_depot)
        if a.start_depot != a.end_depot:
            temp_list.remove(a.end_depot)
        no_end = list(graph.nodes()).copy()
        no_end.remove(a.end_depot)
        no_start = list(graph.nodes()).copy()
        no_start.remove(a.start_depot)
        #if a.start_depot == a.end_depot:
        #    continue
        #problem +=   (pulp.lpSum(V[a.id, v, a.end_depot] for v in graph.nodes() if graph.has_edge( v,a.end_depot)) ) == y[a.id]
        problem +=   (pulp.lpSum(V[a.id, a.start_depot,  v] for v in graph.nodes() if graph.has_edge( a.start_depot, v)) ) ==y[a.id]


    area_capacity = {}
    for r in rebalancing_picked_requests:
        area = np.argmin(np.linalg.norm(np.array(mid_points) - [r.py,r.px], axis=1))
        if area not in area_capacity.keys():
            area_capacity[area] = r.people_
        else:
            area_capacity[area] += r.people_ 
    #print(area_capacity)
    for a in vehicles:
        area = depots_to_area[a.start_depot]
        if area in area_capacity.keys():
            area_capacity[area] -= a.pcapacity_
            if area_capacity[area] <0:
                area_capacity[area] =0
    #    print(f"Starting depot{a.start_depot} and area assigned to id {area}")
    #print(area_capacity)
    #print()
    b = {}
    w = {}
    for a in vehicles:
        w[a.id]= pulp.LpVariable(f"w{a.id}", cat='Binary') 
        for d in depots:
            if d.id != a.start_depot:
                b[a.id,d.id]= pulp.LpVariable(f"b{a.id},{d.id}", cat='Binary') 
        
        problem += y[a.id]<= pulp.lpSum(b[a.id,d.id] for d in depots if d.id != a.start_depot)




    depots_to_consider = []
    for d in depots:
        area = depots_to_area[d.id]
        if area in area_capacity.keys() and area_capacity[area]>0:
            depots_to_consider += [d.id]


    for a in vehicles:
        problem += pulp.lpSum(b[a.id,d.id] for d in depots if d.id != a.start_depot) <= 1
    
    
    for a in vehicles:
        for node in depots_to_area.keys():
            if node != a.start_depot:
                problem += pulp.lpSum(V[a.id, i, node] for i in graph.nodes() if graph.has_edge(i, node)) - \
                        pulp.lpSum(V[a.id, node, j] for j in graph.nodes() if graph.has_edge(node, j)) == (b[a.id,node])#(node == a.end_depot)#


    for d in depots:
        if d.id in depots_to_consider:
            #print(d.id, depots_to_area[d.id])
            problem +=  4>= pulp.lpSum(a.pcapacity_*b[a.id,d.id] for a in vehicles if d.id != a.start_depot) -area_capacity[depots_to_area[d.id]] >=0
    
    

    #c_f = {}
    #for d in depots:
    #    c_f[d.id]= pulp.LpVariable(f"c_f{d.id}") 
    #for d in depots:
    #    if d.id in depots_to_consider:
    #        #print(area,area_capacity[area])
    #        c_f[d.id] = pulp.lpSum(a.pcapacity_*b[a.id,d.id] for a in vehicles if d.id != a.start_depot) -area_capacity[depots_to_area[d.id]]
            


    #problem += pulp.lpSum(V[a.id,u, v]*edge_info[(u,v)][0] for a in vehicles for u,v in graph.edges())
    problem += pulp.lpSum(x[a.id,r.id] for a in vehicles for r in requests) + \
               pulp.lpSum(y[a.id] - pulp.lpSum(b[a.id,d.id] for d in depots if d.id != a.start_depot) for a in vehicles) #-\
               #pulp.lpSum(V[a.id,u,v]*a.R_*edge_info[u,v][1]for u,v in graph.edges() for a in vehicles ) 
    #problem +=  pulp.lpSum(c_f[d.id] for d in depots if d.id in depots_to_consider)
    #problem += pulp.lpSum(0.5*b[a.id,d.id] for a in vehicles for d in depots if d != a.start_depot)
    #problem += pulp.lpSum(V[a.id,u, v]*edge_info[(u,v)][1] for a in vehicles for v in graph.nodes() for u in graph.nodes() if graph.has_edge(u, v))
    return problem, V, x,t,b
    

# %%
import copy
def rec_horizon_problem_with_rebalancing(df, graph,requests,vehicles,edges_info,depots,solver,mid_points,depots_to_area,
                                         link_capacity=50, req_per_i = 3, charging_time = 20, iterations = 1, test = True):
    if not test:
        folder_path = check_and_create_folder("./dio_ti_prego3")
    already_done = 0
    res = -1
    if not isinstance(req_per_i, list):
        req_per_i = [req_per_i]*iterations
    
    def temporary_req_printer(reqs):
        for r in reqs:
            print(r.id)
    progress_bar = tqdm(range(iterations),total=iterations)
    routing_picked_requests = choose_requests(requests,graph, req_per_i[0], 0)
    for i in progress_bar:
        #print("Rputing")
        #temporary_req_printer(routing_picked_requests)
        if routing_picked_requests==-1:
            print("Available requests are over :)")
            return 
        already_done += len(routing_picked_requests)
        if i+already_done<len(requests)-1:
            rebalancing_picked_requests = choose_requests(requests,graph, req_per_i[i], i+already_done)
        else:
            rebalancing_picked_requests = routing_picked_requests
        #print("Rebalancing")
        #temporary_req_printer(rebalancing_picked_requests)
        problem, V, x, t, b= test_problem_with_rebalancing(graph, edge_info=edge_info,vehicles = vehicles, 
                                            requests=routing_picked_requests,rebalancing_picked_requests =rebalancing_picked_requests,
                                              link_capacity=link_capacity,
                                            mid_points = mid_points,depots_to_area = depots_to_area
                                            )
        
        #print(f"Iteration {i}")
        
        
        #res = problem.solve(pulp.PULP_CBC_CMD( msg=0,timeLimit=60))
        
        res = problem.solve(solver)
        if res == 1:
            old_depots = [(a.start_depot, a.end_depot)for a in vehicles]
#           print
            #print(b)
            
            #for a in vehicles:
            #    
            #    for d in depots:
            #        if d.id == a.start_depot:
            #            continue
            #        if pulp.value(b[a.id, d.id])>0.5:
            #            print(f"Vehicle id {a.id}  and (before assignmet){a.end_depot} end depot and (after assignment) end depot{d.id} (area {depots_to_area[d.id]})")
            #            #a.end_depot = d.id
            ##            print(a.id, a.start_depot, d.id, )
            #            print(a.end_depot)
            #print("b completed" )
            #h = 0    
            #for a in vehicles:
            #    for r in rebalancing_picked_requests:
            #        if pulp.value(x[a.id, r.id])>0.5:
            #            h+=1
            #            print( a.id, a.end_depot, r.start, r.end)
            #            check_road(a.id, a.start_depot, V, t)
            #print("Rebalancing Vehicles")
            #for a in vehicles:
            #    if sum(pulp.value(b[a.id, d.id]) for d in depots if d.id != a.start_depot) >0.5:
            #        print( a.id, a.end_depot, r.start, r.end)
            #        check_road(a.id, a.start_depot, V, t)
#
            #print(h)
            
            update_vehicles(vehicles, graph, V, edges_info)
            old_vehicles = copy.deepcopy(vehicles)
            #print()
            update_depots(depots, vehicles, charging_time)
            if not test:
                save_info(V,x, routing_picked_requests,graph, vehicles,old_vehicles, old_depots, edges_info, folder_path, ite= i+1, b = None)
            #print()
            
        routing_picked_requests = rebalancing_picked_requests         
        
        progress_bar.set_description(f"Old iteration {res}")
        
        
        #print("-----------------------------------------------------")

# %%
iterations = 10
req_per_i = 10
solver = pulp.GUROBI_CMD(msg=1, timeLimit=600)

# %%
for link_capacity, charging_time in zip([ 50, 10,5,5], [ 60,30,10,30]):
#for link_capacity, charging_time in zip([ 50], [ 60]):
    rec_horizon_problem_with_rebalancing(df, graph,requests,vehicles,edge_info, depots,mid_points =mid_points, depots_to_area =depots_to_area,
                                         solver = solver,link_capacity=link_capacity, req_per_i = req_per_i, 
                                         charging_time = charging_time, iterations = iterations, test = False)

# %%




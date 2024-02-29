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
df_filter = pd.read_csv("dio_aiutami_mpc/filter_reqs.csv")
filter_ids = df_filter["id"].tolist()
df = df[df["id"].isin(filter_ids)]
with open('graph_from_routes_wd.pkl', 'rb') as f:
    graph_from_routes = pickle.load(f)



# %%
df.head()

# %%
nyc_graph = graph_from_routes.copy()

# %%
import math
edge_info = {}
l = 50 #km/h#/ 3600#m/h
# Iterate over edges and add distances to the dictionary
for u, v, d in nyc_graph.edges(data=True):
    edge_info[(u, v)] = [
        int(d["length"]),
        d["length"] / d["maxspeed"] if not math.isnan(d["maxspeed"]) else d["length"]/l,
        50
    ]
#for k in edge_info.keys():
#    print(edge_info[k][0])

# %%
#edge_info = {}
#l = 50 #km/h#/ 3600#m/h
# Iterate over edges and add distances to the dictionary
#for u, v, d in nyc_graph.edges(data=True):
#    edge_info[(u, v)] = [
#        d["length"]/10,
#        d["length"]/500,
#        50
#    ]

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
    def __init__(self, id, people, start, end,  py=0,px=0, dy=0,dx=0) -> None:
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
def initialize_vehicles_with_specific_start_end(depots, people=4,end_depots = {42432963:42430044,42430044: 6177439750, 6223571524:42432963, 3786901738:42432963, 6177439750:6223571524}):
    vehicles = []
    n_vehicles = sum(d.vehicles_now for d in depots)
    vehicle_id_counter = 0
    for i, d in enumerate(depots):
        start_depot_id = d.id
        end_depot_id = end_depots[start_depot_id]
        for _ in range(d.vehicles_now):
            vehicles.append(Vehicle(id=vehicle_id_counter,
                                    pcapacity=people,
                                    gcapacity=10,
                                    R_=1,
                                    ctype = d.type,
                                    start_depot=start_depot_id,
                                    end_depot=end_depot_id))
            vehicle_id_counter += 1
    return vehicles, n_vehicles


# %%
depots = initialize_depots(nyc_graph, vehicles_amount = [1]*5)
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
def clean_requests(requests, depots, road_network):

    tot = 0
    for r in requests:
        #if r.people_ >4:
        #    requests.remove(r)
        #    continue
            
    
        for d in depots:
            if r.end == d.id or r.start == d.id:
                requests.remove(r)
                continue
                

        if (r.start, r.end) not in road_network.edges():
            requests.remove(r)
            continue
            
        
        tot = r.people_ + tot

    return requests, tot

# %%
edges_to_remove = [(u, v) for u, v in nyc_graph.edges() if u == v]
nyc_graph.remove_edges_from(edges_to_remove)

# %%
def check_road(graph, id, depot_id, V,s):
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
def insert_random_end_point(requests, graph):
    for r in requests:
        r.end = random.choice([x for x in graph.nodes() if x != r.start])
    return requests
    

# %%
p, _, requests = get_requests(df[:300])

depots = initialize_depots(nyc_graph, vehicles_amount = [4]*5)
#requests, tot = clean_requests(requests, depots, nyc_graph)
print(len(requests))
#random.shuffle(requests)
vehicles, _ = initialize_vehicles_with_specific_start_end(depots, people=2)


# %%
len(nyc_graph.edges)

# %%
#for r in requests:
#    if (r.start,r.end) not in nyc_graph.edges():
#        nyc_graph.add_edge(r.start,r.end)
#    if (r.end,r.start) not in nyc_graph.edges():
#        nyc_graph.add_edge(r.end,r.start)

for r in requests:
    #if r.people_ != 2:
       r.people_=2


# %%
test_vehicles, _ = initialize_vehicles_with_specific_start_end(depots)

# %%
test_g = nx.DiGraph()

# Add nodes
test_g.add_nodes_from(range(5,20))

# Add edges between all pairs of nodes
for i in range(5,20):
    for j in range(i+1, 20):
        test_g.add_edge(i, j)
        test_g.add_edge(j, i)

# %%
def initiate_f_x(road_network, avs, nyc = False):
    previous_x = {}
    previous_f = {}
    for a in avs:
        for i,j in road_network.edges():
            previous_x[i,j,a.id] = 0
        for i in road_network.nodes():
            previous_f[i,a.id] = 0
        
        if nyc:
            previous_f[a.start_depot,a.id] += 1
            continue
        previous_f[a.id+5,a.id] += 1
        #print(a.id+5, a.id)
    return previous_f, previous_x

# %%
previous_f_nyc,_ = initiate_f_x(nyc_graph, vehicles, nyc = True)

# %%

#previous_f, _ = initiate_f_x(test_g, vehicles)
#for i in previous_f.keys():
#    if previous_f[i] == 1:
#        print(previous_f[i],i)
#previous_f

# %%
test_requests = []
for i in range(5):
    test_requests += [
        Request(id = i, people = 2, start=i+7, end=i+8 )
    ]

def nodes_directly_connected(graph, node, am = -1):
    connected_nodes = []
    for neighbor in graph.neighbors(node):
        if node in graph.neighbors(neighbor):
            connected_nodes.append(neighbor)
    return connected_nodes[:am]

def create_gts_graph(important_points,road_network, point = 2, r = 7, am=3):
    modified_G = nx.Graph()
    modified_G.add_nodes_from(important_points)
    

    #creates important connections
    immidiate_connections = {}
    for node in modified_G.nodes():
        immidiate_connections[node] = nodes_directly_connected(road_network, node)
    n_copy = list(modified_G.nodes()).copy()
    for n in n_copy:
        modified_G.add_nodes_from(immidiate_connections[n])
        modified_G.add_edges_from([(n, m) for m in immidiate_connections[n]])

    if point ==0:
        return modified_G
    
    for i in range(r):
        immidiate_connections = {}
        for node in modified_G.nodes():
            immidiate_connections[node] = nodes_directly_connected(road_network, node, am=am)
        n_copy = list(modified_G.nodes()).copy()
        for n in n_copy:
            modified_G.add_nodes_from(immidiate_connections[n])
            modified_G.add_edges_from([(n, m) for m in immidiate_connections[n]])
    
    if point==1:
        return modified_G
    
    nodes_to_remove = ["a"]
    while(len(nodes_to_remove)>0):
        nodes_to_remove = [node for node, degree in dict(modified_G.degree()).items() if degree == 1 and node not in important_points]
        # Remove those nodes from the graph
        modified_G.remove_nodes_from(nodes_to_remove)
    

    #Adding info
    for n in modified_G.nodes():
        a = road_network.nodes[n]
        modified_G.nodes[n]["depot"] = a["depot"]
    for u, v, d in nyc_graph.edges(data=True):
        if (u,v) in modified_G.edges():
            modified_G[u][v]["length"] = d["length"]
            modified_G[u][v]["maxspeed"] = d["maxspeed"]
    #making it directed
    directed_graph = nx.DiGraph()
    for node, data in modified_G.nodes(data=True):
        directed_graph.add_node(node, **data)

    # Add edges from the undirected graph and copy over the "length" attribute
    for u, v, data in modified_G.edges(data=True):
        directed_graph.add_edge(u, v, **data)
        directed_graph.add_edge(v, u, **data)  # Add the reverse edge with the same attributes


    return directed_graph




# %%
def get_se_list(requests):
    starts = []
    ends = []
    for r in requests:
        starts += [r.start]
        ends += [r.end]
    return starts, ends

def convert_to_time_list(requests, N):
    # Calculate the number of requests per interval
    requests_per_interval = len(requests) // N
    # Initialize a dictionary to store requests for each time interval
    time_requests = {t: [] for t in range(N)}
    
    # Iterate over each time interval
    for t in range(N):
        # Calculate the start and end indices for the requests in this interval
        start_index = t * requests_per_interval
        end_index = (t + 1) * requests_per_interval
        
        # Add requests to the current time interval
        time_requests[t] = requests[start_index:end_index]
    
    return time_requests

# %%
def set_up_opt_problem(road_network, vehicles, time_delta, previous_f,requests,d,previous_x={}, N = 10,edge_info={},
                       l_ij = 60,v_th = 20,v_max = 30,eps = 0.5, requests_per_interval=True
                       ):
    
    problem = pulp.LpProblem("MPC_ATOD", pulp.LpMinimize)

    b = (l_ij - eps) / (v_max - v_th) 

    V = {} 
    v = {}
    w = {}
    x = {}
    s = {}
    f = {}
    more_l = {}
    M = 10000000000


    for t in range(N):
        for i,j in road_network.edges():
            V[i,j, t] = pulp.LpVariable(f"V_{i},{j}-{t}", cat = "Integer", lowBound=0, upBound=v_max) 
            s[i,j, t] = pulp.pulp.LpVariable(f"s_{i},{j}-{t}", upBound=60, lowBound=eps) 
            more_l[i,j, t] = pulp.pulp.LpVariable(f"more_l_{i},{j}-{t}", cat = "Binary") 


            problem += s[(i, j,t)] >= l_ij- M * (1-more_l[(i, j,t)]) 
            problem += s[(i, j,t)] <= l_ij -  (l_ij - eps) / (v_th - v_max) *  (v_th- V[i,j,t]) #- M * more_l[(i, j)]
            problem += s[(i, j,t)] >= l_ij -  (l_ij - eps) / (v_th - v_max) *  (v_th- V[i,j,t]) - M * more_l[(i, j,t)]
            
            for a in vehicles:
                w[i,j,a.id,t] = pulp.pulp.LpVariable(f"w_{i},{j}-{a.id}-{t}", cat='Binary') 
                v[i,j,a.id,t] = pulp.pulp.LpVariable(f"v_{i},{j}-{a.id}-{t}", cat='Binary') 
                x[i,j,a.id,t] = pulp.pulp.LpVariable(f"x_{i},{j}-{a.id}-{t}",lowBound=0) 
                if t ==0:
                    problem += x[i,j,a.id,0] == 0
                    problem += x[i,j,a.id,0] == 0
                    problem += v[i,j,a.id,0] == 0
                    problem += w[i,j,a.id,0] == 0
                    continue
                problem+=  x[i,j,a.id,t] <= M*(w[i,j,a.id,t-1]+ v[i,j,a.id,t-1])
                problem += x[i,j,a.id,t] >= x[i,j,a.id,t-1]*time_delta + s[(i, j,t-1)] - M * (1-pulp.lpSum((w[i,j,a.id,t-1], v[i,j,a.id,t-1])))
                problem += x[i,j,a.id,t] <= x[i,j,a.id,t-1]*time_delta + s[(i, j,t-1)] 



            problem += V[i,j,t] == pulp.lpSum(w[i,j,a.id,t]+ v[i,j,a.id,t] for a in vehicles)
            
    
    print("First Done")
    
    
    arrived = {}
    for (i, j) in road_network.edges():
        for a in vehicles:
            for t in range(0, N):
                arrived[i,j,a.id,t] = pulp.pulp.LpVariable(f"arrived_{i},{j}-{a.id}-{t}", cat='Binary')
                if t ==0:
                    problem += arrived[i,j,a.id,t] == 0
                    continue

                if edge_info == {}:
                    problem+= x[i,j,a.id,t] - arrived[i,j,a.id,t]*d >=0
                    problem+= x[i,j,a.id,t] -d+1- arrived[i,j,a.id,t]*M<=0
                else:
                    problem+= x[i,j,a.id,t] - arrived[i,j,a.id,t]*edge_info[i,j][0] >=0
                    problem+= x[i,j,a.id,t] -edge_info[i,j][0] +1- arrived[i,j,a.id,t]*M<=0

                problem+=arrived[i,j,a.id,t] >= arrived[i,j,a.id,t-1] -1
                problem+=arrived[i,j,a.id,t] <= 1-arrived[i,j,a.id,t-1]

    #print(arrived)

    #for (i, j) in road_network.edges():
    #    for a in vehicles:
    #        for t in range(1, N):
    #            problem += w[i,j,a.id,t-1] >=  w[i,j,a.id,t] 
    #            problem += w[i,j,a.id,t-1] - x[i,j,a.id,t-1] *1/d <=  w[i,j,a.id,t] 
    #            problem += v[i,j,a.id,t-1] <=  v[i,j,a.id,t] 
    #            problem += v[i,j,a.id,t-1] - x[i,j,a.id,t-1]*1/d <=  v[i,j,a.id,t] 
                

    for (i, j) in road_network.edges():
            for a in vehicles:
                for t in range(1, N):
                    problem += w[i,j,a.id,t] <= 1 - v[i,j,a.id,t]
                    problem += w[i,j,a.id,t] <= 1 - v[i,j,a.id,t-1]
                    problem += w[i,j,a.id,t] <= 1 - arrived[i,j,a.id,t-1]
                    problem += w[i,j,a.id,t] >= v[i,j,a.id,t] + v[i,j,a.id,t-1] +arrived[i,j,a.id,t-1] -2
#
                    problem += v[i,j,a.id,t] <= 1 - w[i,j,a.id,t]
                    problem += v[i,j,a.id,t] <= 1 - w[i,j,a.id,t-1]
                    problem += v[i,j,a.id,t] <= 1 - arrived[i,j,a.id,t-1]
                    problem += v[i,j,a.id,t] >= w[i,j,a.id,t] + w[i,j,a.id,t] +arrived[i,j,a.id,t-1] -2

    
    print("Second Done")

    departed = {}
    for t in range(1,N+1):
        for (i, j) in road_network.edges():
            for a in vehicles:
                departed[i,j,a.id,t-1] = pulp.pulp.LpVariable(f"departed_{i},{j}-{a.id}-{t-1}", cat='Binary')
                if t==1:
                    problem +=departed[i,j,a.id,t-1] == (v[i,j,a.id,t-1]+w[i,j,a.id,t-1])
                else:
                    problem +=departed[i,j,a.id,t-1] <= (v[i,j,a.id,t-1]+w[i,j,a.id,t-1]) 
                    problem +=departed[i,j,a.id,t-1] <= (1-(v[i,j,a.id,t-2]+w[i,j,a.id,t-2]) )
                    problem +=departed[i,j,a.id,t-1] >= (v[i,j,a.id,t-1]+w[i,j,a.id,t-1]) +(1-(v[i,j,a.id,t-2]+w[i,j,a.id,t-2]))-1
                    #problem +=departed[i,j,a.id,t-1] <= (v[i,j,a.id,t-1]+w[i,j,a.id,t-1]) - (v[i,j,a.id,t-2]+w[i,j,a.id,t-2])+1
                    
                    #problem +=departed[i,j,a.id,t-1] == arrived[i,j,a.id,t-2]


    f = {}
    for t in range(N+1):
        for a in vehicles:
            for i in road_network.nodes(): 
                f[i,a.id, t] = pulp.pulp.LpVariable(f"f{i}-{a.id}-{t}", cat='Binary') 
                if t ==0:
                    problem+= f[i,a.id,t] == previous_f[i,a.id]
                    continue   
                if t < N+1:
                    problem+= f[i,a.id,t] == f[i,a.id,t-1] \
                                                - pulp.lpSum((departed[i,j,a.id,t-1]) for j in road_network.nodes() if (i,j) in road_network.edges())\
                                                + pulp.lpSum((arrived[j,i,a.id,t-1]) for j in road_network.nodes() if (j,i) in road_network.edges())
                    continue
                problem+= f[i,a.id,t] == f[i,a.id,t-1] \
                                                + pulp.lpSum((arrived[j,i,a.id,t-1]) for j in road_network.nodes() if (j,i) in road_network.edges())


                
                #to stop the vehicles, one can impose this constraint
                #problem+= f[i,a.id,t] == pulp.lpSum((arrived[j,i,a.id,t-1]) for j in road_network.nodes() if (j,i) in road_network.edges())

    print("Third Done")
    
    
    for t in range(1,N):
         for a in vehicles:
                 problem +=pulp.lpSum(pulp.lpSum(f[i, a.id,t]for i in road_network.nodes())\
                                +pulp.lpSum(w[i,j, a.id,t-1]for i,j in road_network.edges())\
                                +pulp.lpSum(v[i,j, a.id,t-1]for i,j in road_network.edges())\
                                #-pulp.lpSum((arrived[i,j,a.id,t-1]) for i,j in road_network.edges())\
                                #pulp.lpSum((departed[i,j,a.id,t-1]) for i,j in road_network.edges())
                             )==1 

                      
    op = {}
    temp = {}
    request_per_time = convert_to_time_list(requests, N)
    
    for t in range(N+1):
        total_people = {}
        request_to_consider = requests if not requests_per_interval else request_per_time[t]
        for r in request_to_consider:
            if (r.start, t) not in op.keys():
                op[r.start, t] = pulp.LpVariable(f"op{r.start}_{t}", cat="Integer", lowBound=0)
                #if requests_per_interval and t>0:
                #    for tt in range(t-1,0,-1):
                #        if (r.start, tt) not in op.keys():
                #            op[r.start, tt] = pulp.LpVariable(f"op{r.start}_{tt}", cat="Integer", lowBound=0)
                #            problem += op[r.start, tt] ==0
            
            if r.start not in total_people.keys():
                total_people[r.start] = 0
            
            #creating temp
            for a in vehicles:    
                for i in road_network.nodes():
                    if (r.start,i) not in road_network.edges():
                        continue
                    if (r.start,i, a.id, t) not in temp.keys():
                        temp[r.start, i, a.id,t] = pulp.LpVariable(f"temp{r.start}_{i}_{a.id}_{t}", cat="Binary", lowBound=0)
                        if t ==0:
                            problem += temp[r.start, i, a.id,t] ==0
                            continue
                        #if requests_per_interval and t>0:
                        #    for tt in range(t-1,0,-1):
                        #        if (r.start,i, a.id, tt) not in temp.keys():
                        #            temp[r.start, i, a.id, tt] = pulp.LpVariable(f"temp{r.start}_{i}_{a.id}_{tt}", cat="Binary", lowBound=0)
                        #            problem +=temp[r.start, i, a.id, tt] ==0

                        problem +=temp[r.start,i, a.id,t-1] <= departed[r.start,i, a.id, t-1] 
                        problem +=temp[r.start, i,a.id,t-1] <= v[r.start,i, a.id, t-1] 
                        problem +=temp[r.start, i,a.id,t-1] >= v[r.start,i, a.id, t-1] + departed[r.start,i, a.id, t-1] -1
                                                
            total_people[r.start]+= r.people_
        
    print(total_people)
    #
    for k in total_people.keys():
        problem += op[k, 0]== total_people[k]
    for t in range(1,N+1):
        for k in total_people.keys():
            #to_add = 0 if not requests_per_interval else total_people[k]
            problem += op[k, t] == op[k, t - 1] - pulp.lpSum(a.pcapacity_*temp[k, i,a.id, t-1] for a in vehicles for i in road_network.nodes() if (k,i) in road_network.edges() ) #+ to_add



    print("Fourth Done") 
    #for t in range(0,N):
    #     for a in vehicles:
    #        for r in requests:
    #            problem += pulp.lpSum(a.pcapacity_ *v[r.start, r.end, a.id, t]) <= op[r.start, t]


    #for r in requests:
    #    for t in range(N):
    #        problem += op[r.start, t] >= pulp.lpSum(v[r.start, r.end, a.id, t] for a in vehicles)


    problem += pulp.lpSum(op[k,t] for k, t in op.keys() )# + pulp.lpSum(w[p] for p in w.keys()  )#- pulp.lpSum(f[r.end,a.id,N] for a in vehicles for r in requests)
    #problem += pulp.lpSum(w[i,j,a.id,t] for i,j in road_network.edges() for t in range(N) for a in vehicles )
    #problem += -pulp.lpSum(v[r.start, i, a.id, t] for a in vehicles for r in requests for t in range(N) for i in road_network.nodes() if (r.start,i) in road_network.edges())
    #problem += -pulp.lpSum(temp[r.start,i, a.id,t] for t in range(N) for a in vehicles for r in requests for i in road_network.nodes() if (r.start,i) in road_network.edges())
    #problem+=-pulp.lpSum(arrived[i,r.start,a.id,t] for i in road_network.nodes()for t in range(1,N)  for a in vehicles for r in requests if (i, r.start) in road_network.edges())
                
    #problem += -pulp.lpSum(departed[i,j,a.id,t] for i,j in road_network.edges() for t in range(N) for a in vehicles )
    #problem += pulp.lpSum(w[i,j,a.id,t] for i,j in road_network.edges() for t in range(N) for a in vehicles )
    #problem += pulp.lpSum(f[i,a.id,N] for i in road_network.nodes() for a in vehicles if previous_f[i,a.id]==0)\
               #+pulp.lpSum(departed[i,j,a.id,t] for i,j in road_network.edges() for a in vehicles for t in range(N))

    return problem, V, v,w,s,f, x, departed, arrived,op
        
test = False
N = 20 #3 ore
node_limit = 7
requests_to_do = requests[:120]

#print([d.id for d in depots])
reqs_per_d = {d.id:0 for d in depots}

for i,r in enumerate(requests_to_do):
    min_shortest_path = [1]*1000000
    d_short = 0
    for d in depots:
        if reqs_per_d[d.id] >= len(requests_to_do)/len(depots):
            continue
        shortest_path = nx.shortest_path(graph_from_routes, source=d.id, target=r.start, weight='length')
        if len(shortest_path) <len(min_shortest_path):
            min_shortest_path = shortest_path
            d_short = d.id
            #print(r.start, d.id)
    reqs_per_d[d_short] += 1
    #print(min_shortest_path)
    #print(f"REquest : {i}")
    #print(r.start, d_short)
    if len(min_shortest_path) > node_limit:
        r.start = min_shortest_path[node_limit]
    #print(r.start, d_short)
#print(reqs_per_d)
important_points = [d.id for d in depots] + [r.start for r in requests_to_do]



for t in range(N+1):
    total_people = {}
    request_to_consider = requests_to_do
    for r in request_to_consider:
        if r.start not in total_people.keys():
            total_people[r.start] = 0                                        
        total_people[r.start]+= r.people_



simplified_nyc = create_gts_graph(important_points,nyc_graph, r = 4, am=3)

if not test:
    problem, V, v,w,s,f, x, departed, arrived,op = set_up_opt_problem(simplified_nyc, vehicles,requests=requests_to_do,
                                                                N = N, time_delta=0.5, previous_f = previous_f_nyc, d = 30, requests_per_interval = False, edge_info = {})

    # %%
    print("problem created")
    solver = pulp.GUROBI_CMD(msg = 1)

    # %%
    print("problem solving.....")
    solution = problem.solve(solver = solver)

    # %%
    graph = nyc_graph
if test:
    tot = 0
    for i,r in enumerate(requests_to_do):
        tot+= r.people_
        #print(i, r.people_, r.start, r.end)

     

# %%
def save_info(graph,v,w,V,f, op,requests, N,folder_path,name_road = 'road_info.csv',  name_edge_info = "edge_info.csv", name_request_info = "requests_info.csv"):
    for t in range(N):
        road_info = {}
        for i, j in graph.edges():
            road_info[f"{i}-{j}"] = pulp.value(V[i,j,t])
        df = pd.DataFrame.from_dict(road_info, orient='index', columns=["use"]).reset_index().rename(columns={'index': 'edge'})
        df.to_csv(os.path.join(folder_path, f"{t:04d}_{name_road}"))


    for t in range(N):
        road_info = {}
        for i, j in graph.edges():
            edge_used = []
            for a in vehicles:
                edge_used.append(pulp.value(v[i,j,a.id,t]))
                
            road_info[f"{i}-{j}"] = edge_used

        df = pd.DataFrame.from_dict(road_info, orient='index', columns=[a.id for a in vehicles]).reset_index().rename(columns={'index': 'edge'})
        df.to_csv(os.path.join(folder_path, f"{t:04d}_vehicles_{name_road}"))

    for t in range(N):
        road_info = {}
        for i, j in graph.edges():
            edge_used = []
            for a in vehicles:
                edge_used.append(pulp.value(w[i,j,a.id,t]))
            road_info[f"{i}-{j}"] = edge_used
        df = pd.DataFrame.from_dict(road_info, orient='index', columns=[a.id for a in vehicles]).reset_index().rename(columns={'index': 'edge'})
        df.to_csv(os.path.join(folder_path, f"{t:04d}_vehicles_rebalancing_{name_road}"))

    for t in range(N):
        road_info = {}
        for i in graph.nodes():
            edge_used = []
            for a in vehicles:
                edge_used.append(pulp.value(f[i,a.id,t]))
            road_info[f"{i}"] = edge_used

        df = pd.DataFrame.from_dict(road_info, orient='index', columns=[a.id for a in vehicles]).reset_index().rename(columns={'index': 'edge'})
        df.to_csv(os.path.join(folder_path, f"{t:04d}_stationed_vehicles_{name_road}"))


    for t in range(N+1):
        requests_info = {}
        for k, tt in op.keys():
            if tt!=t:
                continue
            if tt>t:
                break
            requests_info[k] = pulp.value(op[k, t])
            #requests_info[r.id] = [
            #                        pulp.value(op[r.start, t]),
            #                        r.start,
            #                        #[ x.id for x in vehicles if sum([v[r.start,i,x.id,t] for i in graph.nodes() if (r.start, i) in graph.edges()])==1  ]
            #                        ]

        
        pd.DataFrame.from_dict(requests_info, orient='index', columns=[
                                                        "quantity",
                                                        #"start",
                                                        #"end",
                                                        #‚"which_vehicle"
                                                    ]).reset_index().rename(columns={'index': 'start'}).to_csv(os.path.join(folder_path, f"{t:04d}_{name_request_info}"))


        

    
if not test:
    folder_path = check_and_create_folder("analysis_on_simplified_graph", name_fold = "simu")
    save_info(graph = simplified_nyc,
                v = v,
                w = w,
                V = V, 
                f=f,
                op = op,
                requests=requests_to_do,
                N = N,
                folder_path = folder_path)
        

    # %%




from hashlib import new
from xml.dom.minidom import Element
import numpy as np
from environment import State
import sys
from functools import cmp_to_key


def _filter_instance(observation: State, mask: np.ndarray):
    res = {}

    for key, value in observation.items():
        if key in ('observation', 'static_info'):
            continue

        if key == 'capacity':
            res[key] = value
            continue

        if key == 'duration_matrix':
            res[key] = value[mask]
            res[key] = res[key][:, mask]
            continue

        res[key] = value[mask]

    return res

# Locate the k nearest neighbors
def get_time_neighbors(duration_matrix, i, num_neighbors, get_furthest=False):
    distances = list()
    for j in range(len(duration_matrix)):
        dist = duration_matrix[i][j] + duration_matrix[j][i]
        distances.append((j, dist))
    distances.sort(key=lambda tup: tup[1], reverse=get_furthest)
    neighbors = list()
    max_neighbors = min(num_neighbors, len(duration_matrix))
    for i in range(max_neighbors):
        neighbors.append(distances[i][0])
    return neighbors

def _greedy(observation: State, rng: np.random.Generator):
    mask = np.copy(observation['must_dispatch'])
    mask[:] = True
    return _filter_instance(observation, mask)


def _lazy(observation: State, rng: np.random.Generator):
    mask = np.copy(observation['must_dispatch'])
    mask[0] = True
    return _filter_instance(observation, mask)


def _random(observation: State, rng: np.random.Generator):
    mask = np.copy(observation['must_dispatch'])
    mask = (mask | rng.binomial(1, p=0.5, size=len(mask)).astype(np.bool8))
    mask[0] = True
    return _filter_instance(observation, mask)

def _random25(observation: State, rng: np.random.Generator):
    mask = np.copy(observation['must_dispatch'])
    mask = (mask | rng.binomial(1, p=0.25, size=len(mask)).astype(np.bool8))
    mask[0] = True
    return _filter_instance(observation, mask)

def _random75(observation: State, rng: np.random.Generator):
    mask = np.copy(observation['must_dispatch'])
    mask = (mask | rng.binomial(1, p=0.75, size=len(mask)).astype(np.bool8))
    mask[0] = True
    return _filter_instance(observation, mask)
    
def _random85(observation: State, rng: np.random.Generator):
    mask = np.copy(observation['must_dispatch'])
    mask = (mask | rng.binomial(1, p=0.85, size=len(mask)).astype(np.bool8))
    mask[0] = True
    return _filter_instance(observation, mask)

def _random95(observation: State, rng: np.random.Generator):
    mask = np.copy(observation['must_dispatch'])
    mask = (mask | rng.binomial(1, p=0.95, size=len(mask)).astype(np.bool8))
    mask[0] = True
    return _filter_instance(observation, mask)

def _knearest_coords(observation: State, rng: np.random.Generator):
    mask = np.copy(observation['must_dispatch'])
    mask[0] = True
    
    # Example of getting neighbors for an instance
    from math import sqrt
    
    # calculate the Euclidean distance between two vectors
    def euclidean_distance(row1, row2):
    	distance = 0.0
    	for i in range(len(row1)-1):
            distance += (row1[i] - row2[i])**2
    	return sqrt(distance)
    
    # Locate the most similar neighbors
    def get_neighbors(train, test_row, num_neighbors):
    	distances = list()
    	for j in range(len(train)):
            train_row = train[j]
            dist = euclidean_distance(test_row, train_row)
            distances.append((j, dist))
    	distances.sort(key=lambda tup: tup[1])
    	neighbors = list()
    	for i in range(num_neighbors):
    		neighbors.append(distances[i][0])
    	return neighbors
    
    # Obtain k neighbors for each customer with 'must_dispatch'
    k = 6
    for i in range(len(observation['must_dispatch'])):
        #if observation['must_dispatch'][i] == True or i==0:
        if mask[i] == True:
            neighbors = get_neighbors(observation['coords'], observation['coords'][i], k)
            for neighbor in neighbors:
                mask[neighbor] = True
    
    for i in range(len(observation['must_dispatch'])):
        #if observation['must_dispatch'][i] == True or i==0:
        if mask[i] == True:
            neighbors = get_neighbors(observation['coords'], observation['coords'][i], k)
            for neighbor in neighbors:
                mask[neighbor] = True
    
    return _filter_instance(observation, mask)

def _knearest_time(observation: State, rng: np.random.Generator):
    mask = np.copy(observation['must_dispatch'])
    mask[0] = True
    
    
    
    # Locate the k nearest neighbors
    def get_time_neighbors(duration_matrix, i, num_neighbors):
    	distances = list()
    	for j in range(len(duration_matrix)):
            dist = duration_matrix[i][j] + duration_matrix[j][i]
            distances.append((j, dist))
    	distances.sort(key=lambda tup: tup[1])
    	neighbors = list()
    	for i in range(num_neighbors):
    		neighbors.append(distances[i][0])
    	return neighbors
    
    # Obtain k neighbors for each customer with 'must_dispatch'
    k = len(observation['must_dispatch'])//10
    for i in range(len(observation['must_dispatch'])):
        #if observation['must_dispatch'][i] == True or i==0:
        if mask[i] == True:
            neighbors = get_time_neighbors(observation['duration_matrix'], i, k)
            for neighbor in neighbors:
                mask[neighbor] = True
    
    return _filter_instance(observation, mask)

# def _modified_knearest_time(observation: State, rng: np.random.Generator):
#     mask = np.copy(observation['must_dispatch'])
#     new_mask = np.copy(observation['must_dispatch'])
#     mask[0] = True
#     new_mask[0] = True
#
#
#
#     # Locate the k furthest neighbors
#     def get_furthest_neighbors(duration_matrix, i, num_neighbors):
#     	distances = list()
#     	for j in range(len(duration_matrix)):
#             dist = duration_matrix[i][j] + duration_matrix[j][i]
#             distances.append((j, dist))
#     	distances.sort(key=lambda tup: tup[1], reverse=True)
#     	neighbors = list()
#     	for i in range(num_neighbors):
#     		neighbors.append(distances[i][0])
#     	return neighbors
#
#
#     def modify_mask_of_neighbors(mask, k):
#         new_mask = np.copy(mask)
#         for i in range(len(observation['must_dispatch'])):
#             if mask[i] == True:
#                 if i == 0: #If depot, get furthest neighbors instead of nearest
#                     must_dispatches = sum(mask)
#                     if must_dispatches < 10:
#                         #neighbors = get_furthest_neighbors(observation['duration_matrix'], i, k)
#                         neighbors = get_time_neighbors(observation['duration_matrix'], i, k)
#                         for neighbor in neighbors:
#                             new_mask[neighbor] = True
#                 else:
#                     neighbors = get_time_neighbors(observation['duration_matrix'], i, k)
#                     for neighbor in neighbors:
#                         new_mask[neighbor] = True
#         return new_mask
#
#     # Obtain k neighbors for each customer with 'must_dispatch'
#     k = 6
#     new_mask = modify_mask_of_neighbors(new_mask, k)
#
#     limit_iterations = 0
#     while (sum(new_mask) < len(observation['must_dispatch'])*.95):
#         k = k+1
#         new_mask = modify_mask_of_neighbors(new_mask, k)
#         limit_iterations += 1
#         if limit_iterations >= 1:
#             break
#
#
#     return _filter_instance(observation, new_mask)

def _find_solitary(observation: State, rng: np.random.Generator):
    mask = np.copy(observation['must_dispatch'])
    mask[0] = True
    
    #Determine average distance between customers in instance
    def get_radius_quantiles(duration_matrix):
        matrix = np.array(duration_matrix)
        q_10 = np.quantile(matrix, 0.10)
        q_15 = np.quantile(matrix, 0.15)
        q_25 = np.quantile(matrix, 0.25)
        q_50 = np.quantile(matrix, 0.5)
        q_75 = np.quantile(matrix, 0.75)
        return [q_10, q_15, q_25, q_50, q_75]
                
    
    # Determine customers within radius of a customer
    def get_customers_in_radius(duration_matrix, i, radius):
        customers_in_radius = list()
        for j in range(len(duration_matrix)):
            if i == j:
                continue
            elif (duration_matrix[i][j] <= radius or duration_matrix[j][i] <= radius):
                customers_in_radius.append(j)
        
        return customers_in_radius
    
    #Set customers as 'must dispatch' if they comply with the threshold of minimum
    #customers in a given radius
    def modify_mask_of_customers(mask, radius, threshold):
        new_mask = np.copy(mask)
        for i in range(len(observation['must_dispatch'])):
            if mask[i] == True:
                continue
            else:
                customers_in_radius = get_customers_in_radius(observation['duration_matrix'], i, radius)
                if len(customers_in_radius) >= threshold:
                     new_mask[i] = True
        return new_mask
    
    def modify_mask_of_neighbors(mask, k):
        new_mask = np.copy(mask)
        for i in range(len(observation['must_dispatch'])):
            if mask[i] == True:
                if i == 0: #If depot
                    must_dispatches = sum(mask)
                    if must_dispatches < 10:
                        neighbors = get_time_neighbors(observation['duration_matrix'], i, k)
                        for neighbor in neighbors:
                            new_mask[neighbor] = True
                else:
                    neighbors = get_time_neighbors(observation['duration_matrix'], i, k)
                    for neighbor in neighbors:
                        new_mask[neighbor] = True
        return new_mask
    
    #Execute
    [q_10, q_15, q_25, q_50, q_75] = get_radius_quantiles(observation['duration_matrix'])
    threshold = len(observation['must_dispatch'])//6
    #threshold = 15
    radius = q_10
    mask = modify_mask_of_customers(mask, radius, threshold)
    new_mask = np.copy(mask)
    
    # Obtain k neighbors for each customer with 'must_dispatch'
    k = 2
    new_mask = modify_mask_of_neighbors(new_mask, k)
    
    limit_iterations = 0
    while (sum(new_mask) < len(observation['must_dispatch'])*.95):
        k = k+1
        new_mask = modify_mask_of_neighbors(new_mask, k)
        limit_iterations += 1
        if limit_iterations >= 4:
            break
    
    
    return _filter_instance(observation, new_mask)

def log(obj, newline=True, flush=False):
    # Write logs to stderr since program uses stdout to communicate with controller
    sys.stderr.write(str(obj))
    if newline:
        sys.stderr.write('\n')
    if flush:
        sys.stderr.flush()

def _get_must_dispatch(observation: State, rng: np.random.Generator): #Si no hay must_dispatch obligatorios obtengo los 10 nearest
    # log(observation['must_dispatch'])
    new_mask = np.copy(observation['must_dispatch'])
    new_mask[0] = True
    # log(new_mask)
    # log("---------------\n")
    # log(sum(new_mask) )
    """
    if(sum(new_mask) < 10): 
        neighbors = get_time_neighbors(observation['duration_matrix'], 0, 10)
        for neighbor in neighbors:
            new_mask[neighbor] = True
    """
    def modify_mask_of_neighbors(mask, k):
        new_mask = np.copy(mask)
        for i in range(len(observation['must_dispatch'])):
            if mask[i] == True:
                if i == 0: #If depot, get furthest neighbors instead of nearest
                    must_dispatches = sum(mask)
                    if must_dispatches < 10:
                        #neighbors = get_furthest_neighbors(observation['duration_matrix'], i, k)
                        neighbors = get_time_neighbors(observation['duration_matrix'], i, k)
                        for neighbor in neighbors:
                            new_mask[neighbor] = True
                else:
                    neighbors = get_time_neighbors(observation['duration_matrix'], i, k)
                    for neighbor in neighbors:
                        new_mask[neighbor] = True
        return new_mask
    
    # Obtain k neighbors for each customer with 'must_dispatch'
    k = 8
    new_mask = modify_mask_of_neighbors(new_mask, k)
    limit_iterations = 0
    while (sum(new_mask) < len(observation['must_dispatch'])*.95):
        
        k = k - 4
        new_mask = modify_mask_of_neighbors(new_mask, k)
        limit_iterations += 1
        if limit_iterations >= 1:
            break


        
    return _filter_instance(observation, new_mask)

    


def _f1(observation: State, rng: np.random.Generator, partial_routes : list, client_ids : dict ):
    
    # log(partial_routes)
    # log(observation)
    new_mask = np.copy(observation['must_dispatch'])
    unused_nodes = []
    for i in range(len(new_mask)):
        unused_nodes.append((i, False))
    # log(partial_routes)
    new_mask[0] = True
    unused_nodes[0] = (0, True) # Posicion 0 de la tupla indica la posicion se refiere a la posicion de la informacion respectiva de ese cliente en el epoch
    for i_route in range(len(partial_routes)):
        for xi in partial_routes[i_route]:
            x = client_ids[xi] #X -> posicion en el epoch
            new_mask[x] = True
            unused_nodes[x]= (x, True)  
        partial_routes[i_route] = np.insert(partial_routes[i_route], 0, 0)
        partial_routes[i_route] = np.append(partial_routes[i_route], 0)
            
    # log(partial_routes)
    
    def custom_compare1(a : tuple, b : tuple):
        xa = observation['demands'][a[0]]
        xb = observation['demands'][b[0]]
        if(xa < xb):
            return -1
        elif(xa > xb):
            return 1
        else:
            return 0
    unused_nodes = sorted(unused_nodes, key = cmp_to_key(custom_compare1)) #Aqui se sortearon por las demandas de menor a mayor
    class Route_info:
        def __init__(self, id_client, limite_inferior, limite_superior, tiempo_viaje, tiempo_servicio, tiempo_llegada, tiempo_mas_tarde):
            self.id_client = id_client #Se refiere a la posicion de la informacion respectiva de ese cliente en el epoch
            self.limite_inferior = limite_inferior
            self.limite_superior = limite_superior
            self.tiempo_viaje = tiempo_viaje #Tiempo de viaje se calcula respecto al cliente i, con el siguiente cliente i+1
            self.tiempo_servicio = tiempo_servicio
            self.tiempo_llegada = tiempo_llegada
            self.tiempo_mas_tarde = tiempo_mas_tarde
    
    def create_route_info(route : list):
        route_precompute = []
        len_route = len(route)
        occupied_capacity = 0
        for idx_client in range(len_route):
            id_client = client_ids[route[idx_client]] # route[idx_client] -> me devuelve el request_idx que identifica al cliente, client_ids[request_idx] -> me devuelve la posicion del cliente
            limite_inferior = observation['time_windows'][id_client][0]
            limite_superior = observation['time_windows'][id_client][1]
            tiempo_servicio = observation['service_times'][id_client]
            tiempo_viaje = 0 
            tiempo_llegada = 0
            tiempo_mas_tarde = 0
            
            occupied_capacity += observation['demands'][id_client]
            
            if(idx_client == len_route - 1): # Calcular tiempo viaje
                tiempo_mas_tarde = limite_superior #Inicializo precomputo de tiempo_mas_tarde
            else:
                next_client = client_ids[route[idx_client+1]]
                tiempo_viaje = observation['duration_matrix'][id_client][next_client]
            
            if(idx_client == 0): #Calcular tiempo de llegada
                tiempo_llegada = 0
            else:
                last_client = route_precompute[idx_client-1]
                tiempo_llegada = max(last_client.limite_inferior, last_client.tiempo_llegada) + last_client.tiempo_viaje + last_client.tiempo_servicio
            
            new_client_info = Route_info(id_client, limite_inferior, limite_superior, tiempo_viaje, tiempo_servicio, tiempo_llegada, tiempo_mas_tarde)
            route_precompute.append(new_client_info)
            
        for idx_client in range(len_route-2, -1, -1):#Calcula tiempo_mas_tarde
            next_client = route_precompute[idx_client+1]
            client = route_precompute[idx_client]
            second_condition = next_client.tiempo_mas_tarde - client.tiempo_servicio - client.tiempo_viaje
            tiempo_mas_tarde = min(client.limite_superior, second_condition)
            route_precompute[idx_client].tiempo_mas_tarde = tiempo_mas_tarde
        #log("------------------------------------------------------------")
        #for x in route_precompute:
            #log(f"id_client -> {x.id_client} limite_inferior -> {x.limite_inferior} limite_superior -> {x.limite_superior} tiempo_servicio -> {x.tiempo_servicio} tiempo_viaje -> {x.tiempo_viaje} tiempo_llegada -> {x.tiempo_llegada} tiempo_mas_tarde -> {x.tiempo_mas_tarde}")
        #log("------------------------------------------------------------")
        
        route_precompute = (route_precompute, occupied_capacity)
        return route_precompute
            
    routes_precompute = [] #array(tuple(array, int)) -> cada ruta en la posicion 0 tiene la informacion de la ruta y en la posicion 1 la capacidad ocupada
    for route in partial_routes:
        routes_precompute.append(create_route_info(route))
        
    def custom_compare2(a : tuple, b : tuple):
            xa = a[1]
            xb = b[1]
            if(xa < xb):
                return -1
            elif(xa > xb):
                return 1
            else:
                return 0
            
    routes_precompute = sorted(routes_precompute, key = cmp_to_key(custom_compare2)) #Aqui se sortearon por lo ocupadas que estan de menor a mayor
        
    # log(f"\n Routes capacity: {observation['capacity']} ")
    # costo_todas_las_rutas = 0
    # for route in routes_precompute:
    #      #log(f" Route total demand: {route[1]}")
    #      mostrar_clientes = []
    #      costo_ruta = 0
    #      for customer in route[0]:
    #          mostrar_clientes.append(customer.id_client)
    #          costo_ruta += customer.tiempo_viaje
    #      #log(f"Clientes: {mostrar_clientes}")
    #      log(f"Obligatory Route total demand: {route[1]}, Costo de ruta: {costo_ruta}, Clientes: {mostrar_clientes}")
    #      costo_todas_las_rutas += costo_ruta
    # log(f"Number of routes: {len(routes_precompute)}, Costo de todas las rutas: {costo_todas_las_rutas}")

    class Best_ans:
        def __init__(self, dist, id_client, route_position, left_client_position):
            self.dist = dist
            self.id_client = id_client #Posicion del cliente en los epoch
            self.route_position = route_position #La posicion de la ruta
            self.left_client_position = left_client_position
            
    
    def insert_client(new_client : Best_ans):
        new_route = routes_precompute[new_client.route_position][0]
        id_client = new_client.id_client #id del cliente nuevo q voy a insertar
        limite_inferior = observation['time_windows'][id_client][0]
        limite_superior = observation['time_windows'][id_client][1]
        tiempo_servicio = observation['service_times'][id_client]
        right_client_id =  new_route[new_client.left_client_position+1].id_client #Este sera el cliente que ira al lado derecho de mi cliente q voy agregar
        tiempo_viaje = observation['duration_matrix'][id_client][right_client_id]
        tiempo_llegada = 0
        tiempo_mas_tarde = 0
        
        
        client_route_info = Route_info(id_client, limite_inferior, limite_superior, tiempo_viaje, tiempo_servicio, tiempo_llegada, tiempo_mas_tarde)
        new_route.insert(new_client.left_client_position+1, client_route_info)
        
        
        len_route = len(new_route)
        for i in range(1, len_route): #i -> iterador de 1..n-1, j -> iterador n-2...0
            j = len_route-i-1
            #Recalcular tiempo de llegada
            last_client = new_route[i-1]
            new_route[i].tiempo_llegada = max(last_client.limite_inferior, last_client.tiempo_llegada) + last_client.tiempo_viaje + last_client.tiempo_servicio
            #Recalcular tiempo_mas_tarde
            next_client = new_route[j+1]
            client = new_route[j]
            second_condition = next_client.tiempo_mas_tarde - client.tiempo_servicio - client.tiempo_viaje
            new_route[j].tiempo_mas_tarde = min(client.limite_superior, second_condition)

        occupied_capacity = routes_precompute[new_client.route_position][1] + observation['demands'][id_client]
        routes_precompute[new_client.route_position] = (new_route, occupied_capacity)

    for node in unused_nodes:
        if node[1] == False: # Si nodo no se encuentra en la solucion actual
           continue
       
        # flag = 0
        current_node_id = node[0]
        k = 8
        neighbors = get_time_neighbors(observation['duration_matrix'], current_node_id, k)
        for neighbor in neighbors:
            node_id = neighbor
            if new_mask[node_id] == True:
                continue
            bestAns = Best_ans(1e15, -1, -1, -1)
            for n_route, route in enumerate(routes_precompute): # iterar sobre las rutas y sobre su indice
                if(n_route == 2):
                    break
                if observation['demands'][node_id]+route[1] > (0.9 * observation['capacity']): # Si supera el 90% de la capacidad del carro
                    # flag += 1 # suma una cuenta al flag para que determine si se sale del loop o no
                    continue
                for i in range(1, len(route[0])):
                    last_node = route[0][i-1]
                    next_node = route[0][i]
                    tiempo_llegada_1 = max(last_node.limite_inferior, last_node.tiempo_llegada) + observation['duration_matrix'][last_node.id_client][node_id] + observation['service_times'][last_node.id_client]
                    limite_superior_1 = observation['time_windows'][node_id][1]
                    if tiempo_llegada_1 <= limite_superior_1:
                        limite_inferior_2 = observation['time_windows'][node_id][0]
                        tiempo_llegada_2 = max(limite_inferior_2, tiempo_llegada_1) + observation['duration_matrix'][node_id][next_node.id_client] + observation['service_times'][node_id]
                        if tiempo_llegada_2 <= next_node.tiempo_mas_tarde:
                            added_distance = observation['duration_matrix'][node_id][next_node.id_client] + observation['duration_matrix'][last_node.id_client][node_id] - observation['duration_matrix'][last_node.id_client][next_node.id_client]
                            if added_distance < bestAns.dist:
                                bestAns = Best_ans(added_distance, node_id, n_route, i-1)
                    else:
                        break

            if bestAns.id_client >= 0:
                insert_client(bestAns)
                new_mask[bestAns.id_client] = True
                break

            # if flag == len(routes_precompute):
            #     break    


    
    log(f"\n Routes capacity: {observation['capacity']} ")
    costo_todas_las_rutas = 0
    for route in routes_precompute:
         #log(f"New Route total demand: {route[1]}")
         mostrar_clientes = []
         costo_ruta = 0
         for customer in route[0]:
             mostrar_clientes.append(customer.id_client)
             costo_ruta += customer.tiempo_viaje
         #log(f"Clientes: {mostrar_clientes}")
         log(f"New Route total demand: {route[1]}, Costo de ruta: {costo_ruta}, Clientes: {mostrar_clientes}")
         costo_todas_las_rutas += costo_ruta
    log(f"Number of routes: {len(routes_precompute)}, Costo de todas las rutas: {costo_todas_las_rutas}")

    return _filter_instance(observation, new_mask)    

def _supervised(observation: State, rng: np.random.Generator, net):
    from baselines.supervised.transform import transform_one
    mask = np.copy(observation['must_dispatch'])
    mask = mask | net(transform_one(observation)).argmax(-1).bool().numpy()
    mask[0] = True
    return _filter_instance(observation, mask)


def _dqn(observation: State, rng: np.random.Generator, net):
    import torch
    from baselines.dqn.utils import get_request_features
    actions = []
    epoch_instance = observation
    observation, static_info = epoch_instance.pop('observation'), epoch_instance.pop('static_info')
    request_features, global_features = get_request_features(observation, static_info, net.k_nearest)
    all_features = torch.cat((request_features, global_features[None, :].repeat(request_features.shape[0], 1)), -1)
    actions = net(all_features).argmax(-1).detach().cpu().tolist()
    mask = epoch_instance['must_dispatch'] | (np.array(actions) == 0)
    mask[0] = True  # Depot always included in scheduling
    return _filter_instance(epoch_instance, mask)

def _all_must_dispatch(observation: State, rng: np.random.Generator):  # Si no hay must_dispatch obligatorios obtengo los 10 nearest
    # log(observation['must_dispatch'])
    new_mask = np.copy(observation['must_dispatch'])
    for i in range(len(new_mask)):
        new_mask[i] = True
    return _filter_instance(observation, new_mask)

def _f2(observation: State, rng: np.random.Generator, partial_routes: list, client_ids: dict, omega: float):
    # log(partial_routes)
    # log(observation)
    log(f"omega: {omega}")
    new_mask = np.copy(observation['must_dispatch'])  # REVISAR SI SI SE ESTA COPIANDO BIEN LA MASCARA
    new_mask[0] = True
    average_costs = []
    omega = omega
    for i in range(len(partial_routes)):
        log(partial_routes[i])
        partial_routes[i] = np.insert(partial_routes[i], 0, 0)
        partial_routes[i] = np.append(partial_routes[i], 0)
        route_info = []
        for j in range(len(partial_routes[i])-1):
            client = client_ids[partial_routes[i][j]]
            next_client = client_ids[partial_routes[i][j+1]]
            route_info.append(observation['duration_matrix'][client][next_client])
        average_costs.append(np.mean(route_info))

    for i in range(len(partial_routes)):
        for j in range(1, len(partial_routes[i]) - 1):
            if new_mask[client_ids[partial_routes[i][j]]] is not True:
                last_node = client_ids[partial_routes[i][j-1]]
                last_node_cost = observation['duration_matrix'][last_node][client_ids[partial_routes[i][j]]]
                next_node = client_ids[partial_routes[i][j+1]]
                next_node_cost = observation['duration_matrix'][client_ids[partial_routes[i][j]]][next_node]
                saved_cost = last_node_cost + next_node_cost - observation['duration_matrix'][last_node][next_node]
                if saved_cost < (average_costs[i]*omega):
                    new_mask[client_ids[partial_routes[i][j]]] = True
                #else:

    
    return _filter_instance(observation, new_mask)


def _knearest_time_distance(observation: State, rng: np.random.Generator, factor: float):
    mask = np.copy(observation['must_dispatch'])
    new_mask = np.copy(observation['must_dispatch'])
    mask[0] = True
    new_mask[0] = True
    log(f"parameter: {factor}")

    def get_neighbors(duration_matrix, i, num_neighbors, alpha):
        distances = list()
        for j in range(len(duration_matrix)):
            limite_superior_nodo = observation['time_windows'][i][1]
            limite_inferior_nodo = observation['time_windows'][i][0]
            limite_superior_vecino = observation['time_windows'][j][1]
            limite_inferior_vecino = observation['time_windows'][j][0]
            if limite_superior_nodo < limite_inferior_vecino:
                if limite_inferior_vecino - limite_superior_nodo <= alpha:
                    dist = duration_matrix[i][j] + duration_matrix[j][i]
                    distances.append((j, dist))
            elif limite_superior_vecino < limite_inferior_nodo:
                if limite_inferior_nodo - limite_superior_vecino <= alpha:
                    dist = duration_matrix[i][j] + duration_matrix[j][i]
                    distances.append((j, dist))
            else:
                dist = duration_matrix[i][j] + duration_matrix[j][i]
                distances.append((j, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        max_neighbors = min(num_neighbors, len(duration_matrix))
        for i in range(max_neighbors):
            neighbors.append(distances[i][0])
        return neighbors

    def modify_mask_of_neighbors(mask, k, alpha):
        new_mask = np.copy(mask)
        for i in range(len(observation['must_dispatch'])):
            if mask[i] == True:
                if i == 0:  # If depot, get furthest neighbors instead of nearest
                    must_dispatches = sum(mask)
                    if must_dispatches < 10:
                        # neighbors = get_furthest_neighbors(observation['duration_matrix'], i, k)
                        neighbors = get_neighbors(observation['duration_matrix'], i, k, alpha)
                        for neighbor in neighbors:
                            new_mask[neighbor] = True
                else:
                    neighbors = get_neighbors(observation['duration_matrix'], i, k, alpha)
                    for neighbor in neighbors:
                        new_mask[neighbor] = True
        return new_mask

    # Obtain k neighbors for each customer with 'must_dispatch'
    k = 6
    alpha = 3600*factor
    new_mask = modify_mask_of_neighbors(new_mask, k, alpha)

    limit_iterations = 0
    while (sum(new_mask) < len(observation['must_dispatch']) * .95):
        k = k + 1
        new_mask = modify_mask_of_neighbors(new_mask, k, alpha)
        limit_iterations += 1
        if limit_iterations >= 1:
            break

    return _filter_instance(observation, new_mask)


def _modified_knearest_time(observation: State, rng: np.random.Generator, first_epoch: int, current_epoch: int):
    mask = np.copy(observation['must_dispatch'])
    new_mask = np.copy(observation['must_dispatch'])
    mask[0] = True
    new_mask[0] = True

    # Parameter to change
    c = 4  # Number of cycle iterations of k-nearest neighbors
    alpha = 2.0  # +- Hours to consider nearest neighbor
    beta = 2  # Hours to TW closing, to mark customer as Obligatory
    k = 8  # K nearest neighbors to consider

    def modify_mask_of_urgent(mask):
        new_mask = np.copy(mask)
        for i in range(len(observation['must_dispatch'])):
            if observation['time_windows'][i][1] <= 3600 * beta:
                new_mask[i] = True

        return new_mask

    def modify_mask_of_neighbors(mask, k, alpha, get_furthest=False):
        new_mask = np.copy(mask)
        for i in range(len(observation['must_dispatch'])):
            if mask[i] == True:
                if (i == 0):
                    continue
                else:
                    neighbors = get_time_neighbors(observation['duration_matrix'], i, k, get_furthest)
                    current_tw_start = observation['time_windows'][i][0]
                    current_tw_end = observation['time_windows'][i][1]
                    for neighbor in neighbors:
                        neighbor_tw_start = observation['time_windows'][neighbor][0]
                        neighbor_tw_end = observation['time_windows'][neighbor][1]
                        if neighbor_tw_end <= current_tw_end + alpha * 3600:
                            new_mask[neighbor] = True
        return new_mask

    if len(observation['must_dispatch']) >= 150:
        c = 5
    elif len(observation['must_dispatch']) >= 100:
        c = 4
    else:
        c = 3

    if current_epoch == first_epoch:
        k = 5
        c = 2

    new_mask = modify_mask_of_neighbors(new_mask, k, alpha, get_furthest=False)

    limit_iterations = 0
    while (sum(new_mask) < len(observation['must_dispatch'])):
        k = k - 1
        new_mask = modify_mask_of_neighbors(new_mask, k, alpha)
        limit_iterations += 1
        if limit_iterations >= c:
            break

    return _filter_instance(observation, new_mask)


def _remove_clients(observation: State, rng: np.random.Generator, partial_routes: list, client_ids: dict, porcentaje: float):
    # log(partial_routes)
    # log(observation)
    log(f"omega: {porcentaje}")
    new_mask = np.copy(observation['must_dispatch'])  # REVISAR SI SI SE ESTA COPIANDO BIEN LA MASCARA
    new_mask[0] = True

    for i in range(len(partial_routes)):
        log(partial_routes[i])
        demand = 0
        must_dispatch = 0
        for j in range(len(partial_routes[i])):
            client = client_ids[partial_routes[i][j]]
            demand += observation['demands'][client]
            must_dispatch += 1 if new_mask[client] else 0
            new_mask[client] = True
        capacity = (demand/observation['capacity'])*100
        if must_dispatch == 0 and capacity <= porcentaje:
            log(f"Capacidad: {capacity}, y tiene must_dispatch {must_dispatch}")
            for j in range(len(partial_routes[i])):
                client = client_ids[partial_routes[i][j]]
                new_mask[client] = False

    return _filter_instance(observation, new_mask)

def _must_dispatch(observation: State, rng: np.random.Generator):  # Si no hay must_dispatch obligatorios obtengo los 10 nearest
    # log(observation['must_dispatch'])
    new_mask = np.copy(observation['must_dispatch'])
    new_mask[0] = True

    clients_ordered = []
    for i in range(len(new_mask)):
        if not new_mask[i]:
            clients_ordered.append((i, observation['time_windows'][i][0]))
    clients_ordered.sort(key=lambda x: x[1], reverse=True)
    log(f"clients_ordered {clients_ordered}")
    not_routed_clients = []
    for client in clients_ordered:
        not_routed_clients.append(client[0])
    log(f"clients_ordered {not_routed_clients}")
    return _filter_instance(observation, new_mask), not_routed_clients

def _remove_ordered_clients(observation: State, rng: np.random.Generator, iteration, not_routed_clients, number_of_clients):
    new_mask = np.copy(observation['must_dispatch'])
    log("----------------------------------------------")

    if iteration*number_of_clients > len(not_routed_clients):
        clients = not_routed_clients
        log(f"not_routed_clients {clients}")
    else:
        clients = not_routed_clients[:int(iteration*number_of_clients)]
        log(f"not_routed_clients {clients}")
    for i in range(len(new_mask)):
        if i not in clients:
            new_mask[i] = True

    return _filter_instance(observation, new_mask)

def _must_dispatch_modifiedknearest(observation: State, rng: np.random.Generator, first_epoch: int, current_epoch: int):
    # log(observation['must_dispatch'])
    must_dispatch = np.copy(observation['must_dispatch'])
    must_dispatch[0] = True

    #mask = np.copy(observation['must_dispatch'])
    new_mask = np.copy(observation['must_dispatch'])
    #mask[0] = True
    new_mask[0] = True

    # Parameter to change
    c = 4  # Number of cycle iterations of k-nearest neighbors
    alpha = 2.0  # +- Hours to consider nearest neighbor
    beta = 2  # Hours to TW closing, to mark customer as Obligatory
    k = 8  # K nearest neighbors to consider

    def modify_mask_of_urgent(mask):
        new_mask = np.copy(mask)
        for i in range(len(observation['must_dispatch'])):
            if observation['time_windows'][i][1] <= 3600 * beta:
                new_mask[i] = True

        return new_mask

    def modify_mask_of_neighbors(mask, k, alpha, get_furthest=False):
        new_mask = np.copy(mask)
        for i in range(len(observation['must_dispatch'])):
            if mask[i] == True:
                if (i == 0):
                    continue
                else:
                    neighbors = get_time_neighbors(observation['duration_matrix'], i, k, get_furthest)
                    current_tw_start = observation['time_windows'][i][0]
                    current_tw_end = observation['time_windows'][i][1]
                    for neighbor in neighbors:
                        neighbor_tw_start = observation['time_windows'][neighbor][0]
                        neighbor_tw_end = observation['time_windows'][neighbor][1]
                        if neighbor_tw_end <= current_tw_end + alpha * 3600:
                            new_mask[neighbor] = True
        return new_mask

    if len(observation['must_dispatch']) >= 150:
        c = 5
    elif len(observation['must_dispatch']) >= 100:
        c = 4
    else:
        c = 3

    if current_epoch == first_epoch:
        k = 5
        c = 2

    new_mask = modify_mask_of_neighbors(new_mask, k, alpha, get_furthest=False)

    limit_iterations = 0
    while (sum(new_mask) < len(observation['must_dispatch'])):
        k = k - 1
        new_mask = modify_mask_of_neighbors(new_mask, k, alpha)
        limit_iterations += 1
        if limit_iterations >= c:
            break

    clients_ordered = []
    log(f"sum = {sum(new_mask)}")
    for i in range(len(new_mask)):
        if new_mask[i] and not must_dispatch[i]:
            clients_ordered.append((i, observation['time_windows'][i][0]))
    clients_ordered.sort(key=lambda x: x[1], reverse=True)
    #log(f"clients_ordered {clients_ordered}")
    clients_to_route = []
    for client in clients_ordered:
        clients_to_route.append(client[0])
    log(f"clients_ordered {clients_to_route}, len = {len(clients_to_route)}")
    return _filter_instance(observation, must_dispatch), clients_to_route

def _remove_ordered_clients_modifiedknearest(observation: State, rng: np.random.Generator, iteration, clients_to_route, number_of_clients):
    new_mask = np.copy(observation['must_dispatch'])
    new_mask[0] = True
    log("----------------------------------------------")

    if iteration*number_of_clients > len(clients_to_route):
        not_routed_clients = clients_to_route
        log(f"not_routed_clients {not_routed_clients}")
    else:
        not_routed_clients = clients_to_route[:int(iteration*number_of_clients)]
        log(f"not_routed_clients {not_routed_clients}")
    for i in range(len(new_mask)):
        if (i in clients_to_route) and (i not in not_routed_clients):
            new_mask[i] = True

    return _filter_instance(observation, new_mask)

STRATEGIES = dict(
    greedy=_greedy,
    lazy=_lazy,
    random=_random,
    supervised=_supervised,
    dqn=_dqn,
    random25=_random25,
    random75=_random75,
    random85=_random85,
    random95=_random95,
    knearestcoords= _knearest_coords,
    knearesttime = _knearest_time,
    modifiedknearest = _modified_knearest_time,
    findsolitary = _find_solitary,
    getMustDispatch = _get_must_dispatch,
    f1 = _f1,
    allmustdispatch = _all_must_dispatch,
    f2 = _f2,
    knearestimedistance = _knearest_time_distance,
    removeclients = _remove_clients,
    mustdispatch = _must_dispatch,
    removeorderedclients = _remove_ordered_clients,
    mustdispatchmodifiedknearest= _must_dispatch_modifiedknearest,
    removeorderedclientsmodifiedknearest = _remove_ordered_clients_modifiedknearest
)

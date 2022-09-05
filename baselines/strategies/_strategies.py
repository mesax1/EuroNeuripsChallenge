from hashlib import new
from xml.dom.minidom import Element
import numpy as np
from environment import State
import sys
from functools import cmp_to_key

def _filter_instance(observation: State, mask: np.ndarray):
    res = {}

    for key, value in observation.items(): 
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
def get_time_neighbors(duration_matrix, i, num_neighbors):
    distances = list()
    for j in range(len(duration_matrix)):
        dist = duration_matrix[i][j] + duration_matrix[j][i]
        distances.append((j, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    max_neighbors = min(num_neighbors, len(duration_matrix))
    for i in range(max_neighbors):
        neighbors.append(distances[i][0])
    return neighbors

# Locate the k furthest neighbors
def get_furthest_neighbors(duration_matrix, i, num_neighbors):
    distances = list()
    for j in range(len(duration_matrix)):
        dist = duration_matrix[i][j] + duration_matrix[j][i]
        distances.append((j, dist))
    distances.sort(key=lambda tup: tup[1], reverse=True)
    neighbors = list()
    max_neighbors = min(num_neighbors, len(duration_matrix))
    for i in range(max_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


def _greedy(observation: State, rng: np.random.Generator):
    return {
        **observation,
        'must_dispatch': np.ones_like(observation['must_dispatch']).astype(np.bool8)
    }



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

def _modified_knearest_time(observation: State, rng: np.random.Generator):
    #log(observation)
    mask = np.copy(observation['must_dispatch'])
    new_mask = np.copy(observation['must_dispatch'])
    mask[0] = True
    new_mask[0] = True
    
        
    
    # Locate the k furthest neighbors
    # def get_furthest_neighbors(duration_matrix, i, num_neighbors):
    # 	distances = list()
    # 	for j in range(len(duration_matrix)):
    #         dist = duration_matrix[i][j] + duration_matrix[j][i]
    #         distances.append((j, dist))
    # 	distances.sort(key=lambda tup: tup[1], reverse=True)
    # 	neighbors = list()
    # 	for i in range(num_neighbors):
    # 		neighbors.append(distances[i][0])
    # 	return neighbors
    
    
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
    k = 7
    #k = min(6,len(observation['must_dispatch'])//(sum(new_mask)))
    #k = end_epoch + 1 - current_epoch//3
    #k = max(end_epoch, end_epoch +6 - current_epoch)
    new_mask = modify_mask_of_neighbors(new_mask, k)
    
    
    limit_iterations = 0
    while (sum(new_mask) < len(observation['must_dispatch'])*.95):
        #k = max(6,len(observation['must_dispatch'])//(sum(new_mask)//2))
        k = k - 1 
        new_mask = modify_mask_of_neighbors(new_mask, k)
        limit_iterations += 1
        if limit_iterations >= 1:
            break
    
    
    return _filter_instance(observation, new_mask)

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
                    continue
                else:
                    neighbors = get_time_neighbors(observation['duration_matrix'], i, k)
                    for neighbor in neighbors:
                        new_mask[neighbor] = True
        return new_mask
    
    #Execute
    [q_10, q_15, q_25, q_50, q_75] = get_radius_quantiles(observation['duration_matrix'])
    threshold = len(observation['must_dispatch'])//5
    radius = q_10
    mask = modify_mask_of_customers(mask, radius, threshold)
    new_mask = np.copy(mask)
    
    # Obtain k neighbors for each customer with 'must_dispatch'
    k = 6
    k = current_epoch + 3
    k = end_epoch + 2 - current_epoch//2
    new_mask = modify_mask_of_neighbors(new_mask, k)
    
    limit_iterations = 0
    while (sum(new_mask) < len(observation['must_dispatch'])*.95):
        k = k+1
        new_mask = modify_mask_of_neighbors(new_mask, k)
        limit_iterations += 1
        if limit_iterations >= 1:
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
    if(sum(new_mask) < 10): 
        neighbors = get_time_neighbors(observation['duration_matrix'], 0, 10)
        for neighbor in neighbors:
            new_mask[neighbor] = True
        
    return _filter_instance(observation, new_mask)

    


def _f1(observation: State, rng: np.random.Generator, partial_routes : list, client_ids : dict ):
    
    # log(partial_routes)
    # log(observation)
    new_mask = np.copy(observation['must_dispatch']) #REVISAR SI SI SE ESTA COPIANDO BIEN LA MASCARA
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
    
    def custom_compare(a : tuple, b : tuple):
        xa = observation['demands'][a[0]]
        xb = observation['demands'][b[0]]
        if(xa < xb):
            return -1
        elif(xa > xb):
            return 1
        else:
            return 0
    unused_nodes = sorted(unused_nodes, key = cmp_to_key(custom_compare)) #Aqui se sortearon por las demandas de menor a mayor
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
        
        
        
        log("------------------------------------------------------------")
        for x in route_precompute:
            log(f"id_client -> {x.id_client} limite_inferior -> {x.limite_inferior} limite_superior -> {x.limite_superior} tiempo_servicio -> {x.tiempo_servicio} tiempo_viaje -> {x.tiempo_viaje} tiempo_llegada -> {x.tiempo_llegada} tiempo_mas_tarde -> {x.tiempo_mas_tarde}")
        log("------------------------------------------------------------")
        
        route_precompute = (route_precompute, occupied_capacity)
        return route_precompute
            
    routes_precompute = [] #array(tuple(array, int)) -> cada ruta en la posicion 0 tiene la informacion de la ruta y en la posicion 1 la capacidad ocupada
    for route in partial_routes:
        routes_precompute.append(create_route_info(route))
        
    
    # for route in routes_precompute:
    #     log(route[1])
    class Best_ans:
        def __init_(self):
            self.dist = int(1e15)
            self.id_client = -1 #Posicion del cliente en los epoch
            self.route_position = -1 #La posicion de la ruta
            self.left_client_position = -1
            
    
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
         
    # log(partial_routes)
    # log(unused_nodes)
    # log(observation['demands'])
    
    
    
    
    
    return _filter_instance(observation, new_mask)


STRATEGIES = dict(
    greedy=_greedy,
    lazy=_lazy,
    random=_random,
    random25=_random25,
    random75=_random75,
    random85=_random85,
    random95=_random95,
    knearestcoords= _knearest_coords,
    knearesttime = _knearest_time,
    modifiedknearest = _modified_knearest_time,
    findsolitary = _find_solitary,
    getMustDispatch = _get_must_dispatch,
    f1 = _f1
)

import numpy as np
from environment import State


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
    mask = np.copy(observation['must_dispatch'])
    new_mask = np.copy(observation['must_dispatch'])
    mask[0] = True
    new_mask[0] = True
    
        
    
    # Locate the k furthest neighbors
    def get_furthest_neighbors(duration_matrix, i, num_neighbors):
    	distances = list()
    	for j in range(len(duration_matrix)):
            dist = duration_matrix[i][j] + duration_matrix[j][i]
            distances.append((j, dist))
    	distances.sort(key=lambda tup: tup[1], reverse=True)
    	neighbors = list()
    	for i in range(num_neighbors):
    		neighbors.append(distances[i][0])
    	return neighbors
    
    
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
    k = 6
    new_mask = modify_mask_of_neighbors(new_mask, k)
    
    limit_iterations = 0
    while (sum(new_mask) < len(observation['must_dispatch'])*.95):
        k = k+1
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
    findsolitary = _find_solitary
)

from turtle import colormode
from environment import State
import matplotlib.pyplot as plt
import sys
import time
import numpy



def log(obj, newline=True, flush=False):
    # Write logs to stderr since program uses stdout to communicate with controller
    sys.stderr.write(str(obj))
    if newline:
        sys.stderr.write('\n')
    if flush:
        sys.stderr.flush()
        

def traza_curvas(route_coords, observation : State, real_positions):
    x1 = []
    y1 = []
    for coords in observation['coords']:
        x1.append(coords[0])
        y1.append(coords[1])
    plt.scatter(x1, y1, s = 0.005, c = "black")
    
    for i in range(len(route_coords)):        
        x1 = []
        y1 = []
        mycolor =  numpy.random.rand(3,)
        for coords in route_coords[i]:
            x1.append(coords[0])
            y1.append(coords[1])
        plt.plot(x1, y1, label=f"Ruta {i}", lw = 0.005, c = mycolor)
        plt.scatter(x1, y1, s = 0.005, c = mycolor)
        
        for j in range(len(x1)):
            plt.text(x1[j], y1[j], real_positions[i][j], size = 2)    
    
    depot = observation['coords'][0]
    plt.scatter(depot[0], depot[1], marker= "v", s = 2 )
    plt.legend()
    name_file = "./output_images/"+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + ".pdf"
    plt.savefig(name_file)

def graficar(observation : State, routes : list, client_ids : dict):
    log(observation['coords'])
    real_positions = []
    routes_coords = []
    for route in routes : 
        temp = []
        for id in route:
            temp.append(client_ids[id])
        real_positions.append(temp)

    for route in real_positions:
        temp = []
        for id in route:
            temp.append(observation['coords'][id])
        routes_coords.append(temp)
    traza_curvas(routes_coords, observation, real_positions)
    
    

    
    
"""
This file runs Ising simulation
Created: Mar 30, 2019
Last Edited: April 6, 2019
By Bill
"""

import IsingGrid
import matplotlib as mpl
import matplotlib.pyplot as plt
from copy import deepcopy


# Fundamental parameters

size = 100
temperature = 1
steps = 400
interval = 100
Jfactor = 0.5

# Generate grid

g = IsingGrid.Grid(size, Jfactor)
g.randomize()

# Animation parameters

fig, ax = plt.subplots()
data = []

# Simulation

print("Simulation begins.")

for step in range(steps):

    # Single/cluster Filp

    # clusterSize = g.singleFlip(temperature)
    clusterSize = g.clusterFlip(temperature)

    if (step + 1) % interval == 0:
        data.append(deepcopy(g.canvas))

    if (step + 1) % (10 * interval) == 0:
        print("Step ", step + 1, "/", steps, ", Cluster size ", clusterSize, "/", size * size)

print("Simulation completes.")

# Animation

print("Animation begins.")

for frame in range(0, len(data)):
    ax.cla()
    ax.imshow(data[frame], cmap=mpl.cm.winter)
    ax.set_title("Step {}".format(frame * interval))
    plt.pause(0.01)

print("Animation completes.")

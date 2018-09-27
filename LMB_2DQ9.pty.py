from numpy import *;
from numpy.linalg import *
import matplotlib.pyplot as plt;
from matplotlib import cm

#https://en.wikipedia.org/wiki/Reynolds_number
# low == more smooth fluid | High == more chaotic and turbulantflow
Reynolds_number = 220.0

plot_width = 520
plot_height = 180

ly = plot_height - 1.0 #TBD

# number of lattice velocities
q = 9

#Cylinder coordinates
cylinder_x = plot_width / 2
cylinder_y = plot_height / 2
cylinder_radius = plot_height / 9

#Velocity in lattice units
velocity_lattice_units = 0.04

#todo Define vars
nulb = velocity_lattice_units * cylinder_radius / Reynolds_number
omega = 1.0 / (3. * nulb + 0.5)  # Relaxation parameter.

###### Lattice Constants ######
D2Q9_FAST_WEIGHTING_VECTOR = 1. / 36.
D2Q9_SLOW_WEIGHTING_VECTOR = 1. / 9.
D2Q9_REST_WEIGHTING_VECTOR = 4. / 9.

lattice_directions = array([(x, y) for x in [0, -1, 1] for y in [0, -1, 1]])

#Intit all weights to fast
lattice_weights = D2Q9_FAST_WEIGHTING_VECTOR * ones(q)

#Init the xaxis and yaxis horizontal with slow weights
lattice_weights[asarray([norm(direction) == 1. for direction in lattice_directions])] = D2Q9_SLOW_WEIGHTING_VECTOR;

#Set center particle to a rest weight
lattice_weights[0] = D2Q9_REST_WEIGHTING_VECTOR

#Find al the inverse indexes of the directions of the lattice #aka: noslip
inverse_direction_indexes = []
for direction in lattice_directions:
    inverse_direction_index = lattice_directions.tolist().index((-direction).tolist())
    inverse_direction_indexes.append(inverse_direction_index)

#Unknow indexes to the walls. todo: Find a nice way to find the indexes of lattice_directions
index_range = arange(q)
left_direction_indexes = index_range[asarray([direction[0] < 0 for direction in lattice_directions])]
vertical_direction_indexes = index_range[asarray([direction[0] == 0 for direction in lattice_directions])]
right_direction_indexes = index_range[asarray([direction[0] > 0 for direction in lattice_directions])]

sumpop = lambda fin: sum(fin, axis=0)  # Helper function for density computation.
from numpy import *;
from numpy.linalg import *
import matplotlib.pyplot as plt;
from matplotlib import cm

#https://en.wikipedia.org/wiki/Reynolds_number
# low == more smooth fluid | High == more chaotic and turbulantflow
n_iterations = 20000  # Total number of time iterations.
Reynolds_number = 4500.0
water_density = 1.0

plot_width = 250
plot_height = 150

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
relaxation_time = 1.0 / (3. * nulb + 0.5)  # Relaxation parameter.

###### Lattice Constants ######
D2Q9_FAST_WEIGHTING_VECTOR = 1. / 36.
D2Q9_SLOW_WEIGHTING_VECTOR = 1. / 9.
D2Q9_REST_WEIGHTING_VECTOR = 4. / 9.
D2Q9_SPEED = 1. / 3.

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

def equilibrium(density, velocity):  # Equilibrium distribution function.
    # To find out: why multiplying with magic number 3.0
    # Calculates velocity for each direction of each element (e.g. 2x2 plane results in 9x2x2 arrays)
    direction_velocities = 3.0 * dot(lattice_directions, velocity.transpose(1, 0, 2))

    # Calculating squared velocity by the D2Q9_SPEED property
    squared_velocity = (velocity[0] ** 2 + velocity[1] ** 2) / (2 * D2Q9_SPEED)

    # Initialising zero-matrices for each direction
    equilibrium_distribution = zeros((q, plot_width, plot_height))

    # Equilibrium mass distribution
    for i in range(q): equilibrium_distribution[i, :, :] = density * lattice_weights[i] * (1. + direction_velocities[i] + 0.5 * direction_velocities[i] ** 2 - squared_velocity)
    return equilibrium_distribution

###### Setup: cylindrical obstacle and velocity inlet with perturbation ########
obstacle = fromfunction(lambda x, y: (x - cylinder_x) ** 2 + (y - cylinder_y) ** 2 < cylinder_radius ** 2, (plot_width, plot_height))
lattice_velocities = fromfunction(lambda d, x, y: (1 - d) * velocity_lattice_units * (1.0 + 1e-4 * sin(y / ly * 2 * pi)), (2, plot_width, plot_height))
equilibrium_distribution = equilibrium(water_density, lattice_velocities)
fin = equilibrium_distribution.copy()
fig, ax = plt.subplots()
###### Main time loop ##########################################################
for time in range(n_iterations):
    #fin[left_direction_indexes, -1, :] = fin[left_direction_indexes, -2, :]  # Right wall: outflow condition.
    density = sumpop(fin)  # Calculate macroscopic density and velocity.
    velocity = dot(lattice_directions.transpose(), fin.transpose((1, 0, 2))) / density

    velocity[:, 0, :] = lattice_velocities[:, 0, :]  # Left wall: compute density from known populations.
    density[0, :] = 1. / (1. - velocity[0, 0, :]) * (sumpop(fin[vertical_direction_indexes, 0, :]) + 2. * sumpop(fin[left_direction_indexes, 0, :]))

    equilibrium_distribution = equilibrium(density, velocity)  # Left wall: Zou/He boundary condition.
    #fin[right_direction_indexes, 0, :] = fin[left_direction_indexes, 0, :] + equilibrium_distribution[right_direction_indexes, 0, :] - fin[left_direction_indexes, 0, :]
    fin[right_direction_indexes, 0, :] = equilibrium_distribution[right_direction_indexes, 0, :]
    fout = fin - relaxation_time * (fin - equilibrium_distribution)  # Collision step.
    for i in range(q): fout[i, obstacle] = fin[inverse_direction_indexes[i], obstacle]
    for i in range(q):  # Streaming step.
        fin[i, :, :] = roll(roll(fout[i, :, :], lattice_directions[i, 0], axis=0), lattice_directions[i, 1], axis=1)

    colorbar = False;

    #if (time % 100 == 0):  # Visualization
    if (time == 2000):
        # fig, ax = plt.subplots()
        # cax = plt.imshow(sqrt(velocity[0] ** 2 + velocity[1] ** 2).transpose(), cmap=cm.Reds)
        # ax.set_title('Speed of single fluid using LBM. Step:{}'.format(time))
        #
        # cbar = fig.colorbar(cax, ticks=[-1, 0, 1], orientation='horizontal')
        # cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])  # horizontal colorbar
        #
        # cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
        # cbar.ax.set_yticklabels(['< -1', '0', '> 1'])  # vertically oriented colorbar
        #
        # plt.show()

        fig, ax = plt.subplots()
        data = sqrt(velocity[0] ** 2 + velocity[1] ** 2).transpose()

        cax = ax.imshow(data, interpolation='nearest', cmap=cm.Reds)
        ax.set_title('Velocity of single fluid using LBM. Re:{} Steps:{}'.format(Reynolds_number, time))

        if (colorbar == False):
            cbar = fig.colorbar(cax, ticks=[0.03999600, amax(velocity)], orientation='vertical')
            cbar.ax.set_yticklabels(['Low', 'High'])  # horizontal colorbar
            colorbar = True

        plt.ylabel('y-nodes')
        plt.xlabel('x-nodes')
        plt.show()
        #plt.pause(0.001)
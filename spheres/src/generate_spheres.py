import numpy as np
from scipy.optimize import minimize
import sys

# CHANGE HERE ##########################################
# Define the size_interval (you can change this value as needed)
size_interval = 0.1  # Change this to your desired value
percentage_fill_with_spheres = 0.9
########################################################

# Define the objective function to maximize the minimum distance between spheres and to domain borders
def objective_function(positions):
    num_spheres = len(radii)
    min_distance_between_spheres = float("inf")
    min_distance_to_domain_borders = float("inf")

    # Calculate the minimum distance between all pairs of spheres
    for i in range(num_spheres):
        for j in range(i + 1, num_spheres):
            distance = np.linalg.norm(positions[i * 3:i * 3 + 3] - positions[j * 3:j * 3 + 3]) - radii[i] - radii[j]
            min_distance_between_spheres = min(min_distance_between_spheres, distance)

    # Calculate the minimum distance from each sphere to the domain borders
    for i in range(num_spheres):
        distance_x_min = positions[i * 3] - radii[i]
        distance_x_max = 1 - positions[i * 3] - radii[i]
        distance_y_min = positions[i * 3 + 1] - radii[i]
        distance_y_max = 1 - positions[i * 3 + 1] - radii[i]
        distance_z_min = positions[i * 3 + 2] - radii[i]
        distance_z_max = 1 - positions[i * 3 + 2] - radii[i]

        min_distance_to_domain_borders = min(
            min_distance_to_domain_borders,
            distance_x_min,
            distance_x_max,
            distance_y_min,
            distance_y_max,
            distance_z_min,
            distance_z_max,
        )

    # We want to maximize both objectives, so negate and combine them
    return -(min_distance_between_spheres + min_distance_to_domain_borders)

# Check if the correct number of command-line arguments are provided
if len(sys.argv) != 2:
    print("Usage: python script.py n")
    sys.exit(1)

# Parse the command-line argument as an integer
try:
    num_spheres = int(sys.argv[1])
except ValueError:
    print("Error: num_spheres must be an integer")
    sys.exit(1)

# Calculate the radii array
radii = np.linspace((percentage_fill_with_spheres) / (2*num_spheres) - size_interval , ( percentage_fill_with_spheres) / (2*num_spheres) + size_interval , num_spheres)

# Define the initial positions of the spheres (within the (0,1) domain)
initial_positions = np.random.rand(len(radii), 3)

# Create constraints to ensure spheres are completely inside the domain
domain_constraints = []
for i in range(len(radii)):
    domain_constraints.extend([
        {'type': 'ineq', 'fun': lambda pos, i=i: pos[i * 3] - radii[i]},
        {'type': 'ineq', 'fun': lambda pos, i=i: pos[i * 3 + 1] - radii[i]},
        {'type': 'ineq', 'fun': lambda pos, i=i: pos[i * 3 + 2] - radii[i]},
        {'type': 'ineq', 'fun': lambda pos, i=i: 1 - pos[i * 3] - radii[i]},
        {'type': 'ineq', 'fun': lambda pos, i=i: 1 - pos[i * 3 + 1] - radii[i]},
        {'type': 'ineq', 'fun': lambda pos, i=i: 1 - pos[i * 3 + 2] - radii[i]},
    ])

# Create bounds for the optimization variables (positions)
bounds = [(0, 1)] * len(radii) * 3  # Each position component (x, y, z) is in the range (0,1)

# Solve the optimization problem
result = minimize(objective_function, initial_positions.flatten(), constraints=domain_constraints, bounds=bounds)

# Extract the optimal positions
optimal_positions = result.x.reshape(-1, 3)

# Print the optimal positions and the maximum distance
print("Optimal Positions (x, y, z):")
spheres_to_plot = []
for i, pos in enumerate(optimal_positions):
    # print(f"Sphere {i + 1}: {pos}")
    print(
            f"    {{ {{ {pos[0]}, {pos[1]}, {pos[2]} }}, {radii[i]} }},"
        )
    spheres_to_plot.append((pos,radii[i]))

max_distance_between_spheres = -result.fun  # The maximum distance between spheres
print(f"Maximum Distance between Spheres: {max_distance_between_spheres:.4f}")

# max_distance_to_domain_borders = -result.fun + min_distance_between_spheres  # The maximum distance to domain borders
# print(f"Maximum Distance to Domain Borders: {max_distance_to_domain_borders:.4f}")

max_distance = -result.fun  # The maximum distance is the negation of the objective function value
print(f"Maximum Distance among Spheres: {max_distance:.4f}")


################################################### PLOT
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the spheres
for (x, y, z), radius in spheres_to_plot:
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = radius * np.outer(np.cos(u), np.sin(v)) + x
    y_sphere = radius * np.outer(np.sin(u), np.sin(v)) + y
    z_sphere = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + z

    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='b', alpha=0.6)

# Set plot limits
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)

# Add labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Save the plot as an image
plt.savefig('spheres_plot.png')  # You can change the file name and format as needed
# plt.show()
# Close the plot
plt.close()
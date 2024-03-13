import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def bezier_curve(points, num_points):
    t = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, 3))
    n = len(points) - 1
    for i in range(num_points):
        for j in range(n+1):
            curve[i] += np.array(points[j]) * (np.math.comb(n, j) * (1-t[i])**(n-j) * t[i]**j)
    return curve

def create_pyramid(base_points, apex, num_points):
    pyramid = []
    for i in range(len(base_points)):
        points = [base_points[i], base_points[(i+1)%len(base_points)], apex]
        pyramid.append(bezier_curve(points, num_points))
    return pyramid

def floating_horizon(pyramid, view_point):
    visible_faces = []
    for face in pyramid:
        normal = np.cross(face[1]-face[0], face[-1]-face[0])
        if np.dot(normal, face[0]-view_point) < 0:
            visible_faces.append(face)
    return visible_faces

# Параметри піраміди
base_points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
apex = (0.5, 0.5, 1)
num_points = 20

# Створення піраміди
pyramid = create_pyramid(base_points, apex, num_points)

# Точка спостереження
view_point = (1.5, 1.5, 1.5)

# Видалення невидимих граней
visible_pyramid = floating_horizon(pyramid, view_point)

# Відображення піраміди з видимими та невидимими гранями
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for face in pyramid:
    ax.plot_surface(face[:, 0].reshape(4, 5),
                    face[:, 1].reshape(4, 5),
                    face[:, 2].reshape(4, 5),
                    color='orange', alpha=0.3)

for face in visible_pyramid:
    ax.plot_surface(face[:, 0].reshape(4, 5),
                    face[:, 1].reshape(4, 5),
                    face[:, 2].reshape(4, 5),
                    color='blue', alpha=0.7)

# Plot the base
base_edges = [[base_points[i], base_points[(i + 1) % len(base_points)]] for i in range(len(base_points))]
for edge in base_edges:
    ax.plot3D(*zip(*edge), color="b")

# Plot the sides
for point in base_points:
    ax.plot3D(*zip(point, apex), color="r")



ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.5)
ax.set_zlim(-0.5, 1.5)
ax.set_box_aspect((1, 1, 1))

plt.tight_layout()
plt.show()
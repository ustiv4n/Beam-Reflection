import numpy as np
import math
from scipy.spatial import Delaunay
import random as rnd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ShapePrimitive import glob_surface_vectors



random_shape = np.array([[1.0, 2.0, 3.0],[-1.0, 3.0, 5.0],[0.0, 0.0, 1.0],[4.0, 2.0, 3.0],[-2.0, 1.0, 2.0],[3.0, 1.0, -1.0],
                         [0.0, 4.0, 5.0],[2.0, -2.0, 0.0],[1.0, 1.0, 0.0],[-3.0, -1.0, -4.0]])

##### CUBE ######
cube = []

for x in range(-1, 2):
        for y in range(-1, 2):
                for z in range(-1, 2):
                        cube.append([x, y, z])
##### SPHERE #######
sphere = []

radius = 1.0  # Радиус полусферы

num_points_phi = 100  # Количество шагов для φ
num_points_theta = 50  # Количество шагов для θ


for i in range(num_points_phi):
    phi = (2 * math.pi * i) / num_points_phi
    
    for j in range(num_points_theta):
        theta = (math.pi * j) / (2 * num_points_theta)
        
        x = radius * math.sin(theta) * math.cos(phi)
        y = radius * math.sin(theta) * math.sin(phi)
        z = radius * math.cos(theta)
        
        sphere.append([x, y, z])



points = glob_surface_vectors
print("Start delaunay")
tri = Delaunay(points[:,:2])
triangles = tri.simplices

# Визуализируем поверхность и треугольную триангуляцию
triangles_polygons = []
# Выводим каждый треугольный полигон
for i in range(len(triangles)):
         # Получаем индексы вершин треугольника
        vertex_indices = triangles[i]
        # Извлекаем координаты вершин треугольника
        triangles_polygons.append(points[vertex_indices]) 

print("Finished :", np.shape(triangles_polygons))
import numpy as np
import math
import random as rnd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.style as mplstyle
from sklearn.cluster import KMeans
import colorsys


def find_nearest_point(start_point, direction, points):
    min_distance = float('inf')
    nearest_point = None
    
    for point in points:
        x, y, z = point
        x0, y0, z0 = start_point
        
        diff = [x-x0, y-y0,z-z0]
        dir_proj=  np.dot(diff, direction)
        distance = math.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
        
        if distance < min_distance:
            min_distance = distance
            nearest_point = point
    
    return nearest_point








mplstyle.use('fast')

fig1, ax1 = plt.subplots()

def mergeSections():
    return
####### PIECE OF SURFACE ######

xyScale = 30
xNum = 100
yNum = xNum
xGrid = np.linspace(0 , xyScale, xNum)
yGrid = np.linspace(0 , xyScale, xNum)
peak_cov = [[2.5, 0],[0, 2.5]]
peakNum = 500

x,y = np.meshgrid(xGrid,yGrid)

grid = np.dstack((x, y))

#mean_vec = np.random.multivariate_normal([0.0,0.0], [[20, 0],[0, 20]],peakNum)
mean_vec = (xyScale - 2* 1)*np.random.random_sample((peakNum,2)) + 1


z = np.empty((xNum,yNum))

for mean_val in mean_vec:
        rv = multivariate_normal(mean_val, peak_cov)
        z += rv.pdf(grid)
z/=np.max(z) 


heat = ax1.imshow(z)
fig1.colorbar(heat)

plt.savefig('heatmap.png')
plt.close(fig1)


gridShaped = grid.reshape((xNum*yNum, 2))
zShaped = z.reshape((xNum*yNum, 1))

surface_vectors = np.concatenate((gridShaped, zShaped),axis=1)

yCell = 5
xCell = 5

glob_surface_vectors = []

for i in range(xCell):
    for j in range(yCell):
        for point in surface_vectors:
            new_point = [point[0] + i*xyScale, point[1] + j*xyScale, point[2]]
            glob_surface_vectors.append(new_point)
glob_surface_vectors = np.array(glob_surface_vectors)


print("Grid configuration",np.shape(glob_surface_vectors))
print(max(glob_surface_vectors[:,0]), max(glob_surface_vectors[:,1]))
print(min(glob_surface_vectors[:,0]), min(glob_surface_vectors[:,1]))

################## КЛАСТЕРИЗАЦИЯ ##################

fig2, ax2= plt.subplots()
ax2 = fig2.add_subplot(111, projection='3d')
num_clusters = 1000
kmeans = KMeans(n_clusters= num_clusters)
kmeans.fit(surface_vectors)

# Получение центров кластеров
cluster_centers = kmeans.cluster_centers_
labels = kmeans.predict(surface_vectors)



#for cluster_index in range(kmeans.n_clusters):
    #ax2.scatter(surface_vectors[labels == cluster_index, 0], surface_vectors[labels == cluster_index, 1],surface_vectors[labels == cluster_index, 2])

# Вывод результатов
ax2.scatter(cluster_centers[:,0], cluster_centers[ :,1],cluster_centers[ :,2])

plt.savefig('clusters.png')
plt.close(fig2)


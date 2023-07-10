import numpy as np
from math import sqrt
from scipy.spatial import Delaunay
import random as rnd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


from Polygons import triangles_polygons
from Polygons import points
from Polygons import triangles

import datetime
import multiprocessing
from functools import partial
#from ShapePrimitive import surface_vectors


def norm(vector):
        return sqrt(np.dot(vector,vector))

def scalar_prod(vector1 , vector2):
        return np.dot(vector1, vector2)

class Polygon:
    def __init__(self, point_arr , ref_index = 1.5):
        self.ref_index =  ref_index
        self.point1 = point_arr[0]
        self.point2 = point_arr[1]
        self.point3 = point_arr[2]
        self.normal_vector , self.side12, self.side23, self.side31 = self.calculate_normal()
        self.center = self.calculate_center()

    def calculate_normal(self):
        vector12 = [self.point1[0] - self.point2[0], self.point1[1] - self.point2[1], self.point1[2] - self.point2[2]]
        vector31 = [self.point3[0] - self.point1[0], self.point3[1] - self.point1[1], self.point3[2] - self.point1[2]]
        vector23 = [self.point2[0] - self.point3[0], self.point2[1] - self.point3[1], self.point2[2] - self.point3[2]]

        
        normal_vector = np.cross(vector12, vector31)

        side12 = np.cross(vector12, normal_vector)
        side23 = np.cross(vector23, normal_vector)
        side31 = np.cross(vector31, normal_vector)
        side12 = np.array(side12)/norm(np.array(side12))
        side23 = np.array(side23)/norm(np.array(side23))
        side31 = np.array(side31)/norm(np.array(side31))
        normal_vector = np.array(normal_vector) /norm(normal_vector)
        
        if normal_vector[2] < 0:
            normal_vector *= -1.0
        return normal_vector, side12, side23, side31

    def calculate_center(self):
        center_x = (self.point1[0] + self.point2[0] + self.point3[0]) / 3
        center_y = (self.point1[1] + self.point2[1] + self.point3[1]) / 3
        center_z = (self.point1[2] + self.point2[2] + self.point3[2]) / 3
        return [center_x, center_y, center_z]
    
    def isPointIntersect(self, point):
        d12 = scalar_prod(self.side12, point - self.point1)
        d23 = scalar_prod(self.side23, point - self.point2)
        d31 = scalar_prod(self.side31, point - self.point3)
        if (d12 >= 0 and d23 >= 0 and d31 >= 0 ) or (d12 <= 0 and d23 <= 0 and d31 <= 0 ):
                return True
        else:
                return False

class StraightLine:
        def __init__(self, start_point, direction_vector, polarization = 's'):
                self.polar = polarization
                self.intensity = 1
                self.start_point = start_point
                self.intersection = [start_point]
                self.direction_vector = np.array(direction_vector)/norm(direction_vector)
                a = self.direction_vector[0]
                b = self.direction_vector[1]
                c = self.direction_vector[2]
                d = -(a * self.start_point[0] + b * self.start_point[1] + c * self.start_point[2])

def intersectionPoint(line: StraightLine, polygon: Polygon):
        diff = np.array(polygon.center - line.start_point)
        d = scalar_prod(diff, polygon.normal_vector)/ scalar_prod(line.direction_vector, polygon.normal_vector)
        return np.array(line.start_point) + np.array(line.direction_vector) * d

def distance_cross(line: StraightLine, polygon: Polygon):
        diff = polygon.center - line.start_point
        return norm(np.cross(diff, line.direction_vector))

def distance_normal(line: StraightLine, polygon: Polygon):
        diff = polygon.center - line.start_point
        return np.dot(diff,line.direction_vector)

def find_polygon_intersection(line: StraightLine, polygons):
        res_polygon = polygons[0]
        intersect_point = []
        dist = 100
        for polygon in polygons:
                if scalar_prod(line.direction_vector, polygon.normal_vector) > 0 :
                        continue
                point = intersectionPoint(line, polygon)
                if polygon.isPointIntersect(point) and distance_normal(line, polygon) < dist and  distance_normal(line, polygon) > 0: 
                        dist = distance_normal(line, polygon)
                        res_polygon = polygon
                        intersect_point = point
        if len(intersect_point) == 3:
                line.intersection.append(intersect_point)
                return res_polygon
        else:
                line.intersection.append(line.start_point + line.direction_vector * 10**4)
                return []


def calc_reflr_angles(d, n, index):
       cos_alpha = -scalar_prod(d, n)
       sin_alpha = sqrt(1 - cos_alpha**2)
       sin_alpha_t = sin_alpha /index
       cos_alpha_t = sqrt(1 - sin_alpha_t**2)
       return cos_alpha, sin_alpha, cos_alpha_t, sin_alpha_t


def calc_reflect(line: StraightLine, polygon: Polygon):
        reflect_direction = line.direction_vector -  np.array(polygon.normal_vector) * 2*scalar_prod(line.direction_vector, polygon.normal_vector) 
        cos, sin, cos_t, sin_t = calc_reflr_angles(line.direction_vector, polygon.normal_vector, polygon.ref_index)
        ref = 1
        trsm = 0
        if line.polar == 's':
                ref = (cos - polygon.ref_index * cos_t)/(cos + polygon.ref_index * cos_t)
                trsm = 2* cos/(cos + polygon.ref_index * cos_t)
        if line.polar == 'p':
                ref = (polygon.ref_index *cos -  cos_t)/(polygon.ref_index *cos + cos_t)
                trsm = 2* cos/(polygon.ref_index * cos + cos_t)
        
        line.direction_vector = reflect_direction / norm(reflect_direction)
        line.start_point = line.intersection[-1]
        line.intensity *= ref**2


############ INIT ###########
def initialize_polygon(vertex):
        return Polygon(vertex)

def initialize_polygons(triangles, num_processes):
        # Создаем пул процессов
        pool = multiprocessing.Pool(processes=num_processes)
        results = []
        # Запускаем функцию инициализации объектов на каждом процессе
        #partial_calc = partial(initialize_polygon,polygons = results)
        results = pool.map(initialize_polygon, triangles)
         # Закрываем пул процессов
        pool.close()
        pool.join()
         # Возвращаем список инициализированных объектов
        return results

def calc_line_track(start_pos, light_direction, polygons):
        print("Line processing")
        line = StraightLine(start_pos, light_direction)
        touch_poly = [ ]
        while True:
                poly = find_polygon_intersection(line, polygons)
                if poly != []:
                        touch_poly.append(poly)
                        calc_reflect(line, poly)
                else:
                        break
        return line

def main():
       
        file = open("log.txt", "w+")
        file.write('Init light and poly: ' + str(datetime.datetime.now()) + '\n')
        file.close()
        print("Init light and poly")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scale = 1.5
        ax.set_xlim([-scale, scale])
        ax.set_ylim([-scale, scale])
        ax.set_zlim([-scale, scale])


        #polygons = initialize_polygons(triangles_polygons, 12)

        polygons = [Polygon(poly, 1.5) for poly in triangles_polygons]
        


        light_start_pos = []
        light_direction = np.array([0.0, 0.5, -sqrt(3)/2])
        # Задаем границы прямоугольника
        z = 5
        x_min = 60
        x_max = 90
        y_min = -z/sqrt(3) + 60
        y_max = -z/sqrt(3) + 90
        

        # Задаем количество лучей, которые нужно сгенерировать
        input_intens = 10

        # Генерируем случайные координаты точек в заданных границах
        for _ in range(input_intens):
                x = rnd.uniform(x_min, x_max)
                y = rnd.uniform(y_min, y_max)
                light_start_pos.append(np.array([x, y, z]))

        file = open("log.txt", "a+")
        file.write('Finished init light and poly: ' + str(datetime.datetime.now()) + '\n')
        file.write('Start calculation' + '\n')
        file.close()
        print("Finished init light and poly")
        print("Start calulate light reflation")


        partial_calc = partial(calc_line_track,light_direction = light_direction, polygons = polygons)
        pool = multiprocessing.Pool(processes=4)
        light_lines  = pool.map(partial_calc, light_start_pos)
        pool.close()
        pool.join()


        scatter_points = []
        intens = []
        output_intens = 0
        for line in light_lines:
                print(line.intersection)
                scatter_direction = line.intersection[-1] - [75,75,0]
                if scatter_direction[1] > 0:
                        output_intens += line.intensity
                scatter_points.append(scatter_direction/norm(scatter_direction))
                intens.append(line.intensity * 10)
        scatter_points = np.array(scatter_points)
        ax.scatter(scatter_points[:,0],scatter_points[:,1],scatter_points[:,2], c = "grey", s = intens)


        print(output_intens)
        """ ax.plot_trisurf(points[:,0], points[:,1], poinWts[:,2], 
                        triangles=triangles, 
                        cmap='viridis',
                        alpha=0.5,
                        linewidth=0.1,
                        antialiased=True) """


        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.show()
        #plt.savefig('sphere.png')
        #plt.close(fig)

        file = open("log.txt", "a+")
        file.write('Finished calculation: ' + str(datetime.datetime.now()) + '\n')
        file.write('Input light intense: ' + str(input_intens) + '\n' )
        file.write('Output light intense: ' + str(output_intens)+ '\n' )
        file.close()


if __name__ == "__main__":
       main()
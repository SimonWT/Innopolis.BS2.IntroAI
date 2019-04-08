import numpy as np
import numpy.random as rm
from heapq import nsmallest
import cv2

num_parents = 3
num_pop = 6
num_generations = 100000
color_range = 255
max_point_range = 20



def gen_poly(high):
    high = abs(high)
    x_0 = rm.randint(0, 510)
    y_0 = rm.randint(0, 510)
    x_max = x_0 + high
    y_max = y_0 + high
    if x_max > 511:
       x_max = 511
    elif x_max == x_0 or x_max < x_0 :
        x_max += x_0 + 1
    if y_max > 511:
       y_max = 511
    elif y_max == y_0 or y_max < y_0 :
        y_max += y_0 + 1

    return np.array([[[x_0, y_0],
                      [rm.randint(x_0, x_max), rm.randint(y_0, y_max )],
                      [rm.randint(x_0, x_max), rm.randint(y_0, y_max )],
                      [rm.randint(x_0, x_max), rm.randint(y_0, y_max )]]], dtype=np.int32)


def gen_color(low, high):
    return (rm.randint(low, high), rm.randint(low, high), rm.randint(low, high))


def find_fitness(goal, pop):
    # Euclidian distance bw vectors
    fit = []
    for vec in pop:
        fit.append(np.linalg.norm(goal - vec))
    return fit


def select_parents(population, fitness, num_parents):
    ind = fitness.index(min(fitness))
    parents = np.array(population[ind])

    smallest_vectors_i = list(map(fitness.index, nsmallest(num_parents, fitness)))
    smallest_vectors_i.pop(0)

    for i in smallest_vectors_i:
        parents = np.vstack((parents, population[i, :]))
    return parents


def crossover(parents, offspring_size):
    offspring = np.copy(parents)
    offspring = np.vstack((offspring, parents))

    return offspring


def mutations(population, point_range, color_range):
    mutated = np.empty((population.shape[0], 512 * 512 * 3), int)
    for vec in population:
        img = vec.reshape(dim[0], dim[1], dim[2])
        cv2.fillPoly(img, gen_poly( point_range), gen_color(0, color_range))
        mutated = np.vstack((mutated, img.reshape(dim[0] * dim[1] * dim[2])))
    return population

im = cv2.imread("klimt-the-kiss.jpg")
dim = im.shape
goal = im.reshape(dim[0] * dim[1] * dim[2])

init = np.full((512*512*3), 255)
new_population = np.array([init, init, init, init, init, init])

point_range = max_point_range
for generation in range(num_generations):
    fitness = find_fitness(goal, new_population)
    print(generation, min(fitness), point_range)
    parents = select_parents(new_population, fitness, num_parents)
    crossovered = crossover(parents, offspring_size=(num_pop, 512 * 512 * 3))
    mutated = mutations(crossovered, point_range, color_range)


    new_population = parents[0, :]
    new_population = np.vstack((new_population, parents[1:(num_pop // 2), :]))
    new_population = np.vstack((new_population, mutated[(num_pop // 2):, :]))

    result = new_population[fitness.index(min(fitness)), :]

    point_range = abs(int(max_point_range - generation*(max_point_range / num_generations)))
    if( point_range < 1 ):
        point_range = 1

    if(generation % 100 ==0 ):
        cv2.imwrite("Result.jpg", result.reshape(dim[0], dim[1], dim[2]))

cv2.imwrite("Result.jpg", result.reshape(dim[0], dim[1], dim[2]))


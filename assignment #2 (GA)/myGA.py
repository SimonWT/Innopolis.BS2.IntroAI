import numpy as np
import numpy.random as rm
from heapq import nsmallest
import cv2


def find_fitness(goal, pop):
    # Euclidian distance bw vectors
    fit = []
    for vec in pop:
        # print(np.linalg.norm(goal - vec))
        fit.append(np.linalg.norm(goal - vec))
    return fit

def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size, int)
    # The point at which crossover takes place between two parents. Usually it is at the center.
    crossover_point = np.uint8(offspring_size[1] / 2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k % parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k + 1) % parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]

    return offspring

def select_parents(population, fitness, num_parents):
    ind = fitness.index(min(fitness))
    parents = np.array(population[ind])

    smallest_vectors_i = map(fitness.index, nsmallest(num_parents, fitness))
    for i in smallest_vectors_i:
        if i != ind:
            parents = np.vstack((parents, population[i, :]))
    return parents

def mutations(population):
    # number
    for ex in population:
        for i in range(0, 512 * 512 * 3):
            ex[i] = abs(256 % (ex[i] + rm.randint(-255, 255)))


num_parents = 3
num_pop = 6
num_generations = 1000

im = cv2.imread("nav.png")

dim = im.shape
goal = im.reshape(dim[0] * dim[1] * dim[2])

im2 = cv2.imread("img.png")
init = im2.reshape(dim[0] * dim[1] * dim[2])

new_population = np.array([init, init, init, init, init, init])

for generation in range(num_generations):

    fitness = find_fitness(goal, new_population)
    parents = select_parents(new_population, fitness, num_parents)
    print(parents.shape[0])
    new_population = crossover(new_population, offspring_size=(num_pop, 512*512*3))
    mutations(new_population)

# new_population[0:3, :] = parents
# new_population[3:, :] = new_population

result = new_population[fitness.index(min(fitness)),:]
print(result)
cv2.imwrite('img_result.jpg', result.reshape(dim[0], dim[1], dim[2]))
#cv2.imshow("image", result.reshape(dim[0], dim[1], dim[2]))
#cv2.waitKey()

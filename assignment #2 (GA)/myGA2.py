import numpy as np
import numpy.random as rm
from heapq import nsmallest
import cv2

def getGradMatrix(goal, parents):
    gradMatrix = np.copy(parents)
    gradMatrix = parents - goal
    normalized = np.vectorize(normalization)
    return normalized(gradMatrix)

def find_fitness(goal, pop):
    # Euclidian distance bw vectors
    fit = []
    for vec in pop:
        fit.append(np.linalg.norm(goal - vec))
    return fit

def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size, int)
    # The point at which crossover takes place between two parents. Usually it is at the center.
    crossover_point = np.int(offspring_size[1] / 2)

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

    smallest_vectors_i = list(map(fitness.index, nsmallest(num_parents, fitness)))
    smallest_vectors_i.pop(0)
    # print(smallest_vectors_i)
    for i in smallest_vectors_i:
        parents = np.vstack((parents, population[i, :]))
    return parents

def mutations(population, num_pop, gradMatrix, step):
    negative = 0 - step
    change_matrix = np.random.randint(negative, step, (num_pop, 512 * 512 * 3))
    new_pop = population + change_matrix * gradMatrix
    mod = np.vectorize(mod256)
    return mod(new_pop)

def normalization(x):
    if(x > 0):
        return 1
    elif(x == 0):
        return 0
    else:
        return -1

def mod256(x):
    return abs(x % 256)

num_parents = 3
num_pop = 6
num_generations = 128

im = cv2.imread("nav.png")
dim = im.shape
goal = im.reshape(dim[0] * dim[1] * dim[2])

im2 = cv2.imread("img.png")
init = im2.reshape(dim[0] * dim[1] * dim[2])

new_population = np.array([init, init, init, init, init, init])

step = 128
for generation in range(num_generations):
    fitness = find_fitness(goal, new_population)
    print(generation, min(fitness), step)
    parents = select_parents(new_population, fitness, num_parents)
    crossovered = crossover(parents, offspring_size=(num_pop, 512 * 512 * 3))
    gradMatrix = getGradMatrix(goal, crossovered)
    mutated = mutations(crossovered, num_pop, gradMatrix, step)

    new_population = parents[0, :]
    new_population = np.vstack((new_population, parents[1:(num_pop // 2), :]))
    new_population = np.vstack((new_population, mutated[(num_pop // 2):, :]))
    if(step > 1):
        step -= 1




result = new_population[fitness.index(min(fitness)), :]
print(result)
cv2.imwrite('img_result4.png', result.reshape(dim[0], dim[1], dim[2]))
# cv2.imshow("image", result.reshape(dim[0], dim[1], dim[2]))
# cv2.waitKey()

import random

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)
random.seed(123)

# Definicja funkcji
def min_func(x):
    return 4 * x**4 - 6 * x**3 - 5 * x**2 + 2 * x - 4

# Przedział poszukiwań
lower_bound = -1
upper_bound = 2

# Parametry algorytmu genetycznego
population_size = 100
mutation_rate = 0.1
n_population = 60
num_generations = 100

# Inicjalizacja populacji początkowej
population = np.random.uniform(low=lower_bound, high=upper_bound, size=population_size-2)
population = np.append(population, lower_bound)
population = np.append(population, upper_bound)
#print("początkowa populacja: {}".format(population))



# Ewolucja populacji
for generation in range(num_generations):
    # Wykres funkcji
    # x = np.linspace(lower_bound, upper_bound, 100)
    # y = min_func(x)
    # plt.plot(x, y, label='f(x)')
    # plt.xlabel('x')
    # plt.ylabel('f(x)')

    # Obliczanie wartości funkcji dla każdego osobnika
    fitness_values = min_func(population)
    #print("Wartości dopasowania: {}".format(fitness_values))

    # Wybieranie najlepszego osobnika
    best_index = np.argmin(fitness_values)
    best_individual = population[best_index]
    best_fitness = fitness_values[best_index]

    # Rysowanie wykresu dla danej generacji
    # plt.scatter(population, fitness_values, label=f'Generation {generation + 1}')
    # plt.scatter(best_individual, best_fitness, color='red', label='Best Individual')
    # plt.legend()
    # plt.show()

    sum = np.sum(fitness_values)
    #print(sum)

    # Krzyżowanie i mutacja
    new_population = []
    weights = []
    for i in range(population_size):
        weights.append(fitness_values[i]/sum)

    #print(weights)

    for i in range(n_population):
        # Wybieranie rodziców
        parents = random.choices(population, weights=weights, k=2)

        # Krzyżowanie (krzyżowanie przez średnią)
        child = np.mean(parents)
        # print("numer losowania: {}".format(i+1))
        # print("rodzice: {}".format(parents))
        # print("dziecko: {}".format(child))
        # Mutacja
        if np.random.random() < mutation_rate:
            child += np.random.uniform(low=-0.1, high=0.1)
            #print("mutation {}".format(i))

        new_population.append(child)

    # Aktualizacja populacji
    # print("początkowa populacja: {}".format(population))
    # print("dzieci: {}".format(new_population))
    population = np.append(population, new_population)
    # print("nowa populacja (poczatkowa+dzieci): {}".format(population))
    population = np.unique(population)
    #print("nowa populacja (bez powtórzeń): {}".format(population))

    # Usuwanie osobników z najniższą wartością funkcji dopasowania
    fitness_values = min_func(population)
    #print("Wartości dopasowania: {}".format(fitness_values))
    idx = np.argsort(fitness_values)[:population_size]
    #print("indeksy osobników z najniższą wartością funkcji dopasowania: {}".format(idx))
    population = np.array(population[idx])
    #print("nowa populacja (bez osobników z najwyższą wartością funkcji dopasowania): {}".format(population))


fitness_values = min_func(population)

# Wybieranie najlepszego osobnika
best_index = np.argmin(fitness_values)
best_individual = population[best_index]
best_fitness = fitness_values[best_index]

print("Najlepszy osobnik: {}, wartosc: {}".format(best_individual, best_fitness))

x = np.linspace(lower_bound, upper_bound, 100)
y = min_func(x)
plt.plot(x, y, label='f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.scatter(best_individual, best_fitness, color='red', label='Best Individual')
plt.legend()
plt.show()
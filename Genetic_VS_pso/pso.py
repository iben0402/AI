import random
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
random.seed(42)


# Funkcja celu
def objective_func(x):
    return 4 * x ** 4 - 6 * x ** 3 - 5 * x ** 2 + 2 * x - 4


# Parametry algorytmu PSO
num_particles = 10
num_iterations = 10
w = 0.5  # Współczynnik wagowy
c1 = 1  # Współczynnik przyśpieszenia cząstki (dla najlepszej pamięci)
c2 = 2  # Współczynnik przyśpieszenia cząstki (dla najlepszej pamięci społecznej)
search_range = (-1, 2)  # Przedział poszukiwań

# Inicjalizacja roju cząstek
particles = np.random.uniform(search_range[0], search_range[1], (num_particles, 1))
velocities = np.zeros((num_particles, 1))
best_positions = particles.copy()
best_values = objective_func(best_positions)

# print('Początkowe pozycje cząstek: {}'.format(particles))
# print("Funkcja celu: {}".format(best_values))

# Wykres dla początkowych cząstek
plt.plot(np.linspace(search_range[0], search_range[1], 100), objective_func(np.linspace(search_range[0], search_range[1], 100)), color='blue', label='Funkcja')
plt.scatter(particles, objective_func(particles), color='red', label='Cząstki')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Początkowe cząstki roju')
plt.legend()
plt.show()


# Główna pętla algorytmu PSO
for iteration in range(num_iterations):
    # plt.figure()  # Nowe okno wykresu dla każdej iteracji
    # plt.plot(np.linspace(search_range[0], search_range[1], 100),
    #          objective_func(np.linspace(search_range[0], search_range[1], 100)), color='blue', label='Funkcja')
    # plt.scatter(particles, objective_func(particles), color='red', label='Cząstki')


    # Wylosowanie liczb losowych
    r1 = np.random.random(size=(num_particles, 1))
    r2 = np.random.random(size=(num_particles, 1))

    # print('Iteracja {}'.format(iteration + 1))
    # print("r1: {}".format(r1))
    # print("r2: {}".format(r2))
    # Aktualizacja prędkości i pozycji cząstek
    for i in range(num_particles):
        velocities[i] = w * velocities[i] + c1 * r1[i] * (best_positions[i] - particles[i]) + c2 * r2[i] * (
                    best_positions[np.argmin(best_values)] - particles[i])
        particles[i] += velocities[i]

        # Ograniczenie pozycji cząstek do przedziału poszukiwań
        particles[i] = np.clip(particles[i], search_range[0], search_range[1])

        # Aktualizacja najlepszych pozycji i wartości
        current_value = objective_func(particles[i])
        # print("Cząstka: {}, Pozycja: {}, Wartość: {}".format(i+1, particles[i], current_value))
        if current_value < best_values[i]:
            best_positions[i] = particles[i]
            best_values[i] = current_value
            # print("Zaktualizowano najlepszą pozycję cząstki")

    # print("Prędkości: {}".format(velocities))
    # print("Pozycje: {}".format(particles))
    #
    # print("Najlepsze pozycje: {}".format(best_positions))
    # print("Najlepsze wartości: {}".format(best_values))

    # plt.scatter(best_positions, best_values, color='green', label='Najlepsze pozycje')
    # plt.xlabel('x')
    # plt.ylabel('f(x)')
    # plt.title(f'Iteracja {iteration + 1}')
    # plt.legend()
    #
    # plt.show()


print("Najlepsza pozycja: {}. Wartość: {}".format(best_positions[np.argmin(best_values)], np.min(best_values)))

# Wykres dla najlepszych pozycji
plt.plot(np.linspace(search_range[0], search_range[1], 100), objective_func(np.linspace(search_range[0], search_range[1], 100)), color='blue', label='Funkcja')
plt.scatter(best_positions[np.argmin(best_values)], np.min(best_values), color='green', label='Najlepsze pozycje')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Najlepsza pozycja')
plt.legend()
plt.show()
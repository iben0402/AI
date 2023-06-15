import matplotlib.pyplot as plt
import numpy as np

def Rosenbrock(x, y):
    return (1 - x)**2 + 100*(y - x**2)**2


def plot():
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = Rosenbrock(X, Y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')

    # Ustawienie etykiet osi
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')

    plt.plot(1, 1, 0, 'ro')

    # Pokazanie wykresu
    plt.show()

plot()
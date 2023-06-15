import numpy as np
import matplotlib.pyplot as plt

def hardlim(z):
    return 1 if z > 0 else 0
class NeuralNetwork:
    def __int__(self, size):
        self.wages1 = np.array([[1, 4, -16],
                          [-1, 1, 6],
                          [-3, -5, 58],
                          [1, -4, 26],
                          [2, 1, -11]])

        self.wages2 = np.array([1, 1, 1, 1, 1, -4])
        self.activation_function = hardlim
        self.size = size


    def predict(self, point):
        z = []
        for i in range(self.size):
            z.append(self.activation_function(np.dot(point, self.wages1[i])))

        z.append(1)

        return self.activation_function(np.dot(z, self.wages2))



def generate_data(amount,min_x, max_x, min_y, max_y):
    points = []
    for i in range(amount):
        x = np.random.random()*(max_x - min_x) + min_x
        y = np.random.random()*(max_y - min_y) + min_y
        points.append([x, y, 1])

    return points


def plot_data(points, nn):

    plt.figure()

    pts = np.array([[4, 3],
                    [8, 2],
                    [11, 5],
                    [6, 8],
                    [2, 7]])



    for point in points:
        if nn.predict(point) == 1:
            plt.plot(point[0], point[1], 'bo')
        else:
            plt.plot(point[0], point[1], 'ro')

    for i in range(len(pts)):
        plt.plot(pts[i][0], pts[i][1], 'g*')

    for i in range(len(pts)):
        if i == len(pts) - 1:
            plt.plot([pts[i][0], pts[0][0]], [pts[i][1], pts[0][1]], 'g-')
        else:
            plt.plot([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]], 'g-')

    plt.axis('equal')
    plt.show()

def main():
    nn = NeuralNetwork()
    nn.__int__(5)
    points = generate_data(1000, 0, 14, 0, 10)
    plot_data(points, nn)
    # for point in points:
    #     print("P({:.2f}, {:.2f}): {}".format(point[0], point[1], nn.predict(point)))


if __name__ == "__main__":
    main()
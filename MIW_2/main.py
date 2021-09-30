from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # konfiguruje generator znaczników i mapę kolorów
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # rysuje wykres powierzchni decyzyjnej
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # rysuje wykres wszystkich próbek
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl,
                    edgecolor='black')

class Perceptron(object):

    # Konstruktor, podajemy współczynik uczenia sie oraz ilość epok
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


class Combiner():

    def __init__(self, classifier_1, classifier_2, classifier_3):
        self.classifier_1 = classifier_1
        self.classifier_2 = classifier_2
        self.classifier_3 = classifier_3

    def predict(self, x):

        return np.where(self.classifier_1.predict(x) == 1, 0, np.where(self.classifier_2.predict(x) == 1, 1, np.where(self.classifier_3.predict(x) == 1, 2, 1)))
        #dziala tylko jesli ostatni perceptron przy niespelnionym warunku przypisuje -1 do klasy 1 -> dlaczego?


def main():

    # pobiera danne do uczenia i testowania
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    # podział danych na testowe i treningowe
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    # selekcja danych z klas 0, 1 i 2

    y_train_01_set = y_train.copy()
    y_train_03_set = y_train.copy()
    y_train_02_set = y_train.copy()
    X_train_set = X_train.copy()

    y_train_01_set[(y_train == 1) | (y_train == 2)] = -1
    y_train_01_set[(y_train == 0)] = 1

    y_train_02_set[(y_train == 0) | (y_train == 2)] = -1
    y_train_02_set[(y_train == 1)] = 1

    y_train_03_set[(y_train == 1) | (y_train == 0)] = -1
    y_train_03_set[(y_train == 2)] = 1

    # w perceptronie wyjście jest albo 1 albo -1
    # perceptron klasy 0
    ppn_0 = Perceptron(eta=0.12, n_iter=300)
    ppn_0.fit(X_train_set, y_train_01_set)

    #perceptron klasy 1
    ppn_1 = Perceptron(eta=0.12, n_iter=300)
    ppn_1.fit(X_train_set, y_train_02_set)

    #perceptron klasy 2
    ppn_2 = Perceptron(eta=0.12, n_iter=300)
    ppn_2.fit(X_train_set, y_train_03_set)

    combiner = Combiner(ppn_0, ppn_1, ppn_2)


    # wyświetla wykres
    plot_decision_regions(X=X_test, y=y_test, classifier=combiner)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()



class LogisticRegressionGD(object):
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        #self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            #cost = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))))
            #self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)

class Combiner_Log():

    def __init__(self, classifier_1, classifier_2, classifier_3):
        self.classifier_1 = classifier_1
        self.classifier_2 = classifier_2
        self.classifier_3 = classifier_3

    def predict(self, x):
       return np.where(self.classifier_1.predict(x) == 1, 0, np.where(self.classifier_2.predict(x) == 1, 1, np.where(self.classifier_3.predict(x) == 1, 2, 1)))


def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    # w regresji logarytmicznej wyjście przyjmuje wartości 0 lub 1 (prawdopodobieństwa)


    y_train_01_set = y_train.copy()
    y_train_03_set = y_train.copy()
    y_train_02_set = y_train.copy()
    X_train_set = X_train.copy()

    y_train_01_set[(y_train == 1) | (y_train == 2)] = 0
    y_train_01_set[(y_train == 0)] = 1

    y_train_02_set[(y_train == 0) | (y_train == 2)] = 0
    y_train_02_set[(y_train == 1)] = 1

    y_train_03_set[(y_train == 0) | (y_train == 1)] = 0
    y_train_03_set[(y_train == 2)] = 1


    #reglog klasa 0
    lrgd_0 = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
    lrgd_0.fit(X_train_set, y_train_01_set)

    #reglog klasa 1
    lrgd_1 = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
    lrgd_1.fit(X_train_set, y_train_02_set)

    #reglog klasa 2
    lrgd_2 = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
    lrgd_2.fit(X_train_set, y_train_03_set)

    y1_predict = lrgd_0.predict(X_test)
    y2_predict = lrgd_1.predict(X_test)
    y3_predict = lrgd_2.predict(X_test)

    #accuracy_1 = accuracy_score(y1_predict, y_train_01_set)
    #accuracy_2 = accuracy_score(y2_predict, y_train_02_set)
    #accuracy_3 = accuracy_score(y3_predict, y_train_03_set)





    combiner_1 = Combiner_Log(lrgd_0, lrgd_1, lrgd_2)
    plot_decision_regions(X, y, classifier=combiner_1)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()



if __name__ == '__main__':
    main()
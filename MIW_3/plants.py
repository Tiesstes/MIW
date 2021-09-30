from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus



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
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl, edgecolor='black')

"""
##1. Rozdziel zestaw danych na podzbiory uczący i testowy,
"""

iris = datasets.load_iris()
X = iris.data[:, [1,2]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, stratify=y)

"""##2. Sprawdź działanie drzewa dla entropii i współczynnika Giniego - porównaj wyniki i uargumentuj rezultaty. 
Przykład tworzenia drzewa decyzyjnego znajduje się poniżej.
Dokumentacja klasy którą można wykorzystać do kostrukcji drzewa decyzyjnego znajduję się w linku poniżej.

"""

# drzewko z entropią - nie wiem na ile skomplikowane mają być wnioski, ale z mojego skromnego punktu widzenia
# jak sama nazwa wskazuje z pomocą przychodzi nam poziom nieuporządkowania i o ile dobrze rozumiem pomaga nam ona
# zdeterminować jak dany atrybut ma wpływ na atrybut decyzyjny
# aparycja bardzo przyjemna

tree_entropy = tree.DecisionTreeClassifier(criterion='entropy')
tree_entropy = tree_entropy.fit(X_train, y_train)
tree.plot_tree(tree_entropy)
plt.show()

plot_decision_regions(X=X_test, y=y_test, classifier=tree_entropy)
plt.title('Tree with entropy')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.legend(loc='upper left')
plt.show()


# drzewko z ginim - tutaj to się chyba nie popiszę a drzewko mnie się nie podoba za bardzo
# różnice pojawiają się z poprzednikiem na niższych poziomach.
# Pan Gini był statystykiem a za statystyką nie przepadam, ale i Włochem a to już lepiej .
# Jego indeks mierzy poziom koncentracji (braku równomierności) zmiennej losowej i z jego pomocą przydziela klasy

tree_gini = tree.DecisionTreeClassifier(criterion='gini') # Uzupełnić parametry konstruktora
tree_gini = tree_gini.fit(X_train, y_train)
tree.plot_tree(tree_gini)
plt.show()

plot_decision_regions(X=X_test, y=y_test, classifier=tree_gini)
plt.title('Tree with Gini')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.legend(loc='upper left')
plt.show()


dot_data = StringIO()
export_graphviz(tree_entropy, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('tree_e.png')
Image(graph.create_png())

dot_data = StringIO()
export_graphviz(tree_gini, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('tree_g.png')
Image(graph.create_png())


"""##3. Sprawdź działanie drzewa dla różnych głębokości drzewa - porównaj wyniki i uargumentuj rezultaty,"""

# wiele tu nie powiem. Ograniczenie głębokości drzewa uniemożliwia stworzenie odpowiedniej ilości klas.
# Jak widać na plotce klasy są mniej zróżnicowane ponieważ musiały 'pomieścić' więcej elementów

tree_gini.max_depth=2
tree_gini = tree_gini.fit(X_train, y_train)
plt.title('Tree with Gini depth 2')
tree.plot_tree(tree_gini)
plt.show()

tree_gini.max_depth = 4
tree_gini = tree_gini.fit(X_train, y_train)
plt.title('Tree with Gini depth 4')
tree.plot_tree(tree_gini)
plt.show()


"""4. Sprawdź działanie lasów losowych dla różnej liczby drzew decyzyjnych - porównaj wyniki i uargumentuj rezultaty."""

# tutaj zmiana którą dostrzega moje oko to zwyczajnie dokładność określenia klas na płaszczyźnie i widać to chyba
# na plotkach. Widzimy że zależnie od układu kwiatków kształtuje się nam granica klasy

forest_entropy_small = RandomForestClassifier(criterion='entropy', n_estimators=30)
forest_entropy_small.fit(X=X_test, y=y_test)
plot_decision_regions(X_test, y_test, classifier=forest_entropy_small, test_idx=range(105, 150))
plt.title('Gini forest 30 iterations')
plt.xlabel(r'Długość płatka irysa?')
plt.ylabel(r'Szerokość płatka irysa?')
plt.legend(loc='upper left')
plt.show()

forest_entropy_big = RandomForestClassifier(criterion='entropy', n_estimators=150)
forest_entropy_big.fit(X=X_test, y=y_test)
plot_decision_regions(X_test, y_test, classifier=forest_entropy_big, test_idx=range(105, 150))
plt.title('Gini forest 150 iterations')
plt.xlabel(r'Długość płatka irysa?')
plt.ylabel(r'Szerokość płatka irysa?')
plt.legend(loc='upper left')
plt.show()

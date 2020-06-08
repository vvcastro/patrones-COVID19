# clasificadores
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def execute_command(x_training, y_training, x_testing, y_testing, commands):
    # commands = [classifier, params]
    functions = {'knn': k_nearest_neighbours, 'bayes': naive_bayes,
                 'svm': support_vector_machine, 'nn': neural_network}

    # inicializamos el clasificador
    classifier = functions[commands[0]](x_training, y_training, commands[1])
    y_predicted = classifier.predict(x_testing)

    # estadisticas
    acc = accuracy_score(y_testing, y_predicted)
    cmatrix = confusion_matrix(y_testing, y_predicted)
    return acc, cmatrix


def k_nearest_neighbours(x_training, y_training, params):
    # parasm = {'k': n_neighbors}
    knn = KNeighborsClassifier(n_neighbors=params['k'])
    knn.fit(x_training, y_training)
    return knn


def naive_bayes(x_training, y_training, params):
    # params = {}
    gnb = GaussianNB()
    gnb.fit(x_training, y_training)
    return gnb


def support_vector_machine(x_training, y_training, params):
    # params = {'kernel': [linear, poly, rbf, sigmoid], 'gamma': [scale, auto]}
    svm = SVC(kernel=params['kernel'], gamma=params['gamma'])
    svm.fit(x_training, y_training)
    return svm


def neural_network(x_training, y_training, params):
    # params = {'solver': [lbfgs, sgd, adam], 'alpha': float, 'hlayers': (a, b), 'rstate': int,
    #           'max_iters': int}
    net = MLPClassifier(solver=params['solver'], alpha=params['alpha'],
                        hidden_layer_sizes=params['hlayers'], random_state=params['rstate'],
                        max_iter=params['max_iters'])

    net.fit(x_training, y_training)
    return net

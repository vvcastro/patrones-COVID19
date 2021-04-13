from pybalu.feature_selection import sfs
import numpy as np


def sequential_forward_selection(x_training, y_training, n_features, method):
    selected_features = sfs(x_training, y_training, n_features=n_features,
                            method=method, show=False)
    return selected_features


def sequential_backward_selection(x_training, y_training, n_features, method):
    selected_features = list(np.arange(x_training.shape[1]))

    iterations = len(selected_features) - n_features
    for i in range(iterations):
        kmin, score_min = 0, np.inf
        for k in selected_features:  # vamos a recorrer cada una de las features
            selected_without_k = selected_features.copy()
            selected_without_k.remove(k)

            # calculamos el score para del nuevo array
            this_x_training = np.hstack([x_training[:, selected_without_k]])
            this_score = score_with_fisher(this_x_training, y_training)

            if this_score < score_min:  # si es un nuevo minimo
                kmin = k  # es candidato a ser eliminado
                score_min = this_score  # seteamos un nuevo score minimo

        # al final de toda la iteración eliminamos al de menor score
        selected_features.remove(kmin)

    return np.array(selected_features)


def score_with_fisher(features, y_training):
    ''' Hace el calculo del score según las features entregadas, se ocupa Fisher para calcularlo,
        Esto está implementado de forma que classes siempre tenga la primera clase con index 0...'''
    # información de la clase de los datos
    unique, counts = np.unique(y_training, return_counts=True)
    counter_per_class = {unique[i]: counts[i] for i in range(unique.shape[0])}  # contador por clase

    # primero debemos calcular las matriz de covarianza intra e inter clase
    c_matrix = {}  # array que contiene las matrices de covarianza para cada clase
    # parametros para los siguientes calculos
    cw_matrix = np.zeros(shape=(features.shape[1], features.shape[1]))  # matriz intra clase
    cb_matrix = np.zeros(shape=(features.shape[1], features.shape[1]))  # matriz inter clase

    # esto esta implementado de forma que la primera clase siempre sea 0
    for _class in range(unique.shape[0]):  # siempre vamos a tener 2 clases
        features_class = features[(y_training == _class), :]  # elegimos todas las features de clase i
        c_matrix[_class] = np.cov(features_class, rowvar=False)  # guardamos la matriz de covarianza

        # probabilidad de la clase la calculamos como el (numero de muestras de la clase / num de clase)
        prob = (counter_per_class[_class] / unique.shape[0])

        # sumamos a la matriz de covarianza intra clase
        cw_matrix += prob * c_matrix[_class]

        # calculamos la covarianza inter clases
        zk_minus_z = (features_class.mean(0) - features.mean(0))
        cb_matrix += prob * (zk_minus_z).dot(zk_minus_z.T)

    try:
        # finalmente retornamos el score; la traza de la cw^-1 * cb
        return np.trace(np.linalg.inv(cw_matrix).dot(cb_matrix))
    except np.linalg.LinAlgError:
        # en el caso de que exista un error (pasa bastante), retornamos que no hay que considerarlo
        return - np.inf

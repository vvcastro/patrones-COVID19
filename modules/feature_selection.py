from pybalu.feature_selection import sfs
from pybalu.feature_analysis import score
import numpy as np
import tqdm


def sequential_forward_selection(x_training, y_training, params):
    # params = {'n_features': int, 'method': [fisher, sp100]}
    selected_features = sfs(x_training, y_training, n_features=params['n_features'],
                            method=params['method'], show=True)
    return selected_features


def sequential_backward_selection(x_training, y_training, params):
    # params = {'n_features': int, 'method': [fisher, sp100]}
    selected_features = list(np.arange(x_training.shape[1]))
    iterations = len(selected_features) - params['n_features']

    for i in tqdm.tqdm(range(iterations), desc='Selecting Features'):
        kmin, score_min = 0, np.inf
        for k in selected_features:  # vamos a recorrer cada una de las features
            selected_without_k = selected_features.copy()
            selected_without_k.remove(k)

            # calculamos el score para del nuevo array
            this_x_training = np.hstack([x_training[:, selected_without_k]])
            this_score = score(this_x_training, y_training, method=params['method'])

            if this_score < score_min:  # si es un nuevo minimo
                kmin = k  # es candidato a ser eliminado
                score_min = this_score  # seteamos un nuevo score minimo

        # al final de toda la iteración eliminamos al de menor score
        selected_features.remove(kmin)

    return np.array(selected_features)

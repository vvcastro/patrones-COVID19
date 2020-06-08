# La idea de este archivo es ejecutar previo a la ejecución de main.py, en este se
# extraen todas las caracteristicas de cada imagen y se guardan en un array en las carpetas de
# training y testing

from modules.feature_extraction import features_from_img
import numpy as np
import os

DATADIR = 'data'
EXTENSION = '.png'

# directorios con archivos organizados por clases
training_path = os.path.join(DATADIR, 'training')
testing_path = os.path.join(DATADIR, 'testing')

# contenedores por clases
training_features, training_labels = [], []
testing_features, testing_labels = [], []

# directorios de cada clase
cdirs = ['class_0', 'class_1', 'class_2']
flabels = None

# cargamos los archivos de training
print('Loading imgs from training...')
for cdir in sorted(cdirs):
    print('    Loading dir: ', cdir)
    files = [file for file in os.listdir(os.path.join(training_path, cdir)) if EXTENSION in file]
    for file in sorted(files):
        features_dict = features_from_img(os.path.join(training_path, cdir, file))
        training_features.append(list(features_dict.values()))
        training_labels.append(file[:4])
        if flabels is None:
            flabels = list(features_dict.keys())

# cargamos los archivos de testing
print('Loading imgs from testing...')
for cdir in sorted(cdirs):
    print('    Loading dir: ', cdir)
    files = [file for file in os.listdir(os.path.join(testing_path, cdir)) if EXTENSION in file]
    for file in sorted(files):
        features_dict = features_from_img(os.path.join(testing_path, cdir, file))
        testing_features.append(list(features_dict.values()))
        testing_labels.append(file[:4])

# guardamos los archivos
np.save(os.path.join(training_path, 'dataset_features.npy'), np.array(training_features))
np.save(os.path.join(training_path, 'dataset_labels.npy'), np.array(training_labels))
np.save(os.path.join(testing_path, 'dataset_features.npy'), np.array(testing_features))
np.save(os.path.join(testing_path, 'dataset_labels.npy'), np.array(testing_labels))
np.save(os.path.join(DATADIR, 'flabels.npy'), np.array(flabels))

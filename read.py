import pandas as pd
import numpy as np
from FFTOperation import fft_with_filter
from sklearn.neural_network import MLPClassifier

file_1 = pd.read_csv("./data/train/Stopper/_80cm_AS_00008.csv", header=None)
data_set = fft_with_filter(file_1)
X = []
for f_list in data_set:
    a = np.ravel(f_list)
    X += [np.ravel(f_list).tolist()]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(50,15,5), random_state=1)
# result = clf.predict(features)
# print(result)
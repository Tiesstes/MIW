
from google.colab import drive
drive.mount('/content/drive')

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


number_of_files = 15
for n in range(number_of_files):

  file = np.loadtxt("/content/drive/MyDrive/miw4/Dane/dane{}.txt".format(n + 1))

  X = file[:, [0]]
  y = file[:, [1]]

  print('Dataset:', str(n + 1))
  print()

  #print('X')
  #print(X, '\n')
  #print('y')
  #print(y, '\n')
  #print('\n')

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # divide whole dataset into train and test sets

  #print('X train')
  #print(X_train, '\n')
  #print('X test')
  #print(X_test, '\n')
  #print('y train')
  #print(y_train, '\n')
  #print('y test')
  #print(y_test, '\n')
  #print('\n')

  # linear model y = ax + 1
  linear_matrix = np.hstack([X_train, np.ones(X_train.shape)]) # matrix [x, 1]
  v_linear = np.linalg.pinv(linear_matrix) @ y_train # pseudo inverted matrix
  linear_model = v_linear[0] * X + v_linear[1]
  print('Linear model: \n', v_linear)

  e_linear_train = sum((y_train - (v_linear[0] * X_train + v_linear[1])) ** 2) / len(X_train) # train average of error vector for linear model
  print('Linear model (train): \n', e_linear_train)

  e_linear_test = sum((y_test - (v_linear[0] * X_test + v_linear[1])) ** 2) / len(X_test) # test average of error vector for linear model
  print('Linear model (test): \n', e_linear_test)
  print('\n')

  # other model y = x^5 - 2* x^2 + 1
  other_matrix = np.hstack([pow(X_train, 5), -2*pow(X_train, 2), np.ones(X_train.shape)]) # matrix [x, 1]
  v_other = np.linalg.pinv(other_matrix) @ y_train # pseudo inverted matrix
  other_model = v_other[0] * pow(X, 5) + (-2) * v_other[1] * pow(X, 2) + 1
  print('Other model: \n', v_other)

  e_other_train = sum((y_train - (v_other[0] * pow(X_train, 5) + (-2) * v_other[1] * pow(X_train, 2) + 1)) ** 2) / len(X_train) # train average of error vector for other model
  print('Other model (train): \n', e_other_train)

  e_other_test = sum((y_test - (v_other[0] * pow(X_test, 5) + (-2) * v_other[1] * pow(X_test, 2) + 1)) ** 2) / len(X_test) # test average of error vector for other model
  print('Other model (test): \n', e_other_test)
  print('\n')

  # draw graph
  plt.plot(X_test, y_test, 'bo') # test data
  plt.plot(X_train, y_train, 'ro') # train data
  plt.plot(X, linear_model) # for linear model
  plt.plot(X, other_model) # for other model
  plt.show()
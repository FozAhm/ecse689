from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn import svm, linear_model
import numpy as np
import csv
import sys

data_location = sys.argv[1]
row_offset = int(sys.argv[2]) - 1 # number of beginning rows to skip over
row_stop = int(sys.argv[3])
data_type = sys.argv[4]
algorithm = sys.argv[5]

print('ECSE 689 Machine Learnng')

all_x = []
all_y = []

with open(data_location) as csv_data:
    csv_reader = csv.reader(csv_data, delimiter=',')

    row_count = 0
    for row in csv_reader:
        if row_count > row_offset and row_count < row_stop:
            if data_type == 'default':
                all_x.append(list(map(int, row[1:24])))
                all_y.append(int(row[24]))
        row_count += 1

for i in range(10):
    print('Sample:', i+1,',Feautures:', all_x[i], ',Target:', all_y[i])

if algorithm == 'SVC':
    X_train, X_test, y_train, y_test = train_test_split(all_x, all_y, test_size=0.2, random_state=0)
    clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    print(clf.score(X_test, y_test))
elif algorithm == 'logistic':
    X_train, X_test, y_train, y_test = train_test_split(all_x, all_y, test_size=0.2, random_state=0)
    clf = linear_model.LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=1000).fit(X_train, y_train)
    print('Score:', clf.score(X_test, y_test))    
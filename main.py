from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import svm, linear_model
import numpy as np
import time
import csv
import sys

start_time = time.time()

data_location = sys.argv[1]
row_offset = int(sys.argv[2]) - 1 # number of beginning rows to skip over
row_stop = int(sys.argv[3])
data_type = sys.argv[4]
algorithm = sys.argv[5]

# Marketing Data Keys
binary = {
    'no': 0,
    'yes': 1,
    'unknown': 2
}
job = {
    'admin.' : 0,
    'blue-collar' : 1,
    'entrepreneur' : 2,
    'housemaid': 3,
    'management': 4,
    'retired': 5,
    'self-employed': 6,
    'services': 7,
    'student': 8,
    'technician': 9,
    'unemployed': 10,
    'unknown': 11
}
marital = {
    'divorced': 0,
    'married': 1,
    'single': 2,
    'unknown': 3
}
education = {
    'basic.4y': 0,
    'basic.6y': 1,
    'basic.9y': 2,
    'high.school': 3,
    'illiterate': 4,
    'professional.course': 5,
    'university.degree': 6,
    'unknown': 7
}
contact = {
    'cellular': 0,
    'telephone': 1
}
month = {
    'jan': 0,
    'feb': 1,
    'mar': 2,
    'apr': 3,
    'may': 4,
    'jun': 5,
    'jul': 6,
    'aug': 7,
    'sep': 8,
    'oct': 9,
    'nov': 10, 
    'dec': 11
}
day = {
    'mon': 0,
    'tue': 1,
    'wed': 2,
    'thu': 3,
    'fri': 4
}
poutcome = {
    'failure': 0,
    'nonexistent': 1,
    'success': 2
}

#Marketting to 

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
            elif data_type == 'market':
                values = [s.replace('"', '') for s in row[0].split(';')]
                
                # Data Processing
                values[0] = int(values[0])
                values[1] = job[values[1]]
                values[2] = marital[values[2]]
                values[3] = education[values[3]]
                values[4] = binary[values[4]]
                values[5] = binary[values[5]]
                values[6] = binary[values[6]]
                values[7] = contact[values[7]]
                values[8] = month[values[8]]
                values[9] = day[values[9]]
                values[10] = int(values[10])
                values[11] = int(values[11])
                values[12] = int(values[13])
                values[13] = int(values[13])
                values[14] = poutcome[values[14]]
                values[15] = float(values[15])
                values[16] = float(values[16])
                values[17] = float(values[17])
                values[18] = float(values[18])
                values[19] = float(values[19])
                values[20] = binary[values[20]]

                all_x.append(list(values[0:20]))
                all_y.append(values[20])
        row_count += 1

print('Number of Rows in Data:', row_count)
for i in range(10):
    print('Sample:', i+1,',Feautures:', all_x[i], ',Target:', all_y[i])

X_train, X_test, y_train, y_test = train_test_split(all_x, all_y, test_size=0.2, random_state=0)

if algorithm == 'SVC':
    clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    print(clf.score(X_test, y_test))
elif algorithm == 'logistic':
    clf = linear_model.LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=1000).fit(X_train, y_train)
    print('Score:', clf.score(X_test, y_test))
else:
    print('Machine Learning Algorithm Wrong')

print("--- Total Program Execution Time ---\n--- %s seconds ---" % (time.time() - start_time))
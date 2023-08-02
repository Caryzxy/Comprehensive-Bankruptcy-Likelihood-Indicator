import string
from tkinter import X
import openpyxl
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.utils import shuffle
import joblib
import random
from sklearn.metrics import accuracy_score

def get_data(file, start_pos_x:int, start_pos_y:int):
    data_list = []
    x = start_pos_x-1
    y = start_pos_y-1
    data_excel = openpyxl.load_workbook(file)
    sheet = data_excel.active
    for row in list(sheet.rows)[y:]:
        valid = True
        row_list=[]
        for cell in row[x:]:
            if type(cell.value) != float:
                valid = False
            row_list.append(cell.value)
        if valid == True:
            data_list.append(row_list)

    return data_list

def save_results(filename, data):
    wb = openpyxl.Workbook()
    ws = wb.active
    for rows in data:
        ws.append(rows)
    data_file_name = filename.split(".",1)
    result_name = data_file_name[0]+"_result.xlsx"
    wb.save(result_name)

def predict_function_test(data):
    result_list = []

    for rows in data:
        result = 0
        target = rows[0]
        for cell in rows:
            if type(cell) != str:
                result = result + cell
        result_list.append([target, result])
    
    return result_list

def make_dataset(b_data:list, n_data:list):
    for row in b_data:
        row.append(0)
    for row in n_data:
        row.append(1)
    
    full_data = b_data + n_data
  
    x_set = []
    y_set = []
    for row in full_data:
        y_set.append(row.pop())
        x_set.append(row)
    x_set, y_set = shuffle(x_set, y_set)
    return x_set, y_set

def list_of_dataset(b_data_list, n_data_list):
    x_data_list = []
    y_data_list = []
    for index in range(len(b_data_list)):
        b = b_data_list[index]
        n = n_data_list[index]
        x_set, y_set = make_dataset(b,n)
        x_data_list.append(x_set)
        y_data_list.append(y_set)

    return x_data_list, y_data_list

def make_random_forest(x_data_list, y_data_list):
    classifier = RandomForestClassifier(max_samples = 0.27,min_weight_fraction_leaf = 0.01)
    clf = classifier.fit(x_data_list,y_data_list)
    return clf

def combine_clf(clf_a, clf_b):
    clf_a.estimators_ += clf_b.estimators_
    clf_a.n_estimators = len(clf_a.estimators_)
    return clf_a

def calc_accuracy(predict_result, target_result):
    score = accuracy_score(predict_result, target_result)
    return score

def save_model(classifier,file_name):
    joblib.dump(classifier, file_name)

























































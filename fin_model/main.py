from pickle import NONE
import random
import openpyxl
import argparse
import functions
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from functools import reduce
import os



def get_args_parser():
    parser = argparse.ArgumentParser('fin model evaluation script', add_help=False)

    #Import the data from excel
    parser.add_argument('--datafile_bankrupt_path', type = str)
    parser.add_argument('--datafile_nonbankrupt_path', type = str)

    #Label the start index of data of the excel, b_x and b_y is for bankrupt
    #while n_x and n_y is for non-bankrupt
    parser.add_argument('--b_x', type = int)
    parser.add_argument('--b_y', type = int)
    parser.add_argument('--n_x', type = int)
    parser.add_argument('--n_y', type = int)

    #Import the test data
    parser.add_argument('--test_data', type = str)
    parser.add_argument('--t_x', type = int)
    parser.add_argument('--t_y', type = int)

    #We can save the model we trained
    parser.add_argument('--save_model_name', type = str, default= None)
    return parser



def main(args):

    #get data from the path given
    bankrupt_data = []
    for file_name in os.listdir(args.datafile_bankrupt_path):
        bankrupt_data.append(args.datafile_bankrupt_path + "/" + file_name)

    nonbankrupt_data = []
    for file_name in os.listdir(args.datafile_nonbankrupt_path):
        nonbankrupt_data.append(args.datafile_nonbankrupt_path + "/" + file_name)


    #build 
    b_data_list = []
    for sheet in bankrupt_data:
        data = functions.get_data(sheet, args.b_x, args.b_y)
        b_data_list.append(data)
    n_data_list = []
    for sheet in nonbankrupt_data:
        data = functions.get_data(sheet, args.n_x, args.n_y)
        n_data_list.append(data)

    full_data, target = functions.list_of_dataset(b_data_list, n_data_list)

    #If you are using "Bankrupt_Companies_with_Basic_Ratios_in_the_US.xlsx"
    #as test data, please comment out the following code.
    #This is to reshape our train data so that the feature size match
    #the one in test data

    #BTW, for some reason "Non_Bankrupt Companies_Last_Quarter.xlsx"
    #has lost one feature compare to the other data
    #for row in full_data:
    #   row.pop(4)
    
    #make a classifier
    list_of_classifier =[]
    for index in range(len(full_data)):
        random_forest = functions.make_random_forest(full_data[index], target[index])
        list_of_classifier.append(random_forest)
    
    classifier_combined = reduce(functions.combine_clf, list_of_classifier)

    #build test dataset and do the test
    #0 means bankruot and 1 means non-bankrupt
    test_data_list = functions.get_data(args.test_data, args.t_x, args.t_y)
    result = classifier_combined.predict(test_data_list)
    print(result)
    
    #calculate the accuracy
    #if you are using "Non_Bankrupt Companies_Last_Quarter.xlsx"
    #target_result = [1] *len(result)
    
    #if you are using "Bankrupt_Companies_with_Basic_Ratios_in the_US.xlsx"
    target_result = [0] *len(result)

    accuracy = functions.calc_accuracy(result, target_result)
    print(accuracy)
    
    #If we choose to save model
    if args.save_model_name != None:
        functions.save_model(classifier_combined, args.save_model_name)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('fin model evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

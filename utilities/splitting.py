import sklearn.model_selection
import pandas as pd
from random import randint

def split_data(master_file, suffix, return_data=False, save_data = True):

    """Splits specified master csv into train, val, and test sets. Takes master file and output file suffix as inputs"""
    
    master = pd.read_csv(master_file)
    
    patient_ids = master[['PatientID', 'Label']].drop_duplicates().reset_index()['PatientID']
    patient_labels = master[['PatientID', 'Label']].drop_duplicates().reset_index()['Label']
    
    Train_IDs_Strat, Test_IDs_Strat, Train_Labels_Strat, Test_Labels_Strat = sklearn.model_selection.train_test_split(patient_ids, patient_labels, test_size = 0.25, random_state=randint(0,100), stratify=patient_labels)
    # split the training set again to get a validation set
    Train_IDs_Strat, Val_IDs_Strat, Train_Labels_Strat, Val_Labels_Strat = sklearn.model_selection.train_test_split(Train_IDs_Strat, Train_Labels_Strat, test_size = 0.2, random_state=randint(0,100), stratify=Train_Labels_Strat)

    Train_DCMs_Strat = master[master['PatientID'].isin(Train_IDs_Strat)]
    Val_DCMs_Strat = master[master['PatientID'].isin(Val_IDs_Strat)]
    Test_DCMs_Strat = master[master['PatientID'].isin(Test_IDs_Strat)]

    Train_file = 'Train_' + suffix + '.csv'
    Val_file = 'Val_' + suffix + '.csv'
    Test_file = 'Test_' + suffix + '.csv'
    
    if save_data:
        Train_DCMs_Strat.to_csv(Train_file, index=False)
        Val_DCMs_Strat.to_csv(Val_file, index=False)
        Test_DCMs_Strat.to_csv(Test_file, index=False)
    
    if return_data:
        return Train_DCMs_Strat.reset_index(drop=True), Val_DCMs_Strat.reset_index(drop=True), Test_DCMs_Strat.reset_index(drop=True)


def split_data_2(master_file, suffix, return_data=False, save_data = True):

    """Splits specified master csv into train, val, and test sets. Takes master file and output file suffix as inputs"""
    
    master = master_file.copy()
    
    patient_ids = master[['PatientID', 'Label']].drop_duplicates().reset_index()['PatientID']
    patient_labels = master[['PatientID', 'Label']].drop_duplicates().reset_index()['Label']
    
    Train_IDs_Strat, Test_IDs_Strat, Train_Labels_Strat, Test_Labels_Strat = sklearn.model_selection.train_test_split(patient_ids, patient_labels, test_size = 0.25, random_state=randint(0,100), stratify=patient_labels)
    # split the training set again to get a validation set
    Train_IDs_Strat, Val_IDs_Strat, Train_Labels_Strat, Val_Labels_Strat = sklearn.model_selection.train_test_split(Train_IDs_Strat, Train_Labels_Strat, test_size = 0.2, random_state=randint(0,100), stratify=Train_Labels_Strat)

    Train_DCMs_Strat = master[master['PatientID'].isin(Train_IDs_Strat)]
    Val_DCMs_Strat = master[master['PatientID'].isin(Val_IDs_Strat)]
    Test_DCMs_Strat = master[master['PatientID'].isin(Test_IDs_Strat)]

    Train_file = 'Train_' + suffix + '.csv'
    Val_file = 'Val_' + suffix + '.csv'
    Test_file = 'Test_' + suffix + '.csv'
    
    if save_data:
        Train_DCMs_Strat.to_csv(Train_file, index=False)
        Val_DCMs_Strat.to_csv(Val_file, index=False)
        Test_DCMs_Strat.to_csv(Test_file, index=False)
    
    if return_data:
        return Train_DCMs_Strat.reset_index(drop=True), Val_DCMs_Strat.reset_index(drop=True), Test_DCMs_Strat.reset_index(drop=True)
    
def holdout_data(master_file, suffix, brand, return_data=False, save_data = True):
    
    """Splits specified master csv into train, val, and test sets while holding out the specified brand from the train and val sets. Takes master file, output file suffix, and holdout brand as inputs"""
    
    master = pd.read_csv(master_file)

    # remove all images from the holdout brand
    holdouts = master[master['Brand']==brand]
    removed = pd.concat([master, holdouts]).drop_duplicates(keep=False)

    patient_ids = removed[['PatientID', 'Label', 'Brand']].drop_duplicates().reset_index()['PatientID']
    patient_labels = removed[['PatientID', 'Label', 'Brand']].drop_duplicates().reset_index()['Label']

    # split the patients in the removed set into train and test sets (80:20)
    Train_IDs_Strat, Test_IDs_Strat, Train_Labels_Strat, Test_Labels_Strat = sklearn.model_selection.train_test_split(patient_ids, patient_labels, test_size = 0.25, random_state=randint(0,100), stratify=patient_labels)
    # split the training set again to get a validation set
    Train_IDs_Strat, Val_IDs_Strat, Train_Labels_Strat, Val_Labels_Strat = sklearn.model_selection.train_test_split(Train_IDs_Strat, Train_Labels_Strat, test_size = 0.2, random_state=randint(0,100))

    Train_DCMs_Strat = removed[removed['PatientID'].isin(Train_IDs_Strat)]
    Val_DCMs_Strat = removed[removed['PatientID'].isin(Val_IDs_Strat)]
    Test_DCMs_Strat = removed[removed['PatientID'].isin(Test_IDs_Strat)]

    # add the held-out images to the test set
    Test_DCMs_Strat = pd.concat([Test_DCMs_Strat, holdouts])

    Train_file = 'Train_' + suffix + '.csv'
    Val_file = 'Val_' + suffix + '.csv'
    Test_file = 'Test_' + suffix + '.csv'
    if save_data:
        Train_DCMs_Strat.to_csv(Train_file, index=False)
        Val_DCMs_Strat.to_csv(Val_file, index=False)
        Test_DCMs_Strat.to_csv(Test_file, index=False)
        
    if return_data:
        return Train_DCMs_Strat.reset_index(drop=True), Val_DCMs_Strat.reset_index(drop=True), Test_DCMs_Strat.reset_index(drop=True)

def holdout_data_2(master_file, suffix, brand, return_data=False, save_data = True):
    
    """Splits specified master csv into train, val, and test sets while holding out the specified brand from the train and val sets. Takes master file, output file suffix, and holdout brand as inputs"""
    
    master = master_file.copy()

    # remove all images from the holdout brand
    holdouts = master[master['Brand']==brand]
    removed = pd.concat([master, holdouts]).drop_duplicates(keep=False)

    patient_ids = removed[['PatientID', 'Label', 'Brand']].drop_duplicates().reset_index()['PatientID']
    patient_labels = removed[['PatientID', 'Label', 'Brand']].drop_duplicates().reset_index()['Label']

    # split the patients in the removed set into train and test sets (80:20)
    Train_IDs_Strat, Test_IDs_Strat, Train_Labels_Strat, Test_Labels_Strat = sklearn.model_selection.train_test_split(patient_ids, patient_labels, test_size = 0.25, random_state=randint(0,100), stratify=patient_labels)
    # split the training set again to get a validation set
    Train_IDs_Strat, Val_IDs_Strat, Train_Labels_Strat, Val_Labels_Strat = sklearn.model_selection.train_test_split(Train_IDs_Strat, Train_Labels_Strat, test_size = 0.2, random_state=randint(0,100))

    Train_DCMs_Strat = removed[removed['PatientID'].isin(Train_IDs_Strat)]
    Val_DCMs_Strat = removed[removed['PatientID'].isin(Val_IDs_Strat)]
    Test_DCMs_Strat = removed[removed['PatientID'].isin(Test_IDs_Strat)]

    # add the held-out images to the test set
    Test_DCMs_Strat = pd.concat([Test_DCMs_Strat, holdouts])

    Train_file = 'Train_' + suffix + '.csv'
    Val_file = 'Val_' + suffix + '.csv'
    Test_file = 'Test_' + suffix + '.csv'
    if save_data:
        Train_DCMs_Strat.to_csv(Train_file, index=False)
        Val_DCMs_Strat.to_csv(Val_file, index=False)
        Test_DCMs_Strat.to_csv(Test_file, index=False)
        
    if return_data:
        return Train_DCMs_Strat.reset_index(drop=True), Val_DCMs_Strat.reset_index(drop=True), Test_DCMs_Strat.reset_index(drop=True), holdouts.shape[0]
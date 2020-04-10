def metrics_function(y_predicted, y_probs, y_true):
    '''
    This function takes an input of predictions and true values and returns weighted precision, recall, f1 scores,
    and AUC scores. 
    Inputs:
        y_predicted: NumPy array of shape (n_samples,) which contains predictions of categories
        y_probs: NumPy array of shape (n_samples, n_classes) which contains probabilities for each class
        y_true: NumPy array of shape (n_samples,) which contains actual labels for samples
    Outputs:
        f1_score: Weighted F1-score
        precision: Weighted Precision score
        recall: Weighted recall score
        auc: Weighted AUC score calculated using One-Versus-Rest Approach
        confusion_matrix: Confusion Matrix
    '''
    import sklearn.metrics
    from sklearn.preprocessing import OneHotEncoder
    import numpy as np
    
    params = {
        'y_true': y_true,
        'y_pred': y_predicted,
        'average': 'weighted'
    }
    f1_score = sklearn.metrics.f1_score(**params)
    precision = sklearn.metrics.precision_score(**params)
    recall = sklearn.metrics.recall_score(**params)
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true = y_true, y_pred = y_predicted)
    
    encoder = OneHotEncoder()
    y_encoded = encoder.fit_transform(np.array(y_true).reshape(-1,1))
    auc = sklearn.metrics.roc_auc_score(y_true = y_encoded.toarray(), y_score = y_probs, average='weighted', multi_class = 'ovr')
    
    return f1_score, precision, recall, auc, confusion_matrix


def compile_numeric_results(f1_scores, precisions, recalls, aucs):
    '''
    Function to compile numeric results coming out of n iterations. Expects lists for each input
    Assumes each list is a collection of scores for each iteration i.e. [10, 20, 30] means score
    for iteration 1 is 10, iteration 2 is 20 and iteration 3 is 30
    Inputs:
        f1_scores: List of F1-Scores
        precisions: List of precisions
        recalls: List of recalls
        aucs: List of aucs
    Output:
        Dataframe with means and 95% confidence intervals
    '''
    import pandas as pd
    import scipy.stats as st
    import numpy as np
    
    compiled = [f1_scores, precisions, recalls, aucs]
    compiled = pd.DataFrame(np.array(compiled).T, columns = ['F1', 'Precision', 'Recall', 'AUC'])
    
    # Calculating compiled results
    conf_int = compiled.apply(lambda x: st.t.interval(0.95, len(x)-1, loc=np.mean(x), scale=st.sem(x)))
    means = compiled.apply(lambda x: np.mean(x))
    support = compiled.apply(lambda x: len(x))
    
    # Putting them together
    final = pd.concat([means,conf_int, support], axis = 1)
    final.columns = ["Mean", "Confidence Intervals", "Support"]
    
    return final

def compile_phase(data, phase):
    '''
    Compiles data from curves (i.e. loss or accuracy) for n epochs. Assumes that the data provided is a list
    of dictionaries i.e. [{train:[1,2,3], 'val': [4,5,6]}, ...] where each index in the list is
    a separate iteration of the model. Assumes only train and val within the dictionary
    Inputs:
        data: A list of dictionaries as described above
        phase: a string (either train or val) specificying which data you're interested in removing
    Outputs:
        One dataframe with mean results for each epoch, confidence intervals and support
    '''
    import pandas as pd
    import numpy as np
    import scipy.stats as st
    
    final_ = pd.DataFrame()
    for i in range(len(data)):
        final_ = pd.concat([final_, pd.DataFrame(data[i][phase])], axis = 1)
        
    # Calculating results    
    conf_int = final_.apply(lambda x: st.t.interval(0.95, len(x)-1, loc=np.mean(x), scale=st.sem(x)), axis=1)
    means = final_.mean(axis=1)
    support = final_.apply(lambda x: len(x), axis=1)

    # Putting them together
    finished = pd.concat([means,conf_int, support], axis = 1)
    finished.columns = ["Mean", "Confidence Intervals", "Support"]
    
    return finished 

def compile_curves(curve_data):
    '''
    Compiles data from curves (i.e. loss or accuracy) for n epochs. Assumes that the data provided is a list
    of dictionaries i.e. [{train:[1,2,3], 'val': [4,5,6]}, ...] where each index in the list is
    a separate iteration of the model. Assumes only train and val within the dictionary
    Inputs:
        curve_data: A list of dictionaries as described above
    Outputs:
        Two dataframes (one for train and one for val) with mean results for each epoch, confidence intervals and support
    '''  
    import pandas as pd

    train_compiled = compile_phase(curve_data, 'train')
    val_compiled = compile_phase(curve_data, 'val')
    concatenated = pd.concat([train_compiled, val_compiled], axis=1)
    concatenated.columns = [f'{x}_{y}' for x in ['train', 'val'] for y in train_compiled.columns]
    
    return concatenated
import numpy as np
import pandas as pd
import math
import random

def retrieve_bootstrap_data(df,current_entity,df_metrics_locations):
    """
    Retrieve the number of TP, FP and FN present in each file.
    
    Parameters : 
    df (dataframe) : Dataframe containing the annotated words, their occurrences, places and related entity.
    current_entity (str) : current working entity selected.  
    df_metrics_locations (dataframe) : contains the locations of each categorized word, and whether they are true positive, false positive or false negative.
    
    Return :
    (dict) : dictionnary containing each file and their associated TP, FP and FN. 
    """
    bootstrap_data = {}
    # Retrieve of TP, TP(corr) and FP
    for file in df_metrics_locations['file'].unique():
        TP = df_metrics_locations[(df_metrics_locations['file'] == file) & ((df_metrics_locations['result'] == "TP"))].shape[0]
        TPcorr = df_metrics_locations[(df_metrics_locations['file'] == file) & ((df_metrics_locations['result'] == "TP(corr)"))].shape[0]
        FP = df_metrics_locations[(df_metrics_locations['file'] == file) & ((df_metrics_locations['result'] == "FP"))].shape[0]
        bootstrap_data[file]={'TP': (TP+TPcorr),'FP': FP,'FN': 0}
        
    #Retrieve of FN
    for index, row in df.iterrows():
        if "category?" in row['category']:
            for place in row['places']:
                file = place[0]+".txt"
                if file in bootstrap_data:
                    bootstrap_data[file]['FN'] += 1
                else : 
                    bootstrap_data[file]={'TP': 0,'FP': 0,'FN': 1}
                    
    return bootstrap_data


def calculate_metrics(TP, FP, FN):
    """
    Calculate the precision, recall and f1 from the metrics.
    
    Parameters : 
    TP (int) : number of true positive (TP).
    FP (int) : number of false positive (FP).
    FN (int) : number of false negative (FN).
    
    Returns :
    (list) : list containg the calculated precision, recall and f1 values.

    """
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 *precision*recall)/(precision+recall) if (precision+recall) else 0
    return precision, recall, f1


def estimate_confidence_intervals_bootstrap(df,current_entity,df_metrics_locations,draw_number=1000, alpha=5.0):
    """
    Calculate the confidence intervals for each entity's metrics (precision and recall) by bootstrapping on the files. 
    
    Parameters : 
    df (dataframe) : Dataframe containing the annotated words, their occurrences, places and related entity.
    current_entity (str) : current working entity selected.  
    df_metrics_locations (dataframe) : contains the locations of each categorized word, and whether they are true positive, false positive or false negative.
    draw_number (int) : number of times that we draw files. 
    alpha (int) : represents the percentage of the distribution that falls outside the confidence interval, with a default value of 5.0
    
    Return :
    bootstrap_results (dict) : dictionnary of the confidence intervals of the metrics (precision and recall) calculated by bootstrap 
    """ 
    
    bootstrap_data = retrieve_bootstrap_data(df,current_entity,df_metrics_locations)
    files = list(bootstrap_data.keys())
    precisions = []
    recalls = []
    f1s = []
    
    for _ in range(draw_number):
        sample_files = random.choices(files, k=len(files))
        total_TP = 0
        total_FP = 0
        total_FN = 0
        
        for file in sample_files:
            total_TP += bootstrap_data[file]['TP']
            total_FP += bootstrap_data[file]['FP']
            total_FN += bootstrap_data[file]['FN']
        
        precision, recall, f1 = calculate_metrics(total_TP, total_FP, total_FN)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    lower_p = alpha / 2
    upper_p = 100 - (alpha / 2)
    
    precision_conf_interval = (np.percentile(precisions, lower_p), np.percentile(precisions, upper_p))
    precision_mean = np.mean(precisions)
    precision_median = np.median(precisions)
    recall_conf_interval = (np.percentile(recalls, lower_p), np.percentile(recalls, upper_p))
    recall_mean = np.mean(recalls)
    recall_median = np.median(recalls)
    f1_conf_interval = (np.percentile(f1s, lower_p), np.percentile(f1s, upper_p))
    f1_mean = np.mean(f1s)
    f1_median = np.median(f1s)
    
    bootstrap_results = {
        "precision" : {"precision_conf_interval": precision_conf_interval,
                      "mean" : precision_mean,
                      "median" : precision_median},
        "recall" : {"recall_conf_interval": recall_conf_interval,
                      "mean" : recall_mean,
                      "median" : recall_median},
        "f1" : {"f1_conf_interval": f1_conf_interval,
                      "mean" : f1_mean,
                      "median" : f1_median}
    }
    
    return bootstrap_results
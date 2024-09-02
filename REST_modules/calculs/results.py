import pandas as pd
import math
import re
import random
import pandas as pd

def initiate_df_results(df,homogeneity_score,ent_cat):
    """
    Create the dataframe containing the results table of the categorization of each entity.
    
    Parameters : 
    df (dataframe) : Dataframe containing the annotated words, their occurrences, places and related entity.
    homogeneity_score (dict) : dictionnary containing the entities and their associated homogeneity score.
    ent_cat (dict) : List of all the entities paired with their categories. 
    
    Return :
    (dataframe) : contains the summary of each entity's results (homogeneity,precision,recall).
    """
    df_values = []
    for entity in ent_cat:
        fn_rows = df[df['entity']== entity]
        fn_value = fn_rows['occurrences'].sum()
        df_values.append([entity,homogeneity_score[entity],0,0,fn_value,0,0,0,0,0,0])
    df_results = pd.DataFrame(df_values,columns= ["entity","homogeneity","TP","FP","FN","precision","precision_conf_inter_down","precision_conf_inter_up","recall","recall_conf_inter_down","recall_conf_inter_up"])
    return df_results


def update_df_results(df_results,df,entity,homogeneity_score,df_metrics,bootstrap_results):
    """
    Update and return the summary of each entity's results (homogeneity,precision,recall).
    
    Parameters : 
    df_results (dataframe) : contains the summary of each entity's results (homogeneity,precision,recall).
    df (dataframe) : Dataframe containing the annotated words, their occurrences, places and related entity.
    entity (str) : current working entity selected.     
    homogeneity_score (dict) : dictionnary containing the entities and their associated homogeneity score.
    df_metrics (dataframe) : dataframe containing the precision and recall score in each category.  
    bootstrap_results (dict) : dictionnary of the confidence intervals of the metrics (precision and recall) calculated by bootstrap
    
    Return :
    (dataframe) : contains the summary of each entity's results (homogeneity,precision,recall).

    """
    TP = df_metrics['TP'].sum()
    if 'TP(corr)' in df_metrics.columns :
        TP += df_metrics['TP(corr)'].sum() 
    FP = df_metrics['FP'].sum()
    FN = df.loc[df['category'].str.contains(re.escape("category?")) & df['entity'].str.contains(entity), 'occurrences'].sum()
    precision = 0
    recall = 0   
    homogeneity = homogeneity_score[entity]
    if TP+FP !=0 : precision = round(TP/(TP+FP),2)
    if TP+FN !=0 :recall = round(TP/(TP+FN),2)
    
    precision_inter1=round(bootstrap_results['precision']['precision_conf_interval'][0],2)
    precision_inter2=round(bootstrap_results['precision']['precision_conf_interval'][1],2)
    recall_inter1=round(bootstrap_results['recall']['recall_conf_interval'][0],2)
    recall_inter2=round(bootstrap_results['recall']['recall_conf_interval'][1],2)
    col = ["TP","FP","FN","precision","precision_conf_inter_down","precision_conf_inter_up","recall","recall_conf_inter_down","recall_conf_inter_up"]
    
    df_results.loc[df_results['entity']==entity,col]= [TP,FP,FN,precision,precision_inter1,precision_inter2,recall,recall_inter1,recall_inter2]
    
    return df_results

def create_categories_infos(df,ent_cat,current_entity):
    """
    Calculate the number of categorized annotations (total and unique).
    
    Parameters : 
    ent_cat (dict) : Sictionnay of all the entities paired with their categories.
    df (dataframe) : Dataframe containing the annotated words, their occurrences, places and related entity.
    current_entity (string) : String of the current working entity.
    
    Return :
    (dataframe) : Dataframe containing the number of categorized annotations. 
    """
    infos=[]
    for cat in ent_cat[current_entity]:
        tot_occurrences= df.loc[df['category'].str.contains(re.escape(cat)) & df['entity'].str.contains(current_entity), 'occurrences'].sum()
        nbr_occurrences= df.loc[df['category'].str.contains(re.escape(cat)) & df['entity'].str.contains(current_entity), 'occurrences'].shape[0]
        infos.append([cat.strip("[]"),tot_occurrences,nbr_occurrences])
    df_cat_infos = pd.DataFrame(infos,columns=["category", "total_annotations_number","total_different_annotations_number"])
    return df_cat_infos


import json
import os
import pandas as pd

def save_progress(path, ent_cat, list_isNotFP, list_isNotFN, ban_words_entities, df_results):
    """
    Save the current progress of the user on a json file called "REST_progress.json" in the corpus directory.
    
    Parameters : 
    path (string) : string containing the dataset path. 
    ent_cat (dict) : List of all the entities paired with their categories.
    list_isNotFP (list) : list of the annotations considered as FP.
    list_isNotFN (list) : list of the annotations considered as FN.
    ban_words_entities (dict) : dictionnary of each entity and their related banwords used in the categorization process. 
    df_results (dataframe) : contains the summary of each entity's results (homogeneity,precision,recall).
    
    """    
    json_file_path = os.path.join(path, 'REST_progress.json')
    
    if os.path.isfile(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    data['ent_cat'] = ent_cat
    data['list_isNotFP'] = list_isNotFP
    data['list_isNotFN'] = list_isNotFN
    data['ban_words_entities'] = ban_words_entities
    data['df_results'] = df_results.values.tolist()
    data['df_columns'] = df_results.columns.tolist()

    with open(json_file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def load_progress(path):
    """
    Load the progress of the user from a json file called "REST_progress.json" in the corpus directory..
    
    Parameters : 
    path (string) : string containing the dataset path. 
    
    Return :
    progress_ent_cat (dict) : List of all the entities paired with their categories.
    progress_isNotFP : list of the annotations considered as FP.
    progress_isNotFN : list of the annotations considered as FN.
    progress_ban_words_entities : dictionnary of each entity and their related banwords used in the categorization process.
    progress_df_results : contains the summary of each entity's results (homogeneity,precision,recall).
    """
    json_file_path = os.path.join(path, 'REST_progress.json')
    progress_ent_cat = None
    progress_isNotFP = None
    progress_isNotFN = None
    progress_ban_words_entities = None
    progress_df_results = None

    if os.path.isfile(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
                progress_ent_cat = data.get('ent_cat')
                progress_isNotFP = data.get('list_isNotFP')
                progress_isNotFN = data.get('list_isNotFN')
                progress_ban_words_entities = data.get('ban_words_entities')
                df_results_list = data.get('df_results')
                df_columns_list = data.get('df_columns')
                if df_results_list and df_columns_list:
                    progress_df_results = pd.DataFrame(df_results_list, columns=df_columns_list)
            except (json.JSONDecodeError, FileNotFoundError):
                pass

    return progress_ent_cat, progress_isNotFP, progress_isNotFN, progress_ban_words_entities, progress_df_results

import pandas as pd
import ipywidgets as widgets
from .calculs import *
from .extraction.normalisation import getEnt

def calculate_categorization(df,ent_cat,current_entity,other_categories,list_spacing_regex):
    """
    For each entered terms in the entity's category, this fonction creates a regex that will match with the annotations.
    If an annotation match with a category, it will return the updated df. Also, 
    
    Parameters : 
    df (dataframe) : Dataframe containing the annotated words, their occurrences, places and related entity.
    current_entity (string) : String of the current working entity.
    other_categories (list) : List containing the other possible category that annotations could belong to. 
    list_spacing_regex (list) : List of all the discontinued regex that will appear in the recommandations section

    Return :
    df (dataframe) : Return a dataframe containing the annotations belonging to the selected category. 
    other_categories (list) : Return a list containing the other possible category that annotations could belong to. 
    list_spacing_regex (list) : Resturn the list of all the discontinued regex that will appear in the recommandations section. 
    """    
    list_spacing_regex=[]
    other_categories=[]
    df['category']= "category?" #reset to "category?" to reset estimation
    ent_cat[current_entity] = [element for element in ent_cat[current_entity] if element != '[]'] #Erase all '[]' categories
    for ent in ent_cat:
        for cat in ent_cat[ent]:
            if cat != "category?":
                pattern, list_spacing_pattern = generate_regex(eval(cat))    
                if list_spacing_pattern and ent == current_entity:
                    list_spacing_regex.extend(list_spacing_pattern)
                mask = (df['entity'] == ent) & df['text'].apply(lambda text: bool(re.search(pattern, text)))
                df.loc[mask, 'category'] = df.loc[mask, 'category'].apply(lambda x: x + 'AND' + cat)
                    
    #Calculation of "other_categories"
    for index,row in df.iterrows():
        categories = row['category'].split("AND")
        for i, category in enumerate(categories):
            if i > 1:
                for place in row['places']:
                    other_categories.append([row['entity'],category.strip(),row['text'],place[0]+".txt",[place[1],place[2]]])
        if len(categories) > 1: 
            df.at[index, 'category'] = categories[1].strip() 
            
    return df,other_categories,list_spacing_regex

def create_ban_words_tfidf(ent_cat) :
    ban_words_tfidf= {}
    for ent in getEnt(ent_cat):
        ban_words_tfidf[ent]=[]
    return ban_words_tfidf

def remove_empty_categories(dict_ent_cat,current_entity):
    dict_ent_cat[current_entity] = [category for category in dict_ent_cat[current_entity] if category != "[]"]
    return dict_ent_cat

def modify_list_isNotFP(list_isNot,new_value):
    list_isNot_temp = []  
    v0 = ', '.join(f"'{element}'" for element in new_value)
    for values in list_isNot:
        if values[8] in new_value:
            list_isNot_temp.append([v0,values[1],values[2],values[3],values[4],values[5],values[6],values[7],values[8]])
    return list_isNot_temp
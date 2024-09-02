import os
import re
import pandas as pd
from ipydatagrid import DataGrid, TextRenderer, BarRenderer, Expr, VegaExpr,CellRenderer
from bqplot import LinearScale, ColorScale, OrdinalColorScale, OrdinalScale
from unidecode import unidecode
from .regex import *
import numpy as np

def calculate_location_metrics(current_entity,ent_cat,path,df,other_categories,list_isNotFP,list_isNotFN,ban_words_entities):
    """
    Retrieves and sorts words from annotated texts into matching categories, then compares whether these words match any of the existing annotations. 
    
    Parameters : 
    current_entity (string) : the current selected entity.
    ent_cat (dict) : List of all the entities paired with their categories.
    path (string) : path of the current working directory.
    df (dataframe) : Dataframe containing the annotated words, their occurrences, places and related entity.
    other_categories (list) : contains all the annotations of the current entity that could belong to another category. 
    
    Return :
    (dict) : dictionnary containing the locations of each categorized word, and whether they are true positive, false positive or false negative. 
    """
    
    location_metrics={}
    location_metrics[current_entity] = []
    locations = {}
    locations_found_currentfile = []
    
    txt_files = []
    for file_name in os.listdir(path):
        if file_name.endswith(".txt"):
            txt_files.append(file_name)
            locations[file_name]=[]
    
    # 1 -  storage of places of each annotations categorised in "locations"
    ent_cat2 = [cat for cat in ent_cat[current_entity] if cat != "category?"]
    for cat in ent_cat2:
        cat_striped=cat.strip("[]")
        #cat sans parenthèse
        filtre = (df['entity'] == current_entity) & (df['category'] == cat)
        for index, row in df[filtre].iterrows():
            for place in row['places']:
                name_document = place[0]+".txt"
                place_document = [place[1],place[2],row['text']]
                if name_document in locations:
                    locations[name_document].append(place_document)
    
        # 2 - Verification by category : TP,FP and FN
        pattern, list_spacing_pattern = generate_regex(eval(cat))
        for file_name in txt_files:
            locations_found_currentfile=[]
            with open(os.path.join(path, file_name), 'r', newline='',encoding='utf-8') as file:
                text = file.read().lower()
                #text_decode = unidecode(text)
                for match in re.finditer(pattern, text):
                    place_start = match.start()
                    place_end = match.end()
                    motif = text[place_start:match.end()]
                    long_motif = "..."+text[max(place_start-70,0):min(place_end+70,len(text))]+"..."
                    FP = True   
                    for location in locations[file_name]: 
                        if location[0]<=place_start and place_end <= location[1]: #place found (TP)
                            if any(banword in text[location[0]:location[1]] for banword in ban_words_entities[current_entity] if banword!="None"):
                                continue
                            annotation=location[2]
                            locations[file_name].remove(location)
                            locations_found_currentfile.append(location)
                            location_metrics[current_entity].append([cat_striped,"TP",False,False,long_motif,file_name,[place_start,place_end],annotation,motif])
                            FP = False 
                            break
                
                    if FP : #verification in other categories, if exist we add in 'FP'
                        df_other_categories = pd.DataFrame(other_categories,columns=["entity", "category","text","file", "places"])
                        filtre2 = (df_other_categories['entity'] == current_entity) & (df_other_categories['category'] == cat) & (df_other_categories['file'] == file_name)
                        add_to_FP = True   
                        if any(banword in long_motif for banword in ban_words_entities[current_entity] if banword!="None"):
                            add_to_FP = False
                            continue
                        for location in locations_found_currentfile : #verification in already exist to current category
                            if location[0]<=place_start and place_end <= location[1]:
                                add_to_FP = False
                        
                        for index, row in df_other_categories[filtre2].iterrows(): #verification in other categories
                            if row['places'][0]<=place_start and place_end <= row['places'][1]:
                                add_to_FP = False
                                break
                           
                        if add_to_FP : #location not found (FP)
                            is_notFP = False
                            for v in list_isNotFP : 
                                if [cat_striped,"TP(corr)",True,False,long_motif,file_name,[place_start,place_end],'no annotation',motif] == v :
                                    is_notFP = True
                                    break
                            if is_notFP :  
                                location_metrics[current_entity].append([cat_striped,"TP(corr)",True,False,long_motif,file_name,[place_start,place_end],'no annotation',motif]) 
                            else : 
                                location_metrics[current_entity].append([cat_striped,"FP",False,False,long_motif,file_name,[place_start,place_end],'no annotation',motif]) 
                        
            if len(locations[file_name])!=0: #locations left (FN)                
                with open(os.path.join(path, file_name), 'r', newline='',encoding='utf-8') as file:
                    text = file.read().lower()
                    text = unidecode(text)
                    for place in locations[file_name]:
                        long_motif = "..."+text[max(place[0]-70,0):min(place[1]+70,len(text))]+"..."
                        is_notFN = False
                        for v in list_isNotFN :
                            if [cat_striped,"Discarded",False,True,long_motif,file_name,[place[0],place[1]],place[2],'no motif'] == v :
                                is_notFN = True
                                break
                        if is_notFN :
                            location_metrics[current_entity].append([cat_striped,"Discarded",False,True,long_motif,file_name,[place[0],place[1]],place[2],'no motif']) 
                        else :         
                            location_metrics[current_entity].append([cat_striped,"FN",False,False,long_motif,file_name,[place[0],place[1]],place[2],'no motif']) 
                    locations[file_name]=[] 
                
    return location_metrics


def calculate_df_metrics(df_location_metrics):
    """
    Calculate the precision and recall score in each category. 
    
    Parameters : 
    df_location_metrics (dataframe) : contains the locations of each categorized word, and whether they are true positive, false positive or false negative.
    
    Return :
    df_metrics (dataframe) : dataframe containing the precision and recall score in each category.  
    """
    
    metrics = []
    for cat in df_location_metrics['category'].unique():
        TP = df_location_metrics[(df_location_metrics['category'] == cat) & (df_location_metrics['result'] == "TP")].shape[0]
        TPcorr = df_location_metrics[(df_location_metrics['category'] == cat) & (df_location_metrics['result'] == "TP(corr)")].shape[0]
        FP = df_location_metrics[(df_location_metrics['category'] == cat) & (df_location_metrics['result'] == "FP")].shape[0]
        FN = df_location_metrics[(df_location_metrics['category'] == cat) & (df_location_metrics['result'] == "FN")].shape[0]
        precision = 0
        if TP+FP !=0:
            precision = round((TP+TPcorr)/(TP+TPcorr+FP),2)
        recall = 0 
        if TP+FN !=0:
            recall= round((TP+TPcorr)/(TP+TPcorr+FN),2) 
        metrics.append([cat,TPcorr,TP,FP,FN,precision,recall])
        
    df_metrics = pd.DataFrame(metrics,columns=["category","TP(corr)","TP","FP","FN", "precision","recall"])
    df_metrics["precision"] = df_metrics["precision"].round(2)
    df_metrics["recall"] = df_metrics["recall"].round(2)
    if df_metrics["TP(corr)"].sum() == 0 :
        df_metrics.drop(columns=["TP(corr)"], inplace=True)
    return df_metrics


def text_color(cell):
    if cell.value =="FN":
        return "#ff1414" #red
    else : 
        return "#ffffff" 
        
def background_color(cell):

    if cell.value in ("TP" ,"FN") :
        return "#006400" #green
    elif cell.value in  ("TP(corr)"):
        return "#228B22" #lightgreen
    elif cell.value in  ("FP") :
        return "#500000" #red
    elif cell.value in  ("Discarded") :
        return "#999999" #grey
    else : 
        return None

def create_grid_metrics_locations(df_metrics_locations,current_entity):
    """
    Create a grid with renderers from the location metrics dictionnary. 
    
    Parameters (dict) : dictionnary containing the locations of each categorized word, and whether they are true positive, false positive or false negative.
    current_entity (string) : the current selected entity.
    
    Return :
    (datagrid) : datagrid containing the locations of each categorized word, and whether they are true positive, false positive or false negative 
    """
    
    dg_location_metrics = DataGrid(df_metrics_locations[["category", "result","text","file", "places"]],column_widths={"category":200,"result":80,"text":650,"file":100,"places":80},layout={"height":"350px","width":"1200px"},base_row_size=25,selection_mode='row')  
    
    renderers = {
        "result": TextRenderer(
            text_color=Expr(text_color),
            background_color=Expr(background_color), 
            horizontal_alignment="center"),
        "text": TextRenderer(
            horizontal_alignment="left")
    }
    dg_location_metrics.renderers = renderers
    return dg_location_metrics

def background_color_tp(cell):
    #post
    if cell.column == 4 and cell.value > cell.metadata.data["('raw highlights', 'TP')"]: #TP
        return "green"
    elif cell.column == 5 and cell.value < cell.metadata.data["('raw highlights', 'FP')"]: #FP
        return "green"

    #pre
    elif cell.column == 0 and cell.value == cell.metadata.data["('corrected highlights', 'TP(corr)')"]:#TP
        return "orange"
    elif cell.column == 1 and cell.value == cell.metadata.data["('corrected highlights', 'FP(corr)')"]: #FP
        return "orange"
     
    else :
        return None 
    


def generate_dg_metrics_results(df_metrics):
    """
    Create a datagrid from the dataframe "df_metrics".
    
    Parameters : 
    df_metrics (dataframe) : dataframe containing the precision and recall score of each category.  
    
    Return :
    (datagrid) : datagrid containing the precision and recall score of each category.
    """
    upper_index = ['raw highlights'] * 4 + ['corrected highlights'] * 4
    lower_index = ['TP', 'FP', 'FN', 'precision'] + ['TP(corr)', 'FP(corr)', 'FN(corr)', 'precision']
    
    df_metrics_results = pd.DataFrame(index=df_metrics['category'], 
                                      columns=pd.MultiIndex.from_arrays([upper_index, lower_index]))
    
    TPcorr = [0]*len(df_metrics)
    if 'TP(corr)' in df_metrics.columns:
        TPcorr=df_metrics['TP(corr)'].tolist()
    
    pre_FP = np.add(df_metrics['FP'].tolist(), TPcorr).tolist()
    precision = [round(tp / (tp + fp),2)  for tp, fp in zip(df_metrics['TP'].tolist(), pre_FP)] 
    #pre
    df_metrics_results['raw highlights','TP'] = df_metrics['TP'].tolist()
    df_metrics_results['raw highlights', 'FP'] = pre_FP
    df_metrics_results['raw highlights', 'FN'] = df_metrics['FN'].tolist()
    df_metrics_results['raw highlights', 'precision'] = precision
    #post
    df_metrics_results['corrected highlights','TP(corr)'] = np.add(df_metrics['TP'].tolist(), TPcorr).tolist()
    df_metrics_results['corrected highlights', 'FP(corr)'] = df_metrics['FP'].tolist()
    df_metrics_results['corrected highlights', 'FN(corr)'] = df_metrics['FN'].tolist()
    df_metrics_results['corrected highlights', 'precision'] = df_metrics['precision'].tolist()
    
    height = len(df_metrics_results)*25+48
    dg_metrics_results = DataGrid(df_metrics_results,
                                 base_column_size=78,
                                 column_widths={('category', ''): 300,
                                                ('raw highlights','TP'):75,
                                                ('corrected highlights','TP(corr)'):75,
                                                ('raw highlights','FP'):75,
                                                ('corrected highlights','FP(corr)'):75,
                                                ('raw highlights','FN'):75,
                                                ('corrected highlights','FN(corr)'):75,
                                               },
                                 layout={'width':'912px',"height":f"{height}px"},
                                 base_row_size=25,
                                 default_renderer=TextRenderer(horizontal_alignment='center'))
    
    renderers = {
        "('corrected highlights', 'TP(corr)')": TextRenderer(background_color=Expr(background_color_tp), bold=True,horizontal_alignment='center'),
        "('corrected highlights', 'FP(corr)')": TextRenderer(background_color=Expr(background_color_tp), bold=True,horizontal_alignment='center'),
        "('corrected highlights', 'FN(corr)')": TextRenderer(bold=True,horizontal_alignment='center'),
        "('corrected highlights', 'precision')": TextRenderer(bold=True,horizontal_alignment='center'),
        "('raw highlights', 'TP')": TextRenderer(background_color=Expr(background_color_tp), bold=True,horizontal_alignment='center'),
        "('raw highlights', 'FP')": TextRenderer(background_color=Expr(background_color_tp), bold=True,horizontal_alignment='center'),
        "('raw highlights', 'FN')": TextRenderer(bold=True,horizontal_alignment='center'),
        "('raw highlights', 'precision')": TextRenderer(bold=True,horizontal_alignment='center'),
    }    
    
    dg_metrics_results.renderers = renderers
    
    return dg_metrics_results

def compare_common_string(motif, text):
    """
    Check if a motif or a part of it is present in a text. 
    
    Parameters : 
    motif (string) : motif to check.  
    text (string) : text where the motif is checked. 
    
    Return :
    (string) : return the part of the text matching with the motif.
    """
    if len(motif)>len(text):
        temp_motif = motif
        motif = text
        text = temp_motif
    pattern = re.compile(re.escape(motif))
    match = pattern.search(text)
    if match:
        return match.group()
    else:
        for length in range(len(motif), 0, -1):
            for start in range(len(motif) - length + 1):
                sub_pattern = re.escape(motif[start:start+length])
                match = re.search(sub_pattern, text)
                if match:
                    return match.group()
    return ""


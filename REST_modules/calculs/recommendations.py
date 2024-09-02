import re
import os
import matplotlib.pyplot as plt
import random
import seaborn as sns
import pandas as pd
import numpy as np
import ipywidgets as widgets

def check_spacing_regex_locations(spacing_regex,path,len_constant):
    """
    Apply the dashed (spacing) regex on the .txt files of the corpus. For each match, the function save the filename, 
    positions and distance between the two words in the spacing regex. 
    
    Parameters : 
    spacing_regex (string) : string containing a dashed (spacing) regex of interest. 
    path (string) : string containing the dataset path.
    len_constant (int) : constant part of the dashed (spacing) regex, used to calculated the correct distence between the two words in the spacing regex
    
    Return :
    (list) : list of all the match between a spacing regex and the corpus. 

    """
    spacing_regex_results = []
    txt_files = []
    regex= r"\b" + "(" + spacing_regex + ")" + r"\b"
    constant_dist=len_constant
    
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            txt_files.append(filename)
    
    for filename in txt_files:
        with open(os.path.join(path, filename), 'r', newline='',encoding='utf-8') as file:
            text = file.read().lower()
            for match in re.finditer(regex, text):
                place_start = match.start()
                place_end = match.end()
                spacing_regex_results.append([regex, filename, [place_start,place_end],"?",((place_end-place_start)-constant_dist)])
        
    return spacing_regex_results

def compare_spacing_regex_locations(spacing_regex_results,current_entity,df):
    """
    Compare all the match of a spacing regex from the text with the annotations (highlights) made by the expert.
    If it corresponds to an existing highlight, the match is considered as a "TP", else as a "FP".
    
    Parameters : 
    spacing_regex_results (list) : list of all the match between a spacing regex and the corpus. 
    current_entity (str) : current working entity selected. 
    df (dataframe) : Dataframe containing the annotated words, their occurrences, places and related entity.
    
    Return :
    (list) : spacing_regex_results list updated with the value
    """
    updated_spacing_regex_results=[]
    
    for regex,filename,[place_start,place_end],value,distance in spacing_regex_results:
        add_to_TP = False
        for index,row in df[df['entity']==current_entity].iterrows():
            for place in row['places']:
                if (place[0]+".txt"==filename and place[1]<=place_start and place_end<=place[2] and re.finditer(regex,row['text'])):
                    add_to_TP = True
                    break
                    
        if add_to_TP : updated_spacing_regex_results.append([regex, filename, [place_start,place_end],"TP",distance])
        else : updated_spacing_regex_results.append([regex, filename, [place_start,place_end],"FP",distance])
        
    return updated_spacing_regex_results

def create_fig_recommandation(spacing_regex_values, title):
    """
    Create the recommandation figure displaying each dashed (spacing) regex match with the text.
    
    Parameters : 
    spacing_regex_values (list) : list containing distance and value for each spacing regex match.
    title (string) : title of the figure. 
    
    Return :
    matplotlib.pyplot: The matplotlib figure object with the scatter plot.
    
    """
    df = pd.DataFrame(spacing_regex_values, columns=['Type', 'Distance'])

    plt.figure(figsize=(6, 1.2))  

    sns.scatterplot(data=df, x='Distance', y=np.random.uniform(low=-0.28, high=0.28, size=len(df)),
                hue='Type', palette={'FP': 'red', 'TP': 'blue'}, marker='o', s=50, legend=False)

    for index, row in df.iterrows():
        if row['Type'] == 'TP':
            color = 'blue'
        elif row['Type'] == 'FP':
            color = 'red'
        plt.text(row['Distance'] + 1, np.random.uniform(low=-0.15, high=0.15), row['Type'], fontsize=7, ha='left', va='center', color=color)

    plt.yticks([], [])  
    plt.ylim(-0.3, 0.3)  

    plt.title("Estimation of optimized spacing for : " + title, fontsize=10)
    plt.xlabel('')
    plt.grid(True)
    plt.tight_layout()  

    return plt

def create_accordion_recommendations(list_spacing_regex,path,current_entity,df):
    """
    Calculate and return an output displaying the recommandation distance figure for the dashed (spacing regex).
    
    Parameters : 
    list_spacing_regex (list) : list containg all the dashed (spacing) regex of the current entity.
    path (string) : string containing the dataset path. 
    current_entity (str) : current working entity selected.    
    df (dataframe) : Dataframe containing the annotated words, their occurrences, places and related entity.
    
    Return :
    widgets.Output(): output that displays the recommandation distance figure for the dashed (spacing regex).

    """
    if list_spacing_regex:
        output_recommendations = widgets.Output()
        with output_recommendations:
            for index,spacing_regex in enumerate(list_spacing_regex) :
                spacing_regex_results = check_spacing_regex_locations(spacing_regex[0],path,spacing_regex[1])
                spacing_regex_results_compared = compare_spacing_regex_locations(spacing_regex_results,current_entity,df)
                df_spacing_regex_results = pd.DataFrame(spacing_regex_results_compared, columns=["regex", "filename", "location", "value", "distance"])
                spacing_regex_values = df_spacing_regex_results[['value', 'distance']].values.tolist()
                fig = create_fig_recommandation(spacing_regex_values,str(spacing_regex[2]))
                fig.show()
                
        return widgets.Accordion([output_recommendations],titles=('Recommendations for optimal spacing expressions','other'))
    else : 
        return widgets.Output()
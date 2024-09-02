import os
import re
import pandas as pd
from ipydatagrid import DataGrid, TextRenderer, Expr

def get_annotations(docs):
    """
    Retrieve the annotations from the different files.
    
    Parameters : 
    docs (generator) : contains all the informations extracted from the .ann files. 
    
    Return :
    (list) : contains the informations related to each annotation (text, place, document,entity). 
    """
    annotations=[]
    for doc in docs : 
        for ent in doc['entities']:
            annotations.append((doc['num_ann']+".txt",ent['label'],ent['text'],ent['fragments'][0]['begin'],ent['fragments'][0]['end']))
    return annotations

def get_matches(path,pattern):
    """
    Check the occurrences of a pattern in all the .txt files from the given path.
    
    Parameters : 
    path (string) : path of the current working directory.
    pattern (string) : word(s) that are searched in the texts.
    
    Return :
    (list) : contains the sentences in which the pattern is present, as well as the filename and the place in the text. 
    """
    matches=[]
    for file_name in os.listdir(path):
        if file_name.endswith(".txt"):
            with open(os.path.join(path, file_name), 'r', newline='',encoding='utf-8') as file:
                text = file.read().lower()
                for match in re.finditer(pattern, text):
                    place_start = match.start()
                    place_end = match.end()
                    sentence = "..."+text[match.start()-70:match.end()+60]+"..."
                    matches.append((pattern,sentence,file_name,place_start,place_end))
    return matches

def cut_sentence(word,sentence):
    """
    Split the sentence in 3 parts, before and after the word of interest.
    
    Parameters : 
    word (string) : word where the sentence needs to be split.
    sentence (string) : sentence to split in 3 parts.
    
    Return :
    (strings) : the text before the word of interest, the word of interest, and the text after the word of interest. 
    """
    cut=sentence.lower().split(word)
    words_before = cut[0][-70:].rjust(70)
    if len(cut[0]) > 70:
        words_before = "..."+words_before
    try : 
        words_after = cut[1].strip()
    except IndexError:
        words_after=""
    return words_before,word,words_after

def background_color_concordancer(cell):
    if cell.value != "Not annotated":
        return "#006400"
    else:
        return "#db8519"

def calculate_concordancer(pattern,current_entity,path,docs):
    """
    Creates a concordancer (datagrid) where the occurrences of a word in the text files are searched, and check if these occurrences corresponds to existing annotations.
    
    Parameters : 
    pattern (string) : word(s) that are searched in the texts.
    current_entity (string) : the current selected entity.
    path (string) : path of the current working directory.
    docs (generator) : contains all the informations extracted from the .ann files. 
    
    Return :
    (datagrid) : datagrid containing the result of the concordancer.
    """
        
    matches=get_matches(path,pattern)
    annotations=get_annotations(docs)

    res_concordancier=[]
    for match in matches : 
        res=cut_sentence(match[0],match[1])
        in_no_annotation = True
        for annotation in annotations : 
            if match[2]==annotation[0]:
                if annotation[3]<=match[3] and match[4]<=annotation[4]: 
                    if annotation[1].lower()==current_entity: 
                        res_concordancier.append((res[0],res[1],res[2],current_entity))
                    else : 
                        res_concordancier.append((res[0],res[1],res[2],annotation[1]))
                    in_no_annotation = False
                    break
        if in_no_annotation:
            res_concordancier.append((res[0],res[1],res[2],"Not annotated"))
    
    dg_res_concordancer = DataGrid(pd.DataFrame(res_concordancier,columns=['words before','word','words after','entity']),
                                  layout={"height":"350px","width":"1165px"},base_row_size=25,
                                  column_widths={"words before":440,"word":110,"words after":380,"entity":150})
    renderers = {
        "words before": TextRenderer(horizontal_alignment="right"),
        "word": TextRenderer(horizontal_alignment="center"),
        "entity": TextRenderer(background_color=Expr(background_color_concordancer))
    }
    
    dg_res_concordancer.renderers= renderers
    
    return dg_res_concordancer
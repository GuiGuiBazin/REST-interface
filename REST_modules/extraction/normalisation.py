import re
import json
import copy
import glob
import pandas as pd
from nltk.corpus import stopwords
import spacy
from nltk.stem.snowball import SnowballStemmer
from googletrans import Translator
from Levenshtein import distance as lev
import os

stemmer = SnowballStemmer("french")
translator = Translator()
sp=spacy.load("fr_core_news_sm")


def getEnt(ent_cat):
    """
    Return the list of the entities in ent_cat.
    
    Parameters : 
    ent_cat (dict) : dictionnary of the entities and their related categories. 
    
    Return :
    (list) : list of the entities in ent_cat. 
    """    
    x = []
    for ent in list(ent_cat.keys()):
        x.append(ent)
    return x

def getCat(entity,ent_cat):
    """
    Return the categories of the entered entity.
    
    Parameters : 
    ent_cat (dict) : dictionnary of the entities and their related categories. 
    entity (string) : current working entity
    
    Return :
    (list) : list of the entities in ent_cat. 
    """ 
    x = []
    for cat in ent_cat[entity]:
        x.append(cat)
    return x



def contain_digit(chain):
    """
    Return true if the chain in input contains a digit.
    
    Parameters : 
    chain (str) : Current selected entity in the UI. 
    
    Return :
    (bool) : Return true if the chain contains a digit. 
    """
    return bool(re.search(r'\d', chain))

def print_docs(docs):
    """
    Print all the informations retrieved from the the .ann files, using the "load_from_brat" function.
    
    Parameters : 
    docs (generator) : contains all the informations extracted from the .ann files. 
    """
    for doc in docs:
        print(doc.keys())
        for ent in doc['entities']:
            print(ent)

#Main function, that calls others to extract the informations form the docs
def extract_annotations(docs,need_translation) :
    """
    Extract all the desired information from the .ann files (annotations, occurrences, places, text).
    
    Parameters : 
    docs (generator) : contains all the informations extracted from the .ann files. 
    need_translation (boolean) : specify if the translation to a destination language is necessary.
    
    Return :
    (dict) : Return a dictionnary containing the annotations, and their associated caracteristics (occurrences, places, text). 
    """
    list_stem = []
    list_text = {}
    annotations = {}
    
    # 1) Extraction from docs to a dict, and calculation of occurrences
    for doc in docs: 
        for ent in doc['entities']:
            entity = ent['label'].lower()
            text = ent['text'].lower()
            place = [doc['num_ann'],ent['fragments'][0]['begin'],ent['fragments'][0]['end']]
            if entity not in annotations :
                annotations[entity]={}
                annotations[entity]['category?']={}
            if text not in annotations[entity]['category?']:
                annotations[entity]['category?'][text]= {"occurrences":1,"stems" :list_stem,"places" : [place]}
            else : 
                annotations[entity]['category?'][text]['occurrences']+=1
                annotations[entity]['category?'][text]['places'].append(place)
    
    # 2) Translation to french (if necessary)
    if need_translation and not load_cantemist :
        temp ={}
        for entity in list(annotations.keys()) :
            entity_fr=translator.translate(entity,src='es', dest='fr').text
            temp[entity_fr] = {}
            temp[entity_fr]['category?'] = {}
            for text in annotations[entity]['category?'] :
                text_fr = translator.translate(text,src='es', dest='fr').text
                if text_fr in temp[entity_fr]['category?'] :
                    temp[entity_fr]['category?'][text_fr]['occurrences'] += annotations[entity]['category?'][text]['occurrences']
                    for place in annotations[entity]['category?'][text]['places']:
                        temp[entity_fr]['category?'][text_fr]['places'].append(place)
                else : temp[entity_fr]['category?'][text_fr]=annotations[entity]['category?'][text]
        annotations = temp
    return annotations     

#Stemming of all the texts in the 'annotations' dict
def stemming(annotations) :
    """
    Calculate the stems for all the annotations, and store them in a dictionnary.
    
    Parameters : 
    annotations (dict) : contains the annotations, and their associated caracteristics (occurrences, places, text). 
    
    Return :
    (dict) : Return a dictionnary containing the annotations, and their associated caracteristics (occurrences, places, text, stems). 
    """
    current_annotations = copy.deepcopy(annotations)
    stop_words = set(stopwords.words('french'))
    stop_words.update([",","(","d'",")","l'","ni","n'","a"] ) #custom stop words
    for entity in list(current_annotations.keys()) :
        for text in current_annotations[entity]['category?']:
            stems=[]
            sentence = sp(text)
            for word in sentence :
                stem = stemmer.stem(word.text)
                if stem not in stop_words :
                    stems.append(stem)
            current_annotations[entity]['category?'][text]['stems']= stems             
    return current_annotations  

#Merge all the texts, depending on the Levenshtein distance
def Levenshtein(annotations1,dist): 
    """
    Merge the annotations if their levenshtein distance is lower than the choosen distance in input.
    
    Parameters : 
    annotations1 (dict) : contains the annotations, and their associated caracteristics (occurrences, places, text, stems). 
    dist (int) : threshold distance where annotations with Levenshtein score that are below will merge. 
    
    Return :
    tempo_annotations (dict) : Return a dictionnary containing the annotations and their associated caracteristics, where similar annotations calculated with levenshtein are merged. 
    levenshtein_results (dict) : Return a dictionnary containing the annotations merged with levenshtein's score. 
    """
    current_annotations = copy.deepcopy(annotations1)
    tempo_annotations = {}
    levenshtein_results = {}
    for entity in list(current_annotations.keys()) :
        tempo_annotations[entity]= {}
        tempo_annotations[entity]['category?']= {}
        for text in current_annotations[entity]['category?']:
            ajout = True
            stems = " ".join(current_annotations[entity]['category?'][text]['stems']) 
            for tempo_text in tempo_annotations[entity]['category?'] :
                tempo_stems = " ".join(tempo_annotations[entity]['category?'][tempo_text]['stems'])
                dist_leven=lev(stems,tempo_stems)           
                if (dist_leven<dist and not contain_digit(stems) and not contain_digit(tempo_stems) and len(tempo_stems)>4) or (dist_leven==0):
                    tempo_annotations[entity]['category?'][tempo_text]['occurrences']+= current_annotations[entity]['category?'][text]['occurrences']
                    for place in current_annotations[entity]['category?'][text]['places']:
                        tempo_annotations[entity]['category?'][tempo_text]['places'].append(place)
                    ajout =False
                    
                    #MAJ leven
                    if dist_leven>0 and (tempo_stems not in levenshtein_results) :
                        levenshtein_results[tempo_stems] = []
                        levenshtein_results[tempo_stems].append(stems)
                    elif dist_leven>0 and (stems not in levenshtein_results[tempo_stems]):
                        levenshtein_results[tempo_stems].append(stems)
                    break             
            if ajout :
                tempo_annotations[entity]['category?'][text] = current_annotations[entity]['category?'][text] 
    return tempo_annotations,levenshtein_results
    
def createData(annotations2):
    """
    Transform the 'annotations' dictionnary into a list, and create de ent_cat dictionnary.
    
    Parameters : 
    annotations2 (dict) : contains the annotations, and their associated caracteristics (occurrences, places, text, stems). 
    
    Return :
    (list) : Return a dictionnary containing the annotations and their associated caracteristics, where similar annotations calculated with levenshtein are merged. 
    ent_cat (dict) : Contains the different entities and their ongoing created categories. 
    """
    data = []
    ban_words_entities = {}
    ent_cat = {}
    for entity,categories in annotations2.items():
        ent_cat[entity]= []
        ban_words_entities[entity]=["None"]
        for category, texts in categories.items():
            ent_cat[entity].append(category)
            for text,infos in texts.items():
                occurrences = infos['occurrences']
                stems = infos['stems']
                places = infos['places']
                data.append((entity,category,text,occurrences,stems,places))
    return sorted(data, key=lambda x: x[3], reverse=True), ent_cat  ,ban_words_entities
    
def get_all_sentences(path):
    """
    Retrieve all the sentences from the .text files.
    
    Parameters : 
    path (str) : contains the path of the .txt files. 
    
    Return :
    (dataframe) : contains all the sentences, and their associated place in the .txt files. 
    """
    all_sentences = []
    txt_files = glob.glob(path+"/*.txt")

    for file_path in txt_files: 
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            text = file.read()
            start = 0
            segments = text.split('.')
            for segment in segments:
                end = start + len(segment) + 1  # +1 to include the period
                phrase = segment.split('\n')
                all_sentences.append((phrase, file_path, (start,end)))
                start = end
            
    return pd.DataFrame(all_sentences,columns=['text','file','place'])    

def print_annotations_caracteristics(dict_annotations):
    """
    Print and summarize the information associated for each different annotation.
    
    Parameters : 
    dict_annotations (dict) : contains the annotations, and their associated caracteristics (occurrences, places, text, stems).  
    """
    
    entities = list(dict_annotations.keys())
    print("List of the different entities :", entities,"\n")
    
    for entity in entities :
        n = 0
        m = 0
        categories = list(dict_annotations[entity].keys())
        print("Entity information : ",entity)
        for category in categories :   
            for text in dict_annotations[entity][category] :
                n+= dict_annotations[entity][category][text]['occurrences']
                for place in dict_annotations[entity][category][text]['places']:
                    m+=1
            
            print("   Category :",category  )
            print("   Total number of annotations :",n," (","place (verification) :",m,")")
            print("   Total number of different annotations :",len(dict_annotations[entity]['category?']))
            print("\n")
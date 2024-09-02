import pandas as pd
import math
import re
import nltk
from nltk.corpus import stopwords
import numpy as np

def calculate_tfidf(ent_cat,df):
    """
    Calculate the TF-IDF score for each word in each entity.
    
    Parameters : 
    ent_cat (dict) : List of all the entities paired with their categories.
    df (dataframe) : Dataframe containing the annotated words, their occurrences, places and related entity.
    
    Return :
    (dataframe) : Dataframe containing the calculated tfidf and the associated word, occurrences and related entity. 
    """
    tf_results = {}
    stop_words = set(stopwords.words('french'))
    stop_words.update([",","(","d'",")","l'"," ",""] )
    
    for ent in ent_cat:
        tf_results[ent]={}
        for index,row in df.loc[df['entity']==ent].iterrows():
            text = row['text']
            occurrences = row['occurrences']
            for word in text.split(" "):
                word = word.replace(",", "").replace(":", "").replace("(", "").replace(")", "").replace(".", "")
                if word not in stop_words :
                    if word not in tf_results[ent]: 
                        tf_results[ent][word]=occurrences
                    else : 
                        tf_results[ent][word]+=occurrences
                
    df_tf_results = []              
    for entity,words in tf_results.items():
        for word,occurrence in words.items():
            df_tf_results.append((entity,word,occurrence,0))
    df_tf_results = pd.DataFrame(df_tf_results, columns=['entity','word','occurrences','tfidf'])
        
    #Calculation of b,c,d and tfidf
    nbr_entities = len(ent_cat) #c
    for ent in ent_cat :
        current_df = df_tf_results.loc[df_tf_results['entity']==ent]
        tot_nbr_words_in_entity= current_df['occurrences'].sum() #b
        for index,row in current_df.iterrows():
            tot_apparition_word_in_entities= df_tf_results[df_tf_results['word']== row['word']]['word'].count()
            idf = math.log(nbr_entities/tot_apparition_word_in_entities)
            tf=((row['occurrences']/tot_nbr_words_in_entity))
            df_tf_results.at[index, 'tfidf']=tf*idf
    return df_tf_results

def attribution_tf(ent,nbr,df_tf_results,banwords):
    """
    Sort and return the "nbr" top words with the highest tfidf result for the current entity (ent).
    
    Parameters : 
    ent (str) : Current selected entity in the UI.
    nbr (int) : Number of words with the highest tfidf to return.
    df_tf_results (dataframe) : Dataframe containing the calculated tfidf and the associated word, occurrences and related entity. 
    banwords (dict) : dictionnary containing the tfidf banwords for each entity. 
    
    Return :
    (list) : List of the topwords with the highest tfidf for the current selected entity. 
    """
    categoryStopList = ['+','%',',',"'",'(',')',':']
    top_words= df_tf_results.sort_values(by='tfidf',ascending=False).reset_index(drop=True).groupby('entity')
    words=[]
    i=0
    
    for index,row in top_words.get_group(ent).iterrows():
        word = row['word']
        occurrences= row['occurrences']
        if word != '' and not re.search('['+re.escape(''.join(categoryStopList))+']',word) and not re.search('\d', word) and not word == 'a' and word not in banwords : 
            words.append(word)
            i+=1
        if i==10: break
    return words

def attribution_tf_occurrences(top_tfidf_words,df_tf_results,current_entity):
    """
    Create a list containing the top tfidf words and their related occurrences, then used by a Jupyter widget to render them. 
    
    Parameters : 
    top_tfidf_words (list) : words with the highest tfidf score for the current entity.
    df_tf_results (dataframe) : Dataframe containing the calculated tfidf and the associated word, occurrences and related entity. 
    current_entity (str) : current working entity selected. 
    
    Return :
    (list) : List of the topwords with the highest tfidf for the current selected entity paired with their occurrences. 
    """
    tab = df_tf_results[(df_tf_results['entity'] == current_entity) & (df_tf_results['word'].isin(top_tfidf_words))]
    tab = tab.sort_values(by='tfidf',ascending=False)
    tf_occurrences= tab[['word','occurrences']].values.tolist()
    
    return tf_occurrences


def sigmoid(x, k=10):
    """
    Apply a sigmoid transformation to a value. 
    
    Parameters : 
    x (int) : input value to modify .
    
    Return :
    (int) : value which a sigmoid was apply on. 
    """   
    v  = 1 / (1 + np.exp(-k * (x - 0.5)))
    return round(v,2)

def calculate_homogeneity_score(df,ent_cat,k_sigmoid):
    """
    Calculate the homogeneity score for each entity. 
    
    Parameters : 
    df (dataframe) : Dataframe containing the annotated words, their occurrences, places and related entity.
    ent_cat (dict) : List of all the entities paired with their categories.
    k_sigmoid (int) : Sigmoid function slope value.
    
    Return :
    (dict) : dictionnary containing the entities and their associated homogeneity score. 
    """  
    res_homogeneity_score = {}
    
    for entity in ent_cat:
        total_words = []
        for index,row in df[df['entity']==entity].iterrows():
            for stem in row['stems']:
                total_words.extend([stem]*row['occurrences'])
        score = (len(total_words)-len(set(total_words)))/len(total_words)
        res_homogeneity_score[entity]= round(score,2)
    
    res_homogeneity_score_sigmoid = {key: sigmoid(value, k=k_sigmoid) for key, value in res_homogeneity_score.items()}
    return res_homogeneity_score_sigmoid
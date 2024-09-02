import nltk
from .tfidf import attribution_tf

def generate_ngrams(tokens, n):
    """
    Generates n-grams from a list of tokens.

    Parameters:
    tokens (list): list of string (words) from which to generate n-grams.
    n (int): number of tokens to include in each n-gram.

    Returns:
    (list): list of n-grams where each n-gram is a string of n tokens.
    """
    ngrams = nltk.ngrams(tokens, n)
    return [' '.join(gram) for gram in ngrams]

def calculate_n_grams(df,current_entity,df_tf_results,ban_words_tfidf):
    """
    Calculate all the existing n-grams from the annotations, for each word calculated by tfidf.

    Parameters:
    df (dataframe): contains all annotations and their caracteristics (text, occurrences, places, stems).
    current_entity (string) : the current entity of interest.
    df_tf_results (dataframe) : contains all the words annotated, and their associated tfidf score. 
    ban_words_tfidf (dict) : dictionnary containing the tfidf's banwords for each entity. 
    
    Returns:
    (dict): dictionnary containing all the existing n-grams from the annotations, for each word calculated by tfidf.
    """
    dfx = df.loc[(df['entity'] == current_entity)]
    dfx = dfx.filter(["text","occurrences"])
    sentences=[]
    for l in dfx.values.tolist(): 
        for i in range(0,l[1]):
            sentence = l[0].split()
            sentences.append(sentence)

    n_grams = {}
    key_words = attribution_tf(current_entity,10,df_tf_results,ban_words_tfidf)
    for word in key_words:
        n_grams[word]={}
        for sentence in sentences:
            if word in sentence:
                for n in range(2, len(sentence)+1):
                    results= generate_ngrams(sentence,n)
                    for res in results:
                        if word in res:
                            if res not in n_grams[word]:
                                n_grams[word][res]=1
                            else:
                                n_grams[word][res]+=1
    sorted_n_grams={}
    for keyword, n_grams in n_grams.items():
        sorted_n_grams[keyword] = dict(sorted(n_grams.items(), key=lambda x: x[1], reverse=True))
        
    return sorted_n_grams

def treate_n_grams(n_grams,maximum_n_gram):
    """
    Calculate and return the best existing n-grams, by avoiding any form of redundancy.
    
    Parameters:
    n_grams (dict): dictionnary containing all the existing n-grams from the annotations, for each word calculated by tfidf.
    maximum_n_gram (int): maximum number of ngrams we want to show to the user for each top tfidf word.

    Returns:
    (dict): dictionnary containing the best n-grams generated from the top tfidf words, by avoiding any form of redundancy in the n-grams.
    """
    treated_n_grams = {}
    threshold_occurrences = 2

    for current_keyword in n_grams:
        treated_n_grams[current_keyword] = {}
        i = 0
        for current_gram, current_occurrence in n_grams[current_keyword].items():
            
            if current_occurrence >= threshold_occurrences:
                to_remove = None
                for existing_keyword in treated_n_grams.keys() :
                    for existing_gram, existing_occurrence in treated_n_grams[existing_keyword].items():
         
                        if (current_gram in existing_gram or existing_gram in current_gram) and existing_occurrence == current_occurrence:
                            if len(current_gram)>len(existing_gram):
                                to_remove = [existing_keyword,existing_gram]
                            else : to_remove = current_gram  
    
                if to_remove :
                    if to_remove != current_gram:
                        del treated_n_grams[to_remove[0]][to_remove[1]]
                        treated_n_grams[current_keyword][current_gram] = current_occurrence
                elif i < maximum_n_gram :
                    treated_n_grams[current_keyword][current_gram] = current_occurrence
                    i += 1
                    
                
    return treated_n_grams

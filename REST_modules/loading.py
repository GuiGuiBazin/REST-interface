from .extraction import *
from .calculs import *
from .categorization import create_ban_words_tfidf

def load_data_annotations(file_path):
    
    # 1 - Extraction + Stemming of the data
    docs = load_from_brat(file_path, merge_all_fragments=True) 
    annotations = extract_annotations(docs,need_translation = False)
    annotations1 = stemming(annotations)
    annotations2 = annotations1
    
    # 2 - Initialization of major variables
    path = file_path
    data,ent_cat,ban_words_entities = createData(annotations2)
    df = pd.DataFrame(data, columns=["entity", "category", "text", "occurrences", "stems", "places"])
    df_tf_results = calculate_tfidf(ent_cat,df)
    homogeneity_score = calculate_homogeneity_score(df,ent_cat,10)
    ban_words_tfidf = create_ban_words_tfidf(ent_cat)
    current_entity = getEnt(ent_cat)[0]

    return path,ent_cat,ban_words_entities,df,df_tf_results,homogeneity_score,ban_words_tfidf,current_entity


def load_json(path,df,homogeneity_score,ent_cat):

    progress_ent_cat, progress_isNotFP, progress_isNotFN, progress_ban_words_entities, progress_df_results = load_progress(path)
    result = {}
    
    if progress_ent_cat:
        result['ent_cat'] = progress_ent_cat
    if progress_isNotFP:
        result['list_isNotFP'] = progress_isNotFP
    if progress_isNotFN:
        result['list_isNotFN'] = progress_isNotFN
    if progress_ban_words_entities:
        result['ban_words_entities'] = progress_ban_words_entities
    if progress_df_results is not None and not progress_df_results.empty:
        result['df_results'] = progress_df_results
    else:
        result['df_results'] = initiate_df_results(df, homogeneity_score, result.get('ent_cat', ent_cat))
    
    return result


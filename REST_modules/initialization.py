import pandas as pd
import ipywidgets as widgets
from .extraction.normalisation import getCat,getEnt

def initialize_globals():
    """
    Initialize and return the globals variables that will later contain the user's advandcements.
    
    Return :
    path                : string containing the dataset path. 
    ent_cat             : dictionnary of the entities and their related categories. 
    list_isNotFP        : list of the annotations considered as FP.
    list_isNotFN        : list of the annotations considered as FN.
    ban_words_entities  : dictionnary of the entities and their related ban words.
    df                  : main dataframe containing the annotations, and their associated caracteristics (occurrences, places, text). 
    df_tf_results       : dataframe containg the words and their tfidf score from the annotations for each entity.
    ban_words_tfidf     : dictionnary of each entity and their related ban tfidf word.
    homogeneity_score   : dictionnary of the entity and their related homogeneity score.
    df_results          : dataframe of the results of each entitity's results.
    """    
    path=None
    ent_cat={'entity1':['category?']}
    list_isNotFP = []
    list_isNotFN = []
    ban_words_entities={'entity1':["None"]}
    df=pd.DataFrame([['entity1','category?','text1',1,['stem1'],[['text1',1,5]]]],columns=["entity", "category", "text", "occurrences", "stems", "places"])
    df_tf_results = pd.DataFrame([['entity1','word1','occurrences1','tfidf1']],columns=['entity','word','occurrences','tfidf'])
    ban_words_tfidf = {'entity1': []}
    homogeneity_score = {'entity1': 0}
    df_results = pd.DataFrame(columns=["entity",
                                       "homogeneity",
                                       "TP",
                                       "FP",
                                       "FN",
                                       "precision",
                                       "precision_conf_inter_down",
                                       "precision_conf_inter_up",
                                       "recall","recall_conf_inter_down",
                                       "recall_conf_inter_up"])

    return path,ent_cat,list_isNotFP,list_isNotFN,ban_words_entities,df,df_tf_results,ban_words_tfidf,homogeneity_score,df_results

def initialize_widgets_globals(ent_cat):
    """
    Initialize and return the globals variables related to widgets.
    
    Return :
    value_button_results : boolean containing the value of the togglebutton 'results'. 
    current_category     : string containing the current printed category in the visualization tab. 
    options              : list of the categories printed in the visualization tab.
    list_spacing_regex   : list of the discontinued (spacing) regex that needs to.
    other_categories     : list of the annotations and their other associated categories.
    """    
    value_button_results = False
    current_category='category?'
    current_entity='entity1'
    options = [element.strip("[]") for element in getCat(current_entity,ent_cat)]
    list_spacing_regex=[]  
    other_categories = []
    
    return value_button_results,current_category,current_entity,options,list_spacing_regex,other_categories

def initialize_widgets(ent_cat,current_entity,ban_words_entities,options):
    """
    Initialize and return the globals interaction widgets. 
    
    Return :
    t0                        : widget of the loading tab.
    t1                        : widget of the visualization tab.
    t2                        : widget of the categorization tab.
    button_save               : widget of the save button located in the header.
    button_selection_entity   : widget of the entity selection dropdown list located in the header. 
    button_selection_category : widget of the category selection dropdown list located in the visualization tab. 
    button_categorization     : widget of the categorization button located in categorization tab.
    ban_word_tag              : widget of the tag containing the banwords for each entity located in the categorization tab. 
    space                     : empty widget, used to add more distance and visibility between the other widgets. 
    tabs                      : widget of all tool combined, minus the header. 
    """
    t0=None
    t1=None
    t2=None
    t3=None
    button_save=widgets.Button(description ="Save", button_style='info',icon='save',layout=widgets.Layout(width='70px'))   
    button_selection_entity = widgets.Dropdown(options = getEnt(ent_cat),value=current_entity)
    button_selection_category=widgets.Dropdown(options = options,value = options[0], description ='Category : ',layout={'width': '700px'})
    button_categorization = widgets.Button(button_style='primary',description = 'Categorization',icon='tasks')
    ban_word_tag = widgets.TagsInput(value=ban_words_entities[current_entity])
    space = widgets.HTML(layout=widgets.Layout(height='10px'))
    tabs = None
    
    return button_save,button_selection_entity,t0,t1,t2,t3,button_categorization,button_selection_category,ban_word_tag,space,tabs

def initialize_outputs():
    """
    Initialize and return the globals output widgets. 
    
    Return :
    output_results                    : output of the user's main advancements located in the header.
    output_load                       : output printing the loading messages located in the first tab.
    output_t1_visualization_category  : output of the category visualization located in the second tab.  
    output_t2_cat_infos               : output of the categorization results located in the third tab. 
    output_t2_donut                   : output of the donut located in the third tab. 
    output_t3_TEMP                    : empty output used in the last tab to hide the checkbox above the last table. 
    """
    output_results=widgets.Output()
    output_load=widgets.Output()
    output_t1_visualization_category = widgets.Output()
    output_t2_cat_infos = widgets.Output()
    output_t2_donut = widgets.Output()
    output_t3_TEMP = widgets.Output()
    
    return output_results,output_load,output_t1_visualization_category,output_t2_cat_infos,output_t2_donut,output_t3_TEMP
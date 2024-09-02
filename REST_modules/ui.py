from ipyfilechooser import FileChooser
import math
import ipywidgets as widgets
import pandas as pd
import copy
import nltk
from nltk.corpus import stopwords
import plotly.io as pio
import matplotlib.pyplot as plt
from IPython.display import HTML
from ipydatagrid import DataGrid, TextRenderer, BarRenderer, Expr, VegaExpr,CellRenderer
from bqplot import LinearScale, ColorScale, OrdinalColorScale, OrdinalScale
from unidecode import unidecode
import os

from .extraction import * 
from .calculs import *
from .visualization import *
from .initialization import *
from .categorization import *
from .loading import *

# Initialization of global variables
path,ent_cat,list_isNotFP,list_isNotFN,ban_words_entities,df,df_tf_results,ban_words_tfidf,homogeneity_score,df_results = initialize_globals()

# Initialization of widgets and their variables
value_button_results,current_category,current_entity,options,list_spacing_regex,other_categories = initialize_widgets_globals(ent_cat)
button_save,button_selection_entity,t0,t1,t2,t3,button_categorization,button_selection_category,ban_word_tag,space,tabs = initialize_widgets(ent_cat,current_entity,ban_words_entities,options)

# Global outputs
output_results,output_load,output_t1_visualization_category,output_t2_cat_infos,output_t2_donut,output_t3_TEMP = initialize_outputs()

###########
# HEADER #
##########

# Change entity

def on_selection_change_entity(change):
    """
    Handle the event triggered when the entity selection dropdown box changes in the UI:
    -Updates the global variables 'current_entity','current_category' and 'list_spacing_regex'.
    -Triggers the categorization process for the new entity selected. 
    -Refreshes the UI.     
    
    Parameters : 
    change (Change) : contains the new and old values of the dropdown entity selection
    """
    global current_entity,current_category,list_spacing_regex
    
    current_entity=change.new
    list_spacing_regex=calculate_list_spacing_regex(current_entity,ent_cat)
    update_tabs()
    current_category=getCat(current_entity,ent_cat)[0]
    on_visualization_categorie_change("None")
    categorization()
    display_categorization_results()
    create_t2_donut()
    button_categorization.button_style='primary'
    
# Results

def create_button_results():
    """
    Creates a toggle button to display the entities results. 
    Triggers the 'print_dg_results function to display the results' when the button is pressed. 
    
    Return :
    widgets.ToggleButton : the toggle button widget to display the entities results. 
    """
    def change_value_button_results(value):
        global value_button_results
        value_button_results = value.new
        print_dg_results(df_results)
    button_results = widgets.ToggleButton(description ="Results", button_style='info',icon='poll',layout=widgets.Layout(width='100px'))
    button_results.observe(lambda change :change_value_button_results(change),names='value')
    return button_results

def update_df_result(df_metrics,df,df_metrics_locations):
    """
    Update the entities metrics results:
    -Calculates the confidence intervals with the bootstrap method. 
    -Updates the global variable 'df_metrics' with the new categorization results.
    -Displays the new results.
    
    Parameters : 
    df_metrics (dataframe) : dataframe containing the precision and recall score in each category.
    df (dataframe) : Dataframe containing the annotated words, their occurrences, places and related entity.
    df_metrics_locations (dataframe) : contains the locations of each categorized word, and whether they are true positive, false positive or false negative.
    """   
    global df_results
    Xdf = copy.deepcopy(df[df['entity']==current_entity])
    bootstrap_results= estimate_confidence_intervals_bootstrap(Xdf,current_entity,df_metrics_locations,draw_number=1000, alpha=5.0)
    df_results = update_df_results(df_results,df,current_entity,homogeneity_score,df_metrics,bootstrap_results)
    print_dg_results(df_results)

def print_dg_results(df_results):
    """
    Handle the events to display the entities metrics results:
    -Creates a datagrid from the 'df_results' dataframe
    -Displays the datagrid in the 
    
    Parameters : 
    df_metrics (dataframe) : dataframe containing the precision and recall score in each category.  
    
    Return :
    widgets.Output() : the output widget to display the results. 
    """
    with output_results:
        output_results.clear_output()
        if value_button_results:
            dg_results = create_dg_results(df_results)
            display(dg_results)
    return output_results

# Save json

def initiate_button_save(button_save,button_results):
    """
    Handle the events if the button 'button_save' is pressed:
    -Saves the current progress on a json file. 
    -prints on the output 'output_results' a message.
    -Changes the value of the toggle button displaying the results. 
    
    Parameters : 
    button_save (widgets.Button) : button that triggers the events of saving the progress.
    button_results (widgets.ToggleButton) : button that displays the entities metrics results.
    
    Return :
    (widgets.Button) : button that save the current progress.
    """
    def button_save_on_click(b):
        save_progress(path,ent_cat,list_isNotFP,list_isNotFN,ban_words_entities,df_results)
        button_results.value = False
        button_save.button_style='info'
        with output_results:
            output_results.clear_output()
            print("Progress saved")
    button_save.on_click(button_save_on_click)
    return button_save

######################
# TAB 0 : LOAD FILES #
######################

def initiate_loading(file_path):
    """
    Fonction that handles the event when the loading progress is triggered:
    -Retrieves the annotations (highlights) of the corpus from the selected path.
    -Initiates the following global variables from the corpus annotations (highlights) :
        -'path','ent_cat','ban_words_entities','df','','df_tf_results','homogeneity_score','ban_words_tfidf','current_entity'.
    -Checks if a saving files exists in the corpus directory, and change the value of the following global variables :
        -'ent_cat','list_isNotFP','list_isNotFN','ban_words_entities','df_results'
    -Updates the UI with the new global variables values. 
    -Displays on the output 'output_load' the loading process to the user. 
    
    Parameters : 
    file_path (string) : string containing the dataset path. 
    
    """
    global path,ent_cat,current_entity
    global df,df_tf_results
    global ban_words_entities,ban_words_tfidf
    global homogeneity_score
    global list_isNotFP,list_isNotFN,df_results

    # 1 - load data annotation and possible progress
    path,ent_cat,ban_words_entities,df,df_tf_results,homogeneity_score,ban_words_tfidf,current_entity=load_data_annotations(file_path)
    var=load_json(path,df,homogeneity_score,ent_cat)
    ent_cat = var.get('ent_cat', ent_cat)
    list_isNotFP = var.get('list_isNotFP', list_isNotFP)
    list_isNotFN = var.get('list_isNotFN', list_isNotFN)
    ban_words_entities = var.get('ban_words_entities', ban_words_entities)
    df_results = var.get('df_results', df_results)

    # 2 - Update of the ui
    button_selection_entity.options = getEnt(ent_cat) # update_tabs
    on_visualization_categorie_change({'new':getCat(current_entity,ent_cat)[0]})
    categorization()
    display_categorization_results()
    create_t2_donut()
    update_tabs()
    
    # 3 - Load visualization
    with output_load:
        output_load.clear_output()
        print(f"Selected Path: {file_path}")
        print("Extraction and Normalisation Done")
        print("Entities : "+str(getEnt(ent_cat)))
    
def create_t0():
    """
    Creates and assemble the widgets that forms the tab 'load data', handle the following events.
    -Creates a fileChooser widget to select the corpus directory.
    -Triggers the loading process if the button 'button_accept_file' is pressed.
    
    Return :
    (widgets.VBox) : Group of widgets containing the file selection section and the load output, forming the widgets in the 'load data' tab.
    """
    file_select = FileChooser()
    button_accept_file = widgets.Button(description='Load', button_style='Primary', icon='upload', layout=widgets.Layout(width='100px'))
    def on_button_accept_file_clicked(b):
        initiate_loading(file_select.selected)
    button_accept_file.on_click(on_button_accept_file_clicked)
    file_selection = widgets.HBox([button_accept_file, file_select])
    t0 = widgets.VBox([file_selection, output_load])
    return t0

##############################
# TAB I : DATA VISUALIZATION #
##############################

def on_visualization_categorie_change(change):
    """
    Change the visualization of the category's annotations, and displays it with the output "output_t1_visualization_category".
    
    Parameters : 
    change (Change) : contains the new and old values of the dropdown category selection
    """
    df1 = visualize_category_selection(change,df,current_entity)
    with output_t1_visualization_category:
        output_t1_visualization_category.clear_output()
        display(df1)

def create_t1_category_selection():
    """
    Create and assemble the widgets presents in the first accordion of the tab 'data visualization':
    -Creates the event trrigering the visualization of the category's annotations.
    -Merges the dropdown list of the categry selection and the related output. 
    
    Return :
    (widgets.VBox) : VBox composed of the category selection and the related output. 
    
    """    
    button_selection_category.observe(lambda change :on_visualization_categorie_change(change),names='value')
    t1_category_selection = widgets.VBox([button_selection_category,output_t1_visualization_category])
    return t1_category_selection

def on_button_search_clicked(output_concordancer,word):
    """
    Handle the events when the button search is clicked for the concordancer:
    -Loads the corpus
    -Search the occurrences of the word in the corpus and saves it in a datagrid.
    -Display the datagrid in the output 'output_concordancer'.
    
    Parameters : 
    output_concordancer (widgets.Output) : Output displaying the concordancer's results.
    word (string) : string containing the word or term that needs to be search in the corpus.
    """
    docs = load_from_brat(path, merge_all_fragments=True)
    res_concordancer = calculate_concordancer(word,current_entity,path,docs)
    with output_concordancer:
        output_concordancer.clear_output()
        display(res_concordancer)
    
def create_t1a2():
    """
    Creates and assembles the widgets of the second accordion of the tab 'load data', corresponding to the concordancer section.

    Return :
    (widgets.VBox) : VBox containg the typing area, the button, and the output of the concordancer section.
    """
    output_concordancer = widgets.Output()
    text_t1a2 = widgets.Text(value="",description ="Word :")
    button_search = widgets.Button(description = 'Seach',button_style='primary',icon='search')
    button_search.on_click(lambda _: on_button_search_clicked(output_concordancer,text_t1a2.value))
    t1a2 = widgets.VBox([widgets.HBox([text_t1a2,button_search]),output_concordancer])
    return t1a2

def create_t1():
    """
    Creates the accordeon of the tab 'Load data', composed of the highlights (annotations) visualization and the concordancer.
    
    Return :
    (widgets.Accordion) : Accordeon composed of the highlights (annotations) visualization and the concordancer.
    """
    global t1
    t1 = widgets.Accordion([create_t1_category_selection(),create_t1a2()],titles=('Highlights visualization','Concordancer'))
    return t1


###########################
# TAB II : Categorization #
###########################

# T2a1 : Category modification

def create_frequent_tfidf_texts(df,current_entity,df_tf_results,ban_words_tfidf):
    """
    Manage the events to display to the user frequent terms that he can use to create categories:
    -Calculates the top tfidf words and ngrams characterizing the current entity text highlights.
    -Creates the Tags widgets of the top tfidf words and ngrams.
    -Assembles those widgets with their corresponding occurrences.
    
    Parameters : 
    df (dataframe) : Dataframe containing the annotated words, their occurrences, places and related entity.
    current_entity (str) : current working entity selected. 
    df_tf_results (datframe) : Dataframe of the words in the entity text highlights and their corresponding tfidf score.
    ban_words_tfidf (dict) : Contains each entity and their related tfidf banwords. 
    
    Return :
    (widgets.VBox) : VBox composed of the widgets displaying the words with the highest tfidf score, as well as the most frequent ngrams of the entity text highlights. 
    """
    #Tfidf widgets
    top_tfidf_words = attribution_tf(current_entity,10,df_tf_results,ban_words_tfidf[current_entity])
    tf_occurrences = attribution_tf_occurrences(top_tfidf_words,df_tf_results,current_entity)
    tag_tf_occurrences =[]
    for word, occurrence in tf_occurrences:
        tag_tf = widgets.TagsInput(value=word,disabled=True)
        tag_tf.observe(on_tfidf_removed,names='value')
        tag_tf_occurrences.extend([tag_tf,widgets.HTML(value="("+str(occurrence)+")")])         
    widgets_tf_words= widgets.HBox(tag_tf_occurrences)
    
    #Ngrams widgets
    n_grams = calculate_n_grams(df,current_entity,df_tf_results,ban_words_tfidf[current_entity])
    treated_n_grams = treate_n_grams(n_grams,5)

    list_widgets_n_grams=[]
    for keyword in treated_n_grams:
        list_widget=[]
        text=widgets.HTML(value=str(keyword)+":")
        list_widget.append(text)
        add=False
        for gram, occurrence in treated_n_grams[keyword].items():
            list_widget.extend([widgets.TagsInput(value=gram),widgets.HTML(value="("+str(occurrence)+")")])
            add=True
        if add: list_widgets_n_grams.append(widgets.HBox(list_widget))
    
    return widgets.VBox([widgets.VBox([widgets_tf_words]),widgets.VBox(list_widgets_n_grams)])

def on_button_add_category_clicked(_):
    """
    Handle the events when a new category is created:
    -Adds a new empty category to the 'ent_cat' global dictionnary.
    -Creates again the categorization tab. 
    -Change the color of the categorization button. 
    
    Parameters : 
    _ (object): Placeholder for the event object passed by the button click event. It is not used in this function
    """
    global tabs,ent_cat
    ent_cat[current_entity].append("[]")
    tabs.children = (t0,t1,create_t2(),t3)
    button_categorization.button_style='warning'
    
def remove_ent_cat(i):
    """
    Handles the event when a category must be deleted:
    -Deletes the results in the 'list_isNotFP' and 'list_isNotFN' lists related to the deleted category. 
    -Remove the category from the global dictionnary 'ent_cat'
    -Remove the possible other empty categories in the UI. 
    -Update the UI and change the color of the categorization button. 
    
    Parameters : 
    i (int) : number corresponding of the deleted category. 
    """
    global list_isNotFP,list_isNotFN,ent_cat
    cat = getCat(current_entity,ent_cat)[i]
    list_isNotFP = [values_notFP for values_notFP in list_isNotFP if values_notFP[0] != cat]
    list_isNotFN = [values_notFN for values_notFN in list_isNotFN if values_notFN[0] != cat]
    ent_cat[current_entity].remove(ent_cat[current_entity][i])
    ent_cat = remove_empty_categories(ent_cat,current_entity)
    update_tabs()
    button_categorization.button_style='warning'

def on_tag_change(change,i):
    """
    Handles the events when a category value is changed:
    -Change the value of the cateogry name of the elements in the global variable 'list_isNotFP'.
    -Change the name of the category in the global dictionnary 'ent_cat'.
    -Modify the categorization button color. 
    
    Parameters : 
    change (Change) : contains the old and new values of the category name in the corresponding TagsInput widget.
    i (int) : number corresponding to the modified TagsInput widget. 
    """
    global list_isNotFP,ent_cat
    list_isNotFP = modify_list_isNotFP(list_isNotFP,eval(repr(change['new'])))
    ent_cat[current_entity][i] = repr(change['new'])  
    button_categorization.button_style='warning'
    
def on_ban_word_tag_change(change):
    """
    Handles the events when the banword widget is modified:
    -Modify the global dict 'ban_words_entities' by the the values. 
    -Change the color of the save button and the categorization button.  
    
    Parameters : 
    change (Change) : contains the new values of the banwords TagsInput.
    """
    global ban_words_entities
    if not change['new']:
        ban_words_entities[current_entity] = ["None"]
        ban_word_tag.value = ["None"]
    else : ban_words_entities[current_entity] = change['new']
    button_save.button_style='info'
    button_categorization.button_style='warning'

def create_categories_tags():
    """
    Intitializa the widgets allowing the user to create, modify and delete categories:
    -Creates the TagsInput widgets with the current entity categories, from the dictionnary 'ent_cat'.
    -Initializes the event functions linked to every widget. 
    -Assembles and returns the widgets of the 'category modification' section of the tab 'Categorization'

    Return :
    (widgets.VBox) : merges the following widgets : categories TagsInput (modifies categories), delete buttons (delete categories), add button (add a new category), banwords TagsInput (modify the banwords list).
    """
    tags=[]
    output_test = widgets.Output()
    for i,category in enumerate(ent_cat[current_entity]):
        if category == "category?":
            continue
        term=eval(ent_cat[current_entity][i])
        tag = widgets.TagsInput(value=term,allow_duplicates=True)
        tag.observe(lambda change,inti=i : on_tag_change(change,inti), names='value')
        button_supp = widgets.Button(button_style='danger',icon='minus-circle',layout=widgets.Layout(width='40px'))
        button_supp.on_click(lambda _, inti=i: remove_ent_cat(inti))
        text = widgets.Label(value="Category "+str(i)+" :")
        tags.append(widgets.HBox([text,button_supp,tag]))
    ban_word_tag.value = ban_words_entities[current_entity]
    ban_words_tags= widgets.HBox([widgets.Label(value="Banwords : "),ban_word_tag])
    tags=widgets.VBox([widgets.VBox(tags),ban_words_tags])
    button_add_category = widgets.Button(button_style='success',icon='plus-circle',layout=widgets.Layout(width='70px'))
    button_add_category.on_click(on_button_add_category_clicked)
    return widgets.VBox([tags,button_add_category])

def on_tfidf_removed(change):
    """
    Handles the events when a tfidf word is deleted:
    -Removes the word from the global dictionnary 'ban_words_tfidf'.
    -Update the ui.
    
    Parameters : 
    change (Changes) : contains the new and old value of the deleted tfidf TagsInput.
    """
    global ban_words_tfidf,df_tf_results
    ban_words_tfidf[current_entity].append(change['old'][0])
    update_tabs()
    
def on_button_categorization_clicked(b):
    """
    Handles the event when the categorization button is clicked:
    -Change the color of the button 'button_categorization'.
    -Initiate the display process of the categorization results in the third section of the tab 'categorization'.
    -Update the ui.
    -Change the color of the save button, and update the display the the entity text highlight in the tab 'Data visualization'.
    
    Parameters : 
    b (Button): The button object that triggers the event when clicked. Not used within the function
    """
    button_categorization.button_style='primary'
    categorization()
    display_categorization_results()
    create_t2_donut()
    update_tabs() #location metrics calculated with create_tab3()
    button_save.button_style='warning'
    on_visualization_categorie_change("None")

def create_t2a1():
    """
    Creates and assembles the widgets of the section I and II of the tab 'Categorization':
    -Initiates the texts displayed in this tab. 
    -Initiates the widgets of the 'frequent terms' and 'category creation' sections.
    -Creates the Accordion widgets related to the display of recommanded distance, for the spacing regex. 
    
    Return :
    (widgets.VBox) : widget composed of the widgets of the section I and II of the tab 'Categorization'.
    """
    title1 = "I - Frequent terms"
    title2 = "II -  Category creation"
    t2a1_texts= create_frequent_tfidf_texts(df,current_entity,df_tf_results,ban_words_tfidf)
    t2a1_title1 = widgets.HTML(value=f"<h2 style='height: 20px; line-height: 20px; text-align: left; display: flex; align-items: center;'>{title1}</h2>")
    t2a1_title2 = widgets.HTML(value=f"<h2 style='height: 20px; line-height: 20px; text-align: left; display: flex; align-items: center;'>{title2}</h2>")
    t2a1_tags = create_categories_tags()
    accordion_recommendations = create_accordion_recommendations(list_spacing_regex,path,current_entity,df)
    
    return widgets.VBox([t2a1_title1,t2a1_texts,space,accordion_recommendations,t2a1_title2,t2a1_tags,space,button_categorization])

# Category visualization

def create_t2_donut():
    """
    Handles the vents to display the categorized highlights figure to the user:
    -Calculate the number of categorized annotations (total and unique).
    -Creates a donut figure of the categorized highlights.
    -Displays the figure with the global output 'output_t2_donut'.
    """
    cat_infos=create_categories_infos(df,ent_cat,current_entity)
    fig = create_categories_donut(cat_infos,current_entity)
    with output_t2_donut:
        output_t2_donut.clear_output()
        fig.show()

def display_categorization_results():
    """
    Handles the vents to display the categorized highlights table to the user:
    -Calculate the number of categorized annotations (total and unique).
    -Stores the results in a dataframe
    -Displays the table with the global output 'output_t2_cat_infos'.
    """
    def apply_colors(row,col):
        """
        Changes the colors of a dataframe's row. 
        Parameters :
        row : dataframe's row.
        col (list) : list of colors in their hexadecimal form. 
        Return:
        (row) the styled row.
        """
        color_index = row.name % len(col)
        first_col_background = '{}90'.format(col[color_index])  
        return ['background-color: {}'.format(first_col_background)] + [''] * (len(row) - 1)  
    colors = ['#ced4da',  '#147df5',  '#0000ff',  '#580aff',  '#0aefff',  '#ff0000',  '#ff7f00',  '#ff00ff',  '#deff0a',  '#0aff99']
    cat_infos=create_categories_infos(df,ent_cat,current_entity).sort_values(by='total_annotations_number',ascending=False)
    styled_cat_infos = cat_infos.style.apply(apply_colors, axis=1, col=colors)
    with output_t2_cat_infos:
        output_t2_cat_infos.clear_output()
        display(styled_cat_infos)

def categorization():
    """
    Call the fonction that Modifies the global variables related to the categorization of highlights.
    The modified global variables are the following:
        -'df','current_entity','other_categories','list_spacing_regex'.
    """
    global df,current_entity,other_categories,list_spacing_regex    
    df,other_categories,list_spacing_regex = calculate_categorization(df,ent_cat,current_entity,other_categories,list_spacing_regex)

def create_t2a2():
    """
    Creates and assembles the widgets of the third section of this tab together.  
    
    Return :
    (widgets.VBox) : widget composed of the title of the section, and the outputs displaying the categorization results (table and donut figure).
    """
    cat_infos=create_categories_infos(df,ent_cat,current_entity)
    title = "III - Categorization distribution"
    t2a2_title = widgets.HTML(value=f"<h2 style='height: 30px; line-height: 30px; text-align: left; display: flex; align-items: center;'>{title}</h2>")
    t2a2_outputs = widgets.HBox([output_t2_donut,output_t2_cat_infos])
    return widgets.VBox([t2a2_title,t2a2_outputs])

def create_t2():
    """
    Creates and assembles the widgets in the tab 'Categorization'.
    
    Return :
    (widgets.VBox) : widget composed of the three following sections :
        -'Frequent terms', 'Category creation', 'Categorization distribution'
    """
    global t2
    t2 = [create_t2a1(),create_t2a2()]
    return widgets.VBox(t2)

#############################
# TAB III : METRICS RESULTS #
#############################

def change_visualization_metric(index, df_metrics_locations,t3_output1_entity_results ,t3_output2_metrics_results ,t3_output_text_highlight, df_metrics,container_checkBox_isNotFPorFN, output_t3_TEMP, t3_output3_metrics_locations, create_grid_metrics_locations):
    """
    Updates the visualization in the third tab when a specific metric or location is selected.

    This function handles the selection event from the metrics locations table. 
    Based on the selection, it updates the highlighted text, checkbox state, and the content of multiple output widgets that display the metrics and related information.

    Parameters: 
    index (int): Index of the selected row in the metrics locations table.
    df_metrics (dataframe) : dataframe containing the precision and recall score in each category.  
    t3_output1_entity_results (widgets.Output): Output widget for displaying entity results.
    t3_output2_metrics_results (widgets.Output): Output widget for displaying metrics results.
    t3_output_text_highlight (widgets.HTML): HTML widget for displaying highlighted text with annotations and motifs.
    df_metrics (DataFrame): DataFrame containing precision and recall scores for each category.
    container_checkBox_isNotFPorFN (widgets.VBox): Container for the checkbox widget.
    output_t3_TEMP (widgets.Output): Placeholder output widget used for displaying temporary content.
    t3_output3_metrics_locations (widgets.Output): Output widget for displaying the metrics locations table.
    create_grid_metrics_locations (function): Function to create the grid for displaying metrics locations.
    """
    global current_entity
    global list_isNotFP,list_isNotFN
    
    text = df_metrics_locations.iloc[index]['text']
    annotation = df_metrics_locations.iloc[index]['annotation'] 
    category = df_metrics_locations.iloc[index]['category'] 
    file = df_metrics_locations.iloc[index]['file'] 
    places = df_metrics_locations.iloc[index]['places'] 
    motif = df_metrics_locations.iloc[index]['motif']
    result = df_metrics_locations.iloc[index]['result']
    value_isNotFP = bool(df_metrics_locations.iloc[index]['isNotFP'])
    value_isNotFN = bool(df_metrics_locations.iloc[index]['isNotFN'])
        
    # display Text + annotation + motif
    text2=text
    if annotation != "no annotation":
        common_string = compare_common_string(annotation, text)
        text2 = text2.replace(common_string, f'<span style="background-color: orange; padding: 4px 0;">{common_string}</span>')
    text2 = text2.replace(motif, f'<span style="background-color: red;">{motif}</span>') 
    t3_output_text_highlight.value = text2
        
    # Checkbox
    def on_checkBox_isNotFPorFN_clicked(change,instruction):
        """
        Handles the click event for the 'isNotFP' or 'isNotFN' checkbox.

        This function toggles the value of 'isNotFP' or 'isNotFN' for the selected row 
        in the metrics locations DataFrame. It also updates the result status (e.g., TP, FP, 
        FN, Discarded) and refreshes the corresponding widgets and tables.

        Parameters:
        change (dict): The change event dictionary triggered by the checkbox.
        instruction (str): Determines whether the checkbox is handling 'FP' or 'FN'.
        """
        global dg_metrics_results
        
        column= 'isNotFP'
        result= ["TP(corr)","FP"]
        bool_FNorFP_value = [True,False]
        target_list = list_isNotFP
        if instruction=="FN":
            column= 'isNotFN'
            result= ["Discarded","FN"]
            bool_FNorFP_value = [False,True]
            target_list = list_isNotFN
            
        df_metrics_locations.loc[index, column] = not df_metrics_locations.loc[index, column] 
        if df_metrics_locations.loc[index, column]:
            df_metrics_locations.loc[index, 'result'] = result[0]
            target_list.append([category,result[0],bool_FNorFP_value[0],bool_FNorFP_value[1],text,file,places,annotation,motif])
        else : 
            df_metrics_locations.loc[index, 'result'] = result[1]
            target_list.remove([category,result[0],bool_FNorFP_value[0],bool_FNorFP_value[1],text,file,places,annotation,motif])
            
        dg_metrics_locations = create_grid_metrics_locations(df_metrics_locations, current_entity)
        dg_metrics_locations.observe(lambda *_: change_visualization_metric(dg_metrics_locations.selections[0]['r1'], df_metrics_locations,t3_output1_entity_results,t3_output2_metrics_results ,t3_output_text_highlight,df_metrics, container_checkBox_isNotFPorFN, output_t3_TEMP, t3_output3_metrics_locations, create_grid_metrics_locations), names='selections')
        
        df_metrics = calculate_df_metrics(df_metrics_locations)
        dg_metrics_results = generate_dg_metrics_results(df_metrics)
        
        if path: update_df_result(df_metrics,df,df_metrics_locations)
        # Refresh of tab3 outputs
        with t3_output1_entity_results:
            t3_output1_entity_results.clear_output()
            display(create_dg_results(df_results[df_results["entity"]==current_entity]))
        with t3_output2_metrics_results:
            t3_output2_metrics_results.clear_output()
            display(dg_metrics_results)
        with t3_output3_metrics_locations:
            t3_output3_metrics_locations.clear_output()
            display(dg_metrics_locations)   
        button_save.button_style = 'warning' 
        
    if result == "FP" or result == "TP(corr)" or result == "FN" or result == "Discarded":
        description = 'Consider this FP as a TP'
        value = value_isNotFP
        instruction = "FP"
        if result == "FN" or result == "Discarded":
            description = 'Disregard this FN as an annotation'
            value = value_isNotFN
            instruction = "FN"
        checkBox_isNotFPorFN = widgets.Checkbox(value = value, disabled = False, indent = False ,description = description)
        checkBox_isNotFPorFN.observe(lambda change: on_checkBox_isNotFPorFN_clicked(change,instruction),names='value')
        container_checkBox_isNotFPorFN.children= [checkBox_isNotFPorFN,output_t3_TEMP]
    
    else : 
        container_checkBox_isNotFPorFN.children = [output_t3_TEMP,output_t3_TEMP]        
    
def create_t3a1():
    """
    Creates, displays and assembles the following tables, characterizing the tab 'Metrics':
        -The categorization metrics results of the entity, in the output 't3_output1_entity_results'.
        -The categorization metrics results of the entity's categories, in the output 't3_output2_metrics_results'.
        -The texts in the highlights that matches with the categories, in the output 't3_output3_metrics_locations'.
    It handles the following events : 
    -Creation of the tabs
    -Calculation of the metrics dataframes.
    -Creation of datagrids from the dataframes
    -Merges the widgets into one
    
    Return :
    (widgets.VBox) : widgets containg the three outputs of the tab, displaying the entity categorization's results. 
    """
    t3_output1_entity_results = widgets.Output()
    t3_output2_metrics_results = widgets.Output()
    t3_output3_metrics_locations = widgets.Output()
    t3_output_text_highlight = widgets.HTML()
    container_checkBox_isNotFPorFN = widgets.VBox([output_t3_TEMP,output_t3_TEMP]) 
    
    title = "I - Entity metrics results"
    t3a1_title = widgets.HTML(value=f"<h2 style='height: 30px; line-height: 30px; text-align: left; display: flex; align-items: center;'>{title}</h2>")
    title = "II - Categories metrics results"
    t3a2_title = widgets.HTML(value=f"<h2 style='height: 30px; line-height: 30px; text-align: left; display: flex; align-items: center;'>{title}</h2>")
    title = "III - Summary table of found pattern"
    t3a3_title = widgets.HTML(value=f"<h2 style='height: 30px; line-height: 30px; text-align: left; display: flex; align-items: center;'>{title}</h2>")
    
    # calculation of metrics dataframes
    metrics_locations = calculate_location_metrics(current_entity,ent_cat,path,df,other_categories,list_isNotFP,list_isNotFN,ban_words_entities)
    df_metrics_locations = pd.DataFrame(metrics_locations[current_entity],columns=["category", "result","isNotFP","isNotFN","text","file", "places","annotation","motif"])
    dg_metrics_locations = create_grid_metrics_locations(df_metrics_locations,current_entity)
    dg_metrics_locations.observe(lambda *_: change_visualization_metric(dg_metrics_locations.selections[0]['r1'], df_metrics_locations,t3_output1_entity_results, t3_output2_metrics_results ,t3_output_text_highlight,df_metrics ,container_checkBox_isNotFPorFN, output_t3_TEMP, t3_output3_metrics_locations, create_grid_metrics_locations), names='selections')
    df_metrics = calculate_df_metrics(df_metrics_locations)
    dg_metrics_results = generate_dg_metrics_results(df_metrics)
    
    # update of df_results
    if path: update_df_result(df_metrics,df,df_metrics_locations)
    
    with t3_output1_entity_results:
        t3_output1_entity_results.clear_output()
        display(create_dg_results(df_results[df_results["entity"]==current_entity]))
    
    with t3_output2_metrics_results:
        t3_output2_metrics_results.clear_output()
        display(dg_metrics_results)
        
    with t3_output3_metrics_locations:
        t3_output3_metrics_locations.clear_output()
        display(dg_metrics_locations)
    
    return widgets.VBox([t3a1_title,t3_output1_entity_results,t3a2_title,t3_output2_metrics_results,t3a3_title,t3_output_text_highlight,container_checkBox_isNotFPorFN,t3_output3_metrics_locations])

def create_t3():
    """
    Creates and modify the widgets of the tab 'Metrics'
    Return :
    (widgets.VBox) : widgets containg the three outputs of the last tab, displaying the entity categorization's results. 
    """
    global t3
    t3 = create_t3a1()
    return t3

########
# TABS #
########

def Launch_REST():
    """
    Main fonction that initiate the tabs:
    -Sets the event fonction of the entity selection button, ban word InputTags, Categorization button.
    -Creates the button printing the advandcements results. 
    -Assembles the widgets of the tabsand the header as the interface. 
    -Display the interface
    """
    global tabs,t0   
    button_selection_entity.observe(on_selection_change_entity, names='value')
    ban_word_tag.observe(lambda change : on_ban_word_tag_change(change), names='value')
    button_categorization.on_click(on_button_categorization_clicked)
    tabs = widgets.Tab([create_t0(),create_t1(), create_t2(),create_t3()],selected_index=0)

    display(HTML("<style>.widget-dropdown select, .widget-dropdown .widget-readout { font-size: 14px !important; }</style>"))  
    t0 = create_t0()
    tabs = widgets.Tab([t0,create_t1(), create_t2(),create_t3()],selected_index=0)
    tabs.set_title(0, 'Load data')
    tabs.set_title(1, 'Data visualization')
    tabs.set_title(2, 'Categorization')
    tabs.set_title(3, 'Metrics')
    button_results = create_button_results()
    selection_results_save = widgets.HBox([button_selection_entity,button_results,initiate_button_save(button_save,button_results)])
    interface = widgets.VBox([space,space,selection_results_save,output_results,tabs],layout={'border': '2px solid lightblue','width':'100%'})
    on_visualization_categorie_change({'new':getCat(current_entity,ent_cat)[0]})
    display(interface)
    
def update_tabs():
    """
    Updates the tabs of the ui.
    Refresh the options of the selection category button in the tab 'data visualization'
    """
    global tabs,button_selection_category
    tabs.children = (t0,t1,create_t2(),create_t3())
    button_selection_category.options = [element.strip("[]") for element in getCat(current_entity,ent_cat)]

    



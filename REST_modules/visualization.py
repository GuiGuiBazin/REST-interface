import pandas as pd
from ipydatagrid import DataGrid, TextRenderer, BarRenderer, Expr, VegaExpr,CellRenderer
from bqplot import LinearScale, ColorScale, OrdinalColorScale, OrdinalScale
import plotly.graph_objects as go
import copy

def create_dg_results(df_results):
    if df_results.empty : 
        columns=["entity", "homogeneity", "TP", "FP", "FN", "precision", "precision_confidence_interval", "recall","recall_confidence_interval"]
        dg_results = DataGrid(pd.DataFrame(columns=columns),layout={"height":"50px","width":"800px"},base_row_size=20)
        return dg_results

    df_temp = copy.deepcopy(df_results)
    df_temp.loc[:, 'precision_confidence_interval'] = df_temp.apply(lambda row: f"[{row['precision_conf_inter_down']}, {row['precision_conf_inter_up']}]",axis=1)
    df_temp.loc[:, 'recall_confidence_interval'] = df_temp.apply(lambda row: f"[{row['recall_conf_inter_down']}, {row['recall_conf_inter_up']}]",axis=1)
    
    df_temp = df_temp.drop(columns=['precision_conf_inter_down', 'precision_conf_inter_up','recall_conf_inter_down', 'recall_conf_inter_up'])
    
    height= str((len(df_temp)+1)*20+5)+"px"
    columns=["entity", "homogeneity", "TP", "FP", "FN", "precision", "precision_confidence_interval", "recall","recall_confidence_interval"]
    dg_results = DataGrid(df_temp[columns],column_widths={"entity":400,"homogeneity":100,"TP":50,"FP":50,"FN":50,"precision":80,"precision_confidence_interval": 120,"recall":80,"recall_confidence_interval":120},layout={"height":height,"width":"1120px"},base_row_size=20)
    
    renderers = {
        "TP": TextRenderer(horizontal_alignment="center", bold=True, background_color="#90be6d"),
        "FP": TextRenderer(horizontal_alignment="center", bold=True, background_color="#f94144"),
        "FN": TextRenderer(horizontal_alignment="center", bold=True, background_color="#f9c74f"),
        "precision_confidence_interval": TextRenderer(horizontal_alignment="center", bold=True),
        "recall_confidence_interval": TextRenderer(horizontal_alignment="center", bold=True),
        "precision": BarRenderer(bar_horizontal_alignment="center", horizontal_alignment="center",bar_color=ColorScale(min=0, max=1,scheme="cividis"),bar_value=LinearScale(min=0, max=1)),
        "recall": BarRenderer(bar_horizontal_alignment="center", horizontal_alignment="center",bar_color=ColorScale(min=0, max=1, scheme="cividis"),bar_value=LinearScale(min=0, max=1)),
        "homogeneity": BarRenderer(horizontal_alignment="center",bar_color=ColorScale(min=0, max=1, scheme="cividis"),bar_value=LinearScale(min=0, max=1))
    }
    dg_results.renderers = renderers
    
    return dg_results

def create_categories_donut(cat_infos,current_entity):
    """
    Create a donut representing categorizations' results.
    
    Parameters : 
    cat_infos (dict) : Dictionnay of all the entities paired with their categories.
    current_entity (string) : String of the current working entity.

    Return :
    (Figure) : Chart of the categorizaitons' results. 
    """
    colors = ['#ced4da',  '#147df5',  '#0000ff',  '#580aff',  '#0aefff',  '#ff0000',  '#ff7f00',  '#ff00ff',  '#deff0a',  '#0aff99']
    fig = go.Figure(data=[go.Pie(labels=cat_infos['category'].tolist(),
                                 values=cat_infos['total_annotations_number'].tolist(),
                                 hole=0.5,
                                 showlegend=False,
                                 marker=dict(colors=colors))],
                    layout=go.Layout(template="plotly_dark"))
    fig.update_layout(title="Categories distribution for the entity : "+str(current_entity),title_x=0.5)
    fig.update_layout(margin=dict(t=50))     
    
    return fig

def visualize_category_selection(change,df,current_entity):
    """
    Create a df containing the annotations belonging to the selected category.
    
    Parameters : 
    change (dict) : dictionnary containing the old and new value of category selected.
    current_entity (string) : String of the current working entity.
    df (dataframe) : Dataframe containing the annotated words, their occurrences, places and related entity.

    Return :
    (dataframe) : Return a dataframe containing the annotations belonging to the selected category. 
    """
    current_category = "category?"
    if change!= "None" : 
        if change['new'] == "category?":
            current_category = change['new']
        else : 
            current_category="["+change['new'] +"]"
    df1 = df.loc[(df['entity'] == current_entity) & (df['category'] == current_category)]
    df1 = df1.filter(["text","occurrences"])
    height = min(250,len(df1)*25+30)
    df1 = DataGrid(df1,column_widths={"text":800,"occurrences":100},layout={"height":f"{height}px","width":"990px"},base_row_size=25)
    return df1
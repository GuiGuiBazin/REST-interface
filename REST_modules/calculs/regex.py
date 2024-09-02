#from F.text.fr import pluralize
from unidecode import unidecode
import re

def generate_plural_form(word):
    """
    Return the word and its plural form.
    
    Parameters : 
    word (string) : a word of interest
    
    Return :
    plural_form (string) : the word and its plural form
    """
    plural_word = pluralize(word)
    plural_form = "("+word+"|"+plural_word+")"
    return plural_form

def is_parenthese_diff(regex):
    """
    Check if there is a different number of open and close brackets in the string.
    
    Parameters : 
    regex (str) : string containing a regex. 
    
    Return :
    (boolean) : return true if the number of open and close brackets are equal. 
    """
    i1=0
    i2=0
    for char in regex : 
        if char=="(":i1+=1
        elif char==")":i2+=1
    if not (i1-i2==0):
        return True
    else :
        return False

def generate_regex(words):
    """
    Create a regex from the words in input.
    
    Parameters : 
    words (list) : list containing the different words to transform into a regex. 
    
    Return :
    (regex) : the created regex.
    spacing_regex (regex) : return the longest possible spacing if the regex contains a spacing (used in the recommandations)
    len_constant_parts (int) : return the constant length contained in the spacing regex
    """
    regex = "("
    connectors = ["+", "+?"]
    inOr = False
    lastIsWord = False
    len_constant_parts = 0
    list_spacing_regex = []
    
    for word in words:
        word = word.lower()
        spacing_word,len_spacing_word = generate_spacing_word(word,spacing=None)
        if spacing_word :
            max_spacing_word,max_word_len = generate_spacing_word(word, spacing=500)
            list_spacing_regex.append([max_spacing_word,max_word_len,word])
            if lastIsWord:
                regex += "|"
            regex += spacing_word
            lastIsWord = True

        elif word in connectors : 
            # a - closing
            regex += ")"
            if inOr:
                regex += "?"
                inOr = False 
            # b - creating    
            if word == "+": 
                regex += "\s"  
            elif word == "+?": 
                regex += "\s?"
                inOr = True
            regex += "("
            lastIsWord = False
                
        else:
            if lastIsWord:
                regex += "|"
            regex += word
            lastIsWord = True
                
    if inOr:
        regex += ')?'     
    if is_parenthese_diff(regex):
        regex += ")"
          
    regex = r"\b" + "(" + regex + ")" + r"\b"
    regex = re.compile(regex, flags = re.UNICODE | re.DOTALL)
    return regex,list_spacing_regex
    
def generate_spacing_word(word,spacing):
    """
    Check if there is a different number of open and close brackets in the string.
    
    Parameters : 
    word (str) : string containing a word. 
    spacing (int) : length of the desired spacing
    
    Return :
    spacing_word (string) : string containing a regex with the desired spacing between the words. 
    len_spacing_word (int) : eturn the constant length contained in the spacing regex.
    """
    pattern = r'(\S+)\s*\.\.\.\s*(\d+)\s*(\S+)'
    match = re.search(pattern,word)
    
    if match:
        word1=match.group(1)
        word2=match.group(3)
        words1=word1
        words2=word2
        x = match.group(2)
        if spacing :
            x = spacing
        
        spacing_word = "("+words1+")"+"((?!"+words1+f")[^.?!]){{0,{x}}}?"+"("+words2+")"
        len_spacing_word = len(word1)+len(word2)
        return spacing_word,len_spacing_word
    
    else :
        return None,None
    
def calculate_list_spacing_regex(entity,ent_cat):
    """
    Calculate the list of the spacing regex.
    
    Parameters : 
    ent_cat (dict) : Sictionnay of all the entities paired with their categories.
    entity (string) : String of the current working entity.
    
    Return :
    (list) : List of the spacing regex. 
    """
    list_spacing = []
    for cat in ent_cat[entity] :
        if cat != "category?":
            pattern, list_spacing_pattern = generate_regex(eval(cat))    
            list_spacing.extend(list_spacing_pattern)
    return list_spacing


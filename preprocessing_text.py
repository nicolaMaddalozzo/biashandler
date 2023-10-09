"""Preprocessing of the orginal texts and tokens.

The functions contained in this file permit the preprocessing of the original
texts (.txt) and tokens. 

"""
import os
import re
import nltk
import math
from translate import Translator
import json
import numpy as np
from collections import Counter
nltk.download('stopwords')
from nltk.corpus import stopwords

def read_text(folder_path):
    """Loaing of .txt files contained in a folder path.

    This function opens and reads .txt files.


    Parameters
    ----------
    folder_path : str
        folder path with the text files.
 
    Returns
    -------
    One list of lists
        list of lists where each list contains the sentence.

    Raises
    ------
    TypeError
        If d is the wrong form (see message error for details).
    """
    txt_files = os.listdir(folder_path)
    sentences = []
    for txt in txt_files:
        if txt.endswith(".txt"):
            percorso_completo = os.path.join(folder_path, txt)
            with open(percorso_completo, "r") as f:
                contenuto = f.read().splitlines()
                sentences.append(contenuto)
        else :
            raise TypeError("There are files that are not .txt in the folder")
    sentences = [element for nestedlist in sentences for element in nestedlist]
    return sentences

def remove_dash(list_of_lists):
    """Removing of dashes from tokens.

    This function removes, from each token, the dash (if present).


    Parameters
    ----------
    list_of_lists : list
        list of lists, where each list contains the tokens from each sentence.
 
    Returns
    -------
    One list of lists
        list of lists with tokens without dashes.

    """
    res = []
    for l in list_of_lists:
        new_l = [word.strip('-') for word in l]
        res.append(new_l)
    return res

def lower_case(sentences_tokens):
    """Converting to lowercase.

    Converts the tokens to lowercase.


    Parameters
    ----------
    sentences_tokens : list
        list of lists, where each list contains the tokens from each sentence.
 
    Returns
    -------
    One list of lists
        list of lists with lowercase tokens.

    """
    res = [[token.lower() for token in tokens] for tokens in sentences_tokens]
    return res

def is_in_par(sentence):    
    """Checking sentences with parentheses.

    Check if a sentence contanins parts in parentheses () and 
    if a sentence is empty ''.


    Parameters
    ----------
    sentence : str
        string that represents a sentence.
 
    Returns
    -------
    One boolean
        True if a sentence is empty o contains parentheses

    """
    return bool(re.search(r'\([^)]*\)', sentence) or sentence == "")

def remove_point(sentences_tokens) :
    """Removing of dot.

    Check if a token finishes with a dot (.) and removes it.


    Parameters
    ----------
    sentences_tokens : list
        list of lists. Each list contains tokens (each token is a str)
 
    Returns
    -------
    One list of lists
        Where each single list contains tokens without dots

    """
    for tokens in sentences_tokens:
        if tokens and (tokens[-1].endswith(".")):
            tokens[-1] = tokens[-1][:-1]
    return sentences_tokens

def get_stopwords():
    """Getting stopwords.

    It gets the stopwords in Spanish, and also some other stopwords 
    are added by hand.

 
    Returns
    -------
    One set
        set of stopwords

    """
    stopw = stopwords.words('spanish')
    stopw.append("-")
    stopw.append('ce')
    stopw.append('nº')
    stopw.append('y')
    stopw.append('si')
    stopw.append('%')
    stopw = set(stopw)
    stopw.difference_update(['él','ella','ellas','ellos','los','las'])
    return stopw

def get_en_dict(json_data, keys1, keys2) :
    """Removing of dot.

    Check if a token finishes with a dot (.) and removes it.


    Parameters
    ----------
    sentences_tokens : list
        list of lists. Each list contains tokens (each token is a str)
 
    Returns
    -------
    One list of lists
        Where each single list contains tokens without dots
    """    
    set1 = set(keys1)
    set2 = set(keys2)
    elem_com = list(set1.intersection(set2))
    elem_dif = list(set1.symmetric_difference(set2))
    keys = elem_com + elem_dif
    seeds_dict = {}
    for i in range(len(json_data)) :
        k = json_data[i]["Category"]
        if k in keys:
            v = json_data[i]["Seeds"]
            seeds_dict[k] = v
        else :
            continue
    return seeds_dict

translator = Translator(to_lang="es")
def translate_dict(en_dict):
    """Translating function from EN to ES.

    Given a dictionary with every value represented by a list of words in 
    english, this function translates the words in the list from english
    to spanish. The keys represent the concept associated to the words.  


    Parameters
    ----------
    en_dict : dict
        Dictionary. Each key represent a concept and the associated value
        is a list of english words, where each word is a str
 
    Returns
    -------
    One dictionary, es_dict
        Dictionary. Each key represent a concept and the associated value
        is a list of spanish words, where each word is a str 
    """ 
    # initialize spanish dictionary
    es_dict = {}
    for key, value in en_dict.items():
        # Identify placeholders
        placeholders = re.findall(r'{.*?}', value)
        
        # Remove placeholders from the original string
        for placeholder in placeholders:
            value = value.replace(placeholder, '')

        # Translate the string without placeholders
        if (translator.translate(value)) is not None :
            translated_value = translator.translate(value)
        else :
            continue    
        
        # Add the placeholders back to the translated string
        for placeholder in placeholders:
            translated_value += ' ' + placeholder
        
        # spanish dict
        es_dict[key] = str(translated_value).strip()
    for k, v in es_dict.items() :
        v = v.replace(" '", "'")
        v = v.replace("'", "")
        v = v[1:-1]
        v = [termine.strip() for termine in v.split(',')]
        es_dict[k] = v
    return es_dict

def invoc(d, we):
    """Removing of specific values from a dict.

    Given a dictionary, where the key is a concept and the value is a list of seeds,
    some of these seeds could not be contained in the word embedding model. So, these are
    removed. 


    Parameters
    ----------
    d : dictionary
        With list if seeds as values  
    
    we : responsibly.we.bias.BiasWordEmbedding
        reposnibly object that hands the bias using the vectors omega
 
    Returns
    -------
    one dict.
        This dict will have some seeds that are removed because are not present in 
        the word embedding model.     
    """ 
    for k,v in d.items():
        # v[0] represents a target group Xi
        for p in v[0]:
            # if p is not present in the we object, 
            # is removed from the group.
            if p not in we.model.index2word :
                v[0].remove(p)
        # v[1] represents atarget group Yi
        for p in v[1]:
            # if p is not present in the we object, 
            # is removed from the group.
            if p not in we.model.index2word :
                v[1].remove(p)
    return d

def l_of_ls(sent):
    """Transforming str in list of words.

    Given a list of lists, where every element is a str, the funcion transform 
    the str in a list of words separated by ','.


    Parameters
    ----------
    sent : str
        str (such as 'hola,yo,soy'). This str is transformed in 
        'hola','yo','soy'  
 
    Returns
    -------
    one list of words.
        list.     
    """ 
    if sent != []:
        return sent[0].split(",")


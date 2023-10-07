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
    res = []
    for l in list_of_lists:
        new_l = [word.strip('-') for word in l]
        res.append(new_l)
    return res

def lower_case(sentences_tokens):
    res = [[token.lower() for token in tokens] for tokens in sentences_tokens]
    return res

def is_in_par(sentence):
    return bool(re.search(r'\([^)]*\)', sentence) or sentence == "")

def remove_point(sentences_tokens) :
    for tokens in sentences_tokens:
        if tokens and (tokens[-1].endswith(".")):
            tokens[-1] = tokens[-1][:-1]
    return sentences_tokens

def get_stopwords():
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
        
        es_dict[key] = str(translated_value).strip()
    for k, v in es_dict.items() :
        v = v.replace(" '", "'")
        v = v.replace("'", "")
        v = v[1:-1]
        v = [termine.strip() for termine in v.split(',')]
        es_dict[k] = v
    return es_dict

def l_of_ls(sent):
    if sent != []:
        return sent[0].split(",")


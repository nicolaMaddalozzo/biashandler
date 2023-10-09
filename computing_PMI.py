"""Computation of the Pointwise Mutual Information (PMI) and mean frequency.

The functions contained in this file permit to compute the PMI and the mean 
frequency of a seeds group.

"""
from tqdm import tqdm
import math

def calc_fY_Z_fY_nZ(tokens, words_Y, words_Z):
    """Computing of the crucial quantities for PMI formula.

    This function computes the quantities fY_Z and fY_nZ needed 
    to calculate PMI. 


    Parameters
    ----------
    tokens : list
        python list of lists, where each list contains the sentence tokens.
    words_Y : list
        first python list, where each element is a str.
    words_Z : str
        second python list, where each element is a str.

    Returns
    -------
    Two integers
        fY_Z, fY_nZ (see PMI thesis section for details)

    Raises
    ------
    ValueError
        If you insert only a list of lists (tokens) of 1 element.

    """    
    fY_Z = 0
    fY_nZ = 0
    if len(tokens)==1 :
        raise ValueError("The len of dataset is too small")
    for sentence in tokens:
        contains_words_Y = any(word in sentence for word in words_Y)
        if contains_words_Y: 
            contains_words_Z = any(word in sentence for word in words_Z)
            if contains_words_Z :  
                fY_Z += 1
            else :
                fY_nZ += 1
    return fY_Z, fY_nZ


def calculate_bias_pmi(fA_C, fA_nC, fB_C, fB_nC):
    """Computing of PMI.

    This function computes the Pointwise Mutual Information. 


    Parameters
    ----------
    fA_C : int
        see thesis for details.
    fA_nC : int
        see thesis for details.
    fB_C : int
        see thesis for details.
    fB_nC : int
        see thesis for details.

    Returns
    -------
    One foat value
        PMI ratio

    """
    if fA_C == 0 or fB_C ==0:
        return False
    if fA_nC == 0 or fB_nC == 0: # For avoidung division by zero
        bias_pmi = math.log((fA_C / (fA_nC + fA_C)) / (fB_C / (fB_nC + fB_C)))
        
        return bias_pmi  
    bias_pmi = math.log((fA_C / fA_nC) / (fB_C / fB_nC))
    return bias_pmi



def list_pmi(seeds, A, B, tokens_l, d_freq) :
    """Computing of list with PMI.

    This function computes, for each X and with fixed A,B, the PMI. 


    Parameters
    ----------
    seeds : dict
        python dict, where each item contain key (concept) and 
        the associated seeds.
    A : list
        first python list, where each element is a str.
    B : list
        second python list, where each element is a str.
    tokens_l : list
        python list of lists, where each list contains the sentences tokens
        from the original texts.
    d_freq : collections.Counter
        object that contains, for each word, the frequency in the texts. 
    Returns
    -------
    One dict object
        dict where to each group od seeds (X) is associated the PMI

    """
    d = {}
    for k,X in tqdm(seeds.items()):
        # Number of times a word in X is present in the texts.
        count = 0 
        for word in X:
            # If the word (token) has frequency = 0, is removed.
            if d_freq[word]==0:
            
                X.remove(word)
                continue
            else :
                # update the count
                count += d_freq[word]
        # count==0, If True, it means that no seeds associated with the key are present.
        # So, the PMI is not computed for these seeds. 
        if count == 0 :
            d[k] = [False]
            continue
        fA_C, fA_nC = calc_fY_Z_fY_nZ(tokens_l, A, X)
        fB_C, fB_nC = calc_fY_Z_fY_nZ(tokens_l, B, X)
        bias_pmi_result = calculate_bias_pmi(fA_C, fA_nC, fB_C, fB_nC)
        d[k] = [bias_pmi_result]
    return d

def comp_freq(d, seeds, freq):
    """Computing of the mean frequency for each seeds group.

    This function computes, for each X, the mean frequency. This function
    has notthe return, but modifies the dictionary d adding the mean frequency
    in the value (list) for each key 


    Parameters
    ----------
    d : dict
        python dict, where each item contain key (concept) and,
        as value, a list of one elemtn (PMI).
    seeds : dict
        python dict, where each item contain key (concept) and 
        the associated seeds.
    freq : collections.Counter
        object that contains, for each word, the frequency in the texts. 
    
    Raises
    ------
    ValueError
        If d is the wrong form (see message error for details).

    """
    for k in list(d.keys()):
        if len(d[k]) != 1 :
            raise ValueError("The input dict for which computing the mean frequencies\
                   for each seeds group is in the wrong form. For each key\
                   the value is a list of one element: The PMI") 
                   
        count = 0
        fr = 0
        for w in seeds[k]:
            fr += freq[w]
            count += 1
        d[k].append(round(fr/count,0))

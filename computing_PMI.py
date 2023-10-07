from tqdm import tqdm
import math
def calc_fY_Z_fY_nZ(dataset, words_Y, words_Z):
    fY_Z = 0
    fY_nZ = 0
    if len(dataset)==1 :
        raise("The len of dataset is too small")
    for sentence in dataset:
        contains_words_Y = any(word in sentence for word in words_Y)
        if contains_words_Y: 
            contains_words_Z = any(word in sentence for word in words_Z)
            if contains_words_Z :  
                fY_Z += 1
            else :
                fY_nZ += 1
    return fY_Z, fY_nZ


def calculate_bias_pmi(fA_C, fA_nC, fB_C, fB_nC):
    if fA_C == 0 or fB_C ==0:
        return False
    if fA_nC == 0 or fB_nC == 0:
        bias_pmi = math.log((fA_C / (fA_nC + fA_C)) / (fB_C / (fB_nC + fB_C)))
        
        return bias_pmi  # To avoid division by zero
    bias_pmi = math.log((fA_C / fA_nC) / (fB_C / fB_nC))
    return bias_pmi

#i = 0

def list_pmi(seeds, A, B, tokens_l, d_freq) :
    d = {}
    for k,X in tqdm(seeds.items()):
        #freq = 0
        count = 0 
        for word in X:
            if d_freq[word]==0:
        
                X.remove(word)
                continue
            else :
                count += 1
        if count == 0 :
            d[k] = [False]
            continue
        fA_C, fA_nC = calc_fY_Z_fY_nZ(tokens_l, A, X)
        fB_C, fB_nC = calc_fY_Z_fY_nZ(tokens_l, B, X)
        bias_pmi_result = calculate_bias_pmi(fA_C, fA_nC, fB_C, fB_nC)
        d[k] = [bias_pmi_result]
    return d

def comp_freq(d, seeds, freq):
    for k in list(d.keys()):
        count = 0
        fr = 0
        for w in seeds[k]:
            fr += freq[w]
            count += 1
        d[k].append(round(fr/count,0))

import os
import pandas as pd
import numpy as np
import csv
import torch
import difflib
from time import sleep
import sys
from transformers import BertTokenizer
from transformers import BertForMaskedLM
import matplotlib.pyplot as plt
from torch.distributions import kl_divergence
from torch.distributions.normal import Normal
from tqdm import tqdm
import statsmodels.api as sm
import time

class BiasInfo:
    """Build an object that contains the spanish CrowS-Pairs dataset.

    Allows to compute the PLL given two types of sentences: stereotypical and 
    not stereotypical. It gives information about the robustness of a masked 
    model m.

    .. note::
        **Important:** ``BiasInfo`` allows the loading of CrowS-Pairs 
        represented as csv file and with the same structure specified
        in the thesis. A future work is to obtain a more generalize
        code for handling different types of file.

    Methods
    -------
    sent_to_df(col1, col2)
        Computes the PLL for stereotypical sentences saved in col1 and
        not stereotypical saved in col2 and returns a DataFrame
    kl_div(dis1, dis2) 
        Computes the Kullback-Leibler score given two distributions:
        dis1 and dis2
    agree(dist_or, dist_2, method, inf_limit, sup_limit)
        returns M_agree for robustness analysis, as explained in the thesis.
    Returns
    -------
    pandas.core.frame.DataFrame
        DataFrame object that contains the ster/no ster sentences with PLLs
    float
        KL Divergence score
    pandas.core.series.Series
        M_agree for each sentences pair
    

    """

    def __init__(self, path_csv, model, metric):
        """Initialize biasinfo object.

        Initialize ``BiasInfo`` with a specific metric for computing
        PLLs, model and path to CrowS-Pairs csv

        Parameters
        ----------
        path_csv : str
            path to CrowS-Pairs csv dataset 
        model : str
            name of the desired model for computing PLLs  
        metric : str
            name of the metric used for computing PLLs. It can be "cp" (variant of Salazar)
            or "sz" (Salazar).

        Raises
        ------
        ValueError
            If the name metric is wrong. only accepts "cp" and "sz" 
            If the inserted model is not 'beto'
        """
        self.crows_df = read_data(path_csv)
        self.lm_model = model
        if metric not in ["sz", "cp"]:
            raise ValueError("Wrong metric inserted.")
        self.metr = metric

    def sent_to_df(self, col1, col2):
        """Creation of dataframe with PLLs

        Create a DataFrame that contains the ster, noster sentences
        and the PLLs for both. Also the bias type is reported.

        Parameters
        ----------
        path_csv : str
            path to CrowS-Pairs csv dataset. 
        model : str
            name of the desired model for computing PLLs.  
        metric : str
            name of the metric used for computing PLLs.

        Returns
        ------
        pandas.core.frame.DataFrame
        DataFrame object that contains the ster/no ster sentences with PLLs

        Raises
        ------
        ValueError
            If the inserted model is not 'beto'
        """
        
        print("Model:", self.lm_model)

        if self.lm_model == "beto":
            tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')
            model = BertForMaskedLM.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')
            uncased = True
        else :
            raise ValueError("In this version of the code, only 'beto' model is accepted")

        model.eval()
        if torch.cuda.is_available():
            model.to('cuda')

        mask_token = tokenizer.mask_token
        log_softmax = torch.nn.LogSoftmax(dim=0)
        vocab = tokenizer.get_vocab()

        lm = {"model": model,
          "tokenizer": tokenizer,
          "mask_token": mask_token,
          "log_softmax": log_softmax,
          "uncased": uncased
             }

        # score each sentence. 
        # each row in the dataframe has ster sent, noster sent, the score for both and bias type.
        df_score = pd.DataFrame(columns=[col1, 
                                         col2, 
                                         col1+'_score', 
                                         col2+'_score',
                                         'bias_type'])

        N = self.crows_df.loc[:,col1].count()
        for i in tqdm(range(N)) :
             
            data = {
                col1: self.crows_df.iloc[i][col1],
                col2: self.crows_df.iloc[i][col2]
                       }

            score_s = mask_unigram(data, lm, self.metr, col1, col2)
            row = pd.DataFrame({col1: [self.crows_df.iloc[i][col1]],
                    col2: [self.crows_df.iloc[i][col2]],
                    col1+'_score' : [score_s['sent1_score']], 
                    col2+'_score' : [score_s['sent2_score']],
                    'bias_type' : [self.crows_df.iloc[i]['bias_type']]})
            df_score = pd.concat([df_score, row])
        return df_score
    
    def kl_div(self, dis1, dis2) :
        """Computing of KL divergence score.

        Computes the KL score given two distributions.

        Parameters
        ----------
        dis1 : pandas.core.series.Series
            Distribution of ster PLLs 
        dis2 : pandas.core.series.Series
            Distribution of noster PLLs  

        Returns
        ------
        float
            KL div score

        """
        pro_list = torch.tensor(dis1)
        anti_list = torch.tensor(dis2)
        pro_std, pro_mean = torch.std_mean(pro_list)
        anti_std, anti_mean = torch.std_mean(anti_list)
        pro_anti, anti_pro = my_kl_div(pro_mean, pro_std, anti_mean, anti_std)
        score = torch.max(pro_anti / (pro_anti + anti_pro), anti_pro / (pro_anti + anti_pro)).item()
        KLDivS = score * 100
        return KLDivS
    
    def agree(self, dist1, dist2, method, inf_limit, sup_limit) :
        """Computing of M agree. 

        Computes the M agree for each couple of sentences.

        Parameters
        ----------
        dist1 : pandas.core.frame.DataFrame
            Distribution of ster/no ster PLLs of original examples. 
        dist2 : pandas.core.frame.DataFrame
            Distribution of ster/no ster PLLs of random/paraphrased examples.
        method : str
            Method for calculating M agree (signs or range)  
        inf_limit : float 
            lower limit
        sup_limit : float 
            upper limit
        Returns
        ------
        pandas.core.series.Series
            M_agree for each sentences pair

        """
        if method not in ["sign", "my_agree"]:
            raise ValueError("Wrong method inserted.")
        M_diff_1 = dist1.loc[:, 'sent_more_score'] - dist1.loc[:, 'sent_less_score']
        M_diff_2 = dist2.iloc[:, 3] - dist2.iloc[:, 4]
        if method == 'sign' :
            M_bias_1 = (M_diff_1  > 0)            
            M_bias_2 = (M_diff_2 > 0)
            M_agree = (M_bias_1 == M_bias_2)
            return M_agree
        elif method == 'my_agree' :
            M_bias_1 = M_diff_1.apply(mbias, args=(inf_limit, sup_limit))
            M_bias_2 = M_diff_2.apply(mbias, args=(inf_limit, sup_limit))
            M_agree = (M_bias_1 == M_bias_2)
            return M_agree 

def read_data(input_file):
    """Loading csv file. 
            reads and convert csv file to dataframe.

        Parameters
        ----------
        input_file: str
            path to csv file.

        Returns
        ------
        pandas.core.frame.DataFrame

    """
    df_data = pd.read_csv(input_file, encoding='latin-1')
    return df_data


def get_log_prob_unigram(masked_token_ids, token_ids, mask_idx, lm, metr):
    """Computing of log probabilities for computing PLL
        Given a sequence of token ids, with one masked token, return the log probability of the masked token given the others.

        Parameters
        ----------
        masked_token_ids : torch.Tensor
            tensor of ids of the sentence tokens also with the single id of masked token
        token_ids : torch.Tensor
           tensor of ids of all sentence tokens 
        mask_idx : int
            index that represent the position of maked token in masked_token_ids
        lm : dict
            contains all the information about the model (tokenizer, activation function, etc.)
        metr : str
            name of the metric. The metric decides how to mask the sentence (see thesis for details)

        Returns
        ------
        torch.Tensor
            Tensor with all the log probabilities
    """
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    log_softmax = lm["log_softmax"]
    mask_token = lm["mask_token"]
    uncased = lm["uncased"]
    
    
    
    # get model hidden states
    if metr in ["cp","sz","ss"] :
        output = model(masked_token_ids)
    elif metr == "aul" :
        output = model(token_ids)
    hidden_states = output[0].squeeze(0)
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)

    # we only need log_prob for the MASK tokens
    assert masked_token_ids[0][mask_idx] == mask_id
    #print("Mask token that is going to be in input of the model:", mask_id)
    hs = hidden_states[mask_idx]
    target_id = token_ids[0][mask_idx]
    log_probs = log_softmax(hs)[target_id]

    return log_probs

def get_span(seq1, seq2):
    """Extracting spans 
        This function extracts shared spans between two sequences.
        
        Parameters
        ----------
        masked_token_ids : torch.Tensor
            tensor of ids of the sentence tokens also with the single id of masked token
        token_ids : torch.Tensor
           tensor of ids of all sentence tokens 
        mask_idx : int
            index that represent the position of maked token in masked_token_ids
        lm : dict
            contains all the information about the model (tokenizer, activation function, etc.)
        metr : str
            name of the metric. The metric decides how to mask the sentence (see thesis for details)
        
        Returns
        ------
        torch.Tensor
            Tensor with all the log probabilities (PLLs)
    """

    seq1 = [str(x) for x in seq1.tolist()]
    seq2 = [str(x) for x in seq2.tolist()]

    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    template1, template2 = [], []
    for op in matcher.get_opcodes():
        if op[0] == 'equal':
            template1 += [x for x in range(op[1], op[2], 1)]
            template2 += [x for x in range(op[3], op[4], 1)]

    return template1, template2


def mask_unigram(data, lm, metr, col1, col2, n=1):
    """Scoring of a sentence (PLL)
        Score each sentence by masking one word at a time.
        
        Parameters
        ----------
        data : dict
            dict with two keys, where each key represent the ster/noster sentence
            and the value is the sentence itself (str). 
        lm : dict
            contains all the information about the model (tokenizer, activation function, etc.). 
        metr : str
            name of the metric. 
        col1 : str
            Name of the column that contains the ster sentences.
        col2 : str
            Name of the column that contains the no ster sentences.
        
        Returns
        ------
        dict
            Where the key represent the ster/no ster sentence
            and the value is the PLL
    """
    # Loading model information
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    log_softmax = lm["log_softmax"]
    mask_token = lm["mask_token"]
    uncased = lm["uncased"]

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    sent1, sent2 = data[col1], data[col2]

    if uncased:
        sent1 = sent1.lower()
        sent2 = sent2.lower()

    # tokenize
    sent1_token_ids = tokenizer.encode(sent1, return_tensors='pt')
    sent2_token_ids = tokenizer.encode(sent2, return_tensors='pt')
  
    if metr=="cp":
    # get spans of non-changing tokens
        template1, template2 = get_span(sent1_token_ids[0], sent2_token_ids[0])
        assert len(template1) == len(template2)
        N = len(template1)
       
    elif metr=="sz":   
    # get all tokens
        N = len(sent1_token_ids[0].tolist())
    
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)
    sent1_log_probs = 0.
    sent2_log_probs = 0.

    
    if metr == "cp" :
        # skipping CLS and SEP tokens, they'll never be masked
        for i in range(1, N-1):
            sent1_masked_token_ids = sent1_token_ids.clone().detach()
            sent2_masked_token_ids = sent2_token_ids.clone().detach()

            sent1_masked_token_ids[0][template1[i]] = mask_id
            sent2_masked_token_ids[0][template2[i]] = mask_id


            score1 = get_log_prob_unigram(sent1_masked_token_ids, sent1_token_ids, template1[i], lm, metr)
            score2 = get_log_prob_unigram(sent2_masked_token_ids, sent2_token_ids, template2[i], lm, metr)

            sent1_log_probs += score1.item()
            sent2_log_probs += score2.item()

        score = {}
        # average over iterations
        score["sent1_score"] = sent1_log_probs
        score["sent2_score"] = sent2_log_probs

        return score
    
    elif metr == "sz" :
    # skipping CLS and SEP tokens, they'll never be masked
        count_1 = 0
        count_2 = 0
        for i in range(1, N-1):
        
            sent1_masked_token_ids = sent1_token_ids.clone().detach()
            sent1_masked_token_ids[0][i] = mask_id
            score1 = get_log_prob_unigram(sent1_masked_token_ids, sent1_token_ids, i, lm, metr)
            sent1_log_probs += score1.item()
            count_1 += 1
        N = len(sent2_token_ids[0].tolist())
        for i in range(1, N-1):
        
            sent2_masked_token_ids = sent2_token_ids.clone().detach()
            sent2_masked_token_ids[0][i] = mask_id
            score2 = get_log_prob_unigram(sent2_masked_token_ids, sent2_token_ids, i, lm, metr)
            sent2_log_probs += score2.item()
            count_2 += 1
        score = {}
        # average over iterations
        score["sent1_score"] = sent1_log_probs
        score["sent2_score"] = sent2_log_probs
        return score

def my_kl_div(pro_mean, pro_std, anti_mean, anti_std):
    """Computing KL div
        Computing KL div for KL score.
        
        Parameters
        ----------
        pro_mean : torch.Tensor
            with one torch.float64 element. Mean of the ster PLLs distribution.  
        pro_std : torch.Tensor
            with one torch.float64 element. Standard dev. of the ster PLLs distribution. 
        anti_mean : torch.Tensor
            with one torch.float64 element. Mean of the no ster PLLs distribution.
        anti_std : torch.Tensor
            with one torch.float64 element. Standard dev. of the no ster PLLs distribution.
        
        
        Returns
        ------
        Two torch.Tensor elements.
            Each tensor contains one torch.float64 element.
            Represent the value of the divergence. 
    """
    pro_dist = Normal(pro_mean, pro_std)
    anti_dist = Normal(anti_mean, anti_std)
    pro_anti = kl_divergence(pro_dist, anti_dist)
    anti_pro = kl_divergence(anti_dist, pro_dist)
    return pro_anti, anti_pro
    
def mbias(value, inf_limit, sup_limit):
    """Computing M_bias quantity (see thesis)
        Computes M_bias as described in the method proposed for analyzing the robustness of MLM
        
        Parameters
        ----------
        value : float
            M_diff, difference between PLL of ster sent. and no ster sent.  
        inf_limit : float
            inf limit of range (see thesis for details)
        sup_limit : float
            sup limit of range (see thesis for details)
        
        
        Returns
        ------
        int.
            -1 or 0 or 1. Value of M bias 
    """
    if value < inf_limit:
        return -1
    elif inf_limit <= value <= sup_limit:
        return 0
    else:
        return 1

def scores_to_df(df_n, df_p, df_r) :
    """Creation of a dataframe
        It creates a dataframe that put together all the scores from original examples, par. and random examples.
        
        Parameters
        ----------
        df_n : pandas.core.frame.DataFrame
            Dataframe with the sentences and associated PLL from original examples of CrowS-Pairs.  
        df_p : pandas.core.frame.DataFrame.
            Dataframe with the sentences and associated PLL from paraphrased examples of CrowS-Pairs.
        df_r : pandas.core.frame.DataFrame
            Dataframe with the sentences and associated PLL from random examples of CrowS-Pairs.  
        
        
        Returns
        ------
        pandas.core.frame.DataFrame
            With all the information of input dataframes together. 
    """
    scores_sent_more = df_n.loc[:,"sent_more_score"]
    scores_sent_less = df_n.loc[:,"sent_less_score"]
    scores_sent_more_par = df_p.loc[:,"sent_more_par_score"]
    scores_sent_less_par = df_p.loc[:,"sent_less_par_score"]
    scores_sent_more_ran = df_r.loc[:,"sent_more_ran_score"]
    scores_sent_less_ran = df_r.loc[:,"sent_less_ran_score"]
    data = {'sent_more_score': scores_sent_more, 'sent_less_score': scores_sent_less, 
        'sent_more_par_score': scores_sent_more_par, 'sent_less_par_score': scores_sent_less_par, 
        'sent_more_ran_score': scores_sent_more_ran, 'sent_less_ran_score': scores_sent_less_ran}
    return pd.DataFrame(data)

def info_props(df_scores, alfa):
    """Computing information about proportion prop 
        Computes proportion of generating ster sentences for original examples, paraphrased examples and random examples.
        
        Parameters
        ----------
        df_scores : pandas.core.frame.DataFrame
            Dataframe with PLLs of original examples, paraphrased examples and random examples.  
        alfa : float
            Fixed alfa level for testing. 
        
        
        Returns
        ------
        pandas.core.frame.DataFrame
            With all the information about proportions. 
    """
    prop = {}
    pr_d = []
    for i in range(0, 5, 2) :
        l = []
        n_ster = (df_scores.iloc[:,i] - df_scores.iloc[:,i+1]) > 0 
        prop[i] = [sum(n_ster), df_scores.iloc[:,i].count()]
        pr_d.append(round(sum(n_ster)/df_scores.iloc[:,i].count(),2))
    test = {}
    p_value = {}
    for value in prop.values() :
        p,f = test_pr(value[0], value[1], 0.5, alfa)
        test[len(test)] = f
        p_value[len(test)] = p
    data = {'prop': pr_d, 'isBias': list(test.values()), "p_value":list(p_value.values())}
    return pd.DataFrame(data, index=["or","par","ran"])

def test_pr(suc, n, prop_dic, alfa, alt = "two-sided") :
    """Proportion test 
        Computes proportion test.

        Parameters
        ----------
        suc : int
            Number of successes (number of times the model prefers the stereotype sentence).  
        n : int
            Sample size. 
        prop_dic : float
            Proportion declared in the null hypothesis (H0). 
        alfa : float
            Fixed level for testing.
        alt : str
            Type of test
        Returns
        ------
        float
            p value. 
    """
    prop_test = sm.stats.proportions_ztest(suc, n, prop_dic, alt)
    z_score, p_value = prop_test
    if p_value < alfa:
        return (round(p_value,5), True) #refuses H0
    else:
        return(round(p_value,5), False) #does not refuse H0

def isRobus(diff1, diff2) :
    """Proportion test 
        Computes proportion test.

        Parameters
        ----------
        suc : int
            Number of successes (number of times the model prefers the stereotype sentence).  
        n : int
            Sample size. 
        prop_dic : float
            Proportion declared in the null hypothesis (H0). 
        alfa : float
            Fixed level for testing.
        alt : str
            Type of test
        Returns
        ------
        float
            p value. 
    """
    flags_1 = diff1 > 0
    flags_2 = diff2 > 0
    n = sum(flags_1 == flags_2)
    p1,f1 = test_pr(n, len(flags_1), 0.8)
    n = sum((diff1 - diff2).between(-1,1))
    p2,f2 = test_pr(n, len(diff1), 0.9)
    test = {}
    test["sign"] = [p1,f1]
    test["range"] = [p2,f2]
    return test

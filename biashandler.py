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
import time

class BiasInfo:
    # Inizializzato con 0: path del csv; 1: MLM
    def __init__(self, *args):
        self.crows_df = read_data(args[0])
        self.lm_model = args[1]
        if args[2] not in ["sz", "cp", "ss", "aul"]:
            raise ValueError("Wrong metric inserted.")
        self.metr = args[2]
    def sent_to_df(self, col1, col2):
        """
        Compute the scores for two types of sentences that are in two columns: col1 and col2
        """
        print("Model:", self.lm_model)

        if self.lm_model == "beto":
            #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            #model = BertForMaskedLM.from_pretrained('bert-base-uncased')
            tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')#, do_lower_case=False)
            model = BertForMaskedLM.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')
            uncased = True
        elif self.lm_model == "roberta":
            tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
            model = RobertaForMaskedLM.from_pretrained('roberta-large')
            uncased = False
        elif self.lm_model == "albert":
            tokenizer = AlbertTokenizer.from_pretrained('albert-xxlarge-v2')
            model = AlbertForMaskedLM.from_pretrained('albert-xxlarge-v2')
            uncased = True

        model.eval()
        if torch.cuda.is_available():
            model.to('cuda')

        mask_token = tokenizer.mask_token
        log_softmax = torch.nn.LogSoftmax(dim=0)
        vocab = tokenizer.get_vocab()
        #with open(self.lm_model + ".vocab", "w") as f:
        #    f.write(json.dumps(vocab))

        lm = {"model": model,
          "tokenizer": tokenizer,
          "mask_token": mask_token,
          "log_softmax": log_softmax,
          "uncased": uncased
             }

        # score each sentence. 
        # each row in the dataframe has the sentid and score for pro and anti stereo.
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
            """
            print("##############################")
            print("Sentencess\n:", self.crows_df.iloc[i][col1], self.crows_df.iloc[i][col2])
            print("##############################")
            """
            score_s = mask_unigram(data, lm, self.metr, col1, col2)
            #print("score_s:", score_s)
            row = pd.DataFrame({col1: [self.crows_df.iloc[i][col1]],
                    col2: [self.crows_df.iloc[i][col2]],
                    col1+'_score' : [score_s['sent1_score']], 
                    col2+'_score' : [score_s['sent2_score']],
                    'bias_type' : [self.crows_df.iloc[i]['bias_type']]})
            df_score = pd.concat([df_score, row])
        return df_score    
    def kl_div(self, dis1, dis2) :
        pro_list = torch.tensor(dis1)
        anti_list = torch.tensor(dis2)
        pro_std, pro_mean = torch.std_mean(pro_list)
        anti_std, anti_mean = torch.std_mean(anti_list)
        pro_anti, anti_pro = my_kl_div(pro_mean, pro_std, anti_mean, anti_std)
        score = torch.max(pro_anti / (pro_anti + anti_pro), anti_pro / (pro_anti + anti_pro)).item()
        KLDivS = score * 100
        return KLDivS
    
    def robus(self, dist_or, dist_2, method, inf_limit, sup_limit) :
        if method not in ["sign", "my_agree"]:
            raise ValueError("Wrong method inserted.")
        M_diff_or = dist_or.loc[:, 'sent_more_score'] - dist_or.loc[:, 'sent_less_score']
        M_diff_2 = dist_2.iloc[:, 3] - dist_2.iloc[:, 4]
        if method == 'sign' :
            M_bias_or = (M_diff_or  > 0)            
            M_bias_2 = (M_diff_2 > 0)
            M_agree = (M_bias_or == M_bias_2)
            return M_agree
        elif method == 'my_agree' :
            M_bias_or = M_diff_or.apply(transform, args=(inf_limit, sup_limit))
            M_bias_par = M_diff_2.apply(transform, args=(inf_limit, sup_limit))
            M_agree = (M_bias_or == M_bias_par)
            return M_agree 

def read_data(input_file):
    df_data = pd.read_csv(input_file, encoding='latin-1')
    return df_data


def get_log_prob_unigram(masked_token_ids, token_ids, mask_idx, lm, metr):
    """
    Given a sequence of token ids, with one masked token, return the log probability of the masked token.
    """
    """
    print("##############################")
    print("check in get log prob")
    print("masked_token_ids:", masked_token_ids)
    print("token_ids:", token_ids)
    print("mask_idx:", mask_idx)
    print("##############################")
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
    """
    This function extract spans that are shared between two sequences.
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
    """
    Score each sentence by masking one word at a time.
    The score for a sentence is the sum of log probability of each word in
    the sentence.
    n = n-gram of token that is masked, if n > 1, we mask tokens with overlapping
    n-grams.
    """
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    log_softmax = lm["log_softmax"]
    mask_token = lm["mask_token"]
    uncased = lm["uncased"]

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    data
    sent1, sent2 = data[col1], data[col2]

    if uncased:
        sent1 = sent1.lower()
        sent2 = sent2.lower()

    # tokenize
    sent1_token_ids = tokenizer.encode(sent1, return_tensors='pt')
    sent2_token_ids = tokenizer.encode(sent2, return_tensors='pt')
    """
    print("##############################")
    print("tokens sent1:", sent1_token_ids)
    print("tokens sent2:", sent2_token_ids)
    print("##############################")
    """
    if metr=="cp":
    # get spans of non-changing tokens
        template1, template2 = get_span(sent1_token_ids[0], sent2_token_ids[0])
        assert len(template1) == len(template2)
        N = len(template1)
        """
        print("##############################")
        print("tokens U cp t1:", template1)
        print("tokens U cp t2:", template2)
        print("##############################")
        """
    elif metr in ["sz", "aul"]:   
        N = len(sent1_token_ids[0].tolist())
    
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)
    sent1_log_probs = 0.
    sent2_log_probs = 0.

    
    if metr in ["cp", "ss"] :
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
    
    elif metr in ["sz", "aul"] :
    # skipping CLS and SEP tokens, they'll never be masked
        count_1 = 0
        count_2 = 0
        for i in range(1, N-1):
        
            sent1_masked_token_ids = sent1_token_ids.clone().detach()
            sent1_masked_token_ids[0][i] = mask_id
            """
            print("##############################")
            print("check computation of log prod of sent1, inside mask unigram")
            print("sent1_masked_token_ids", sent1_masked_token_ids)
            print("sent1_token_ids", sent1_token_ids) 
            print("##############################")
            """
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
        if metr == "aul" :
            score["sent1_score"] = sent1_log_probs/count_1
            score["sent2_score"] = sent2_log_probs/count_2
        elif metr=="sz" :
            score["sent1_score"] = sent1_log_probs
            score["sent2_score"] = sent2_log_probs
        return score

def my_kl_div(pro_mean, pro_std, anti_mean, anti_std):
    pro_dist = Normal(pro_mean, pro_std)
    anti_dist = Normal(anti_mean, anti_std)
    pro_anti = kl_divergence(pro_dist, anti_dist)
    anti_pro = kl_divergence(anti_dist, pro_dist)
    return pro_anti, anti_pro
    
def transform(value, inf_limit, sup_limit):
    if value < inf_limit:
        return -1
    elif inf_limit <= value <= sup_limit:
        return 0
    else:
        return 1

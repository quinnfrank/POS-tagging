import numpy as np
import pandas as pd

from hmm import HMM
from blstm import BLSTM, POSDataset


class TaggingResults():
    """Class which computes and stores several accuracy metrics for a given
       model and dataset.  The model must be on the CPU, if it is a BiLSTM,
       and the dataset must be an iterable of iterable of (token, POS) paris.
       Must also pass in the `vocab` and `tagset` pd.Series used to train `model`.
       
       When initialized, creates a master DataFrame with the following schema:
       - sent_num: a sentence number, from 1 to len(dataset)
       - token: the actual token string, prior to any preprocessing
       - y_true: the index of the correct POS
       - y_predict: the index of the predicted POS
       - oov: Boolean; True if the token is in the vocab
              (if `model` is an HMM, this will convert to lowercase before checking)
       - ambig: Boolean; indicates this token has multiple correct tags that could be applied"""
    
    def __init__(self, model, dataset, vocab, tagset, embedder=None):
        
        # If model is a BiLSTM, need to use a POSDataset object
        # If model is an HMM, need to convert tokens to lowercase
        if type(model) is BLSTM:
            dataset = POSDataset(dataset, embedder, tagset)
            self.prep_token = lambda x: x
        elif type(model) is HMM:
            self.prep_token = str.lower
        else:
            raise ValueError("Model must be a BLSTM or HMM object")
            
        sent_list, y_true, y_predict = model.split_predict(dataset)
        results = {'sent_num': [],
                   'token': [],
                   'y_true': [],
                   'y_predict': [],
                   'oov': []}
            
        for i, (sent, true, predict) in enumerate(zip(sent_list, y_true, y_predict)):
            results['sent_num'].extend([i+1] * len(sent))
            results['y_true'].extend(true)
            results['y_predict'].extend(predict)
            
            # For each token, preprocess if necessary and check if it is in vocab
            for token in sent:
                results['token'].append(token)
                results['oov'].append(self.prep_token(token) not in vocab)
        
        self.res = pd.DataFrame(results)
        self.tagset = tagset
        
        # Group by token (wrt lowercase decision) and count # distinct y_true vals
        # Those with >= 2 are defined as "ambiguous" tokens, as multiple POS tags may apply
        res_copy = pd.concat([self.res.token.map(self.prep_token), self.res.y_true], axis=1)
        pos_counts = res_copy.groupby(['token', 'y_true']).size().reset_index() \
                             .groupby('token').size().sort_values(ascending=False)
        self.pos_counts = pos_counts  
        
        # Loop over each token, and check if it is ambiguous
        ambig_tokens = pos_counts[pos_counts > 1].index
        self.res['ambig'] = self.res.token.map(self.prep_token).isin(ambig_tokens)
        
    
    def get_accuracy(self):
        """Produces a printout of relevant accuracy metrics:
           - TOTAL ACCURACY (% of all tokens classified correctly),
           - IV ACCURACY (% of tokens in vocab classified correctly),
           - OOV ACCURACY (% of tokens not in vocab classified correctly),
           - AMBIGUOUS ACCURACY (% of tokens with multiple correct tags in dataset;
                if model is an HMM, this converts to lowercase first)"""
        
        total = np.mean(self.res.y_true == self.res.y_predict)
        iv = np.mean(self.res.loc[self.res.oov == False, 'y_true'] ==
                     self.res.loc[self.res.oov == False, 'y_predict'])
        oov = np.mean(self.res.loc[self.res.oov == True, 'y_true'] ==
                      self.res.loc[self.res.oov == True, 'y_predict'])
        ambig = np.mean(self.res.loc[self.res.ambig == True, 'y_true'] ==
                        self.res.loc[self.res.ambig == True, 'y_predict'])       
        
        self.accs = pd.Series({'total': total, 'in vocab': iv,
                               'out of vocab': oov, 'ambiguous tokens': ambig}).round(4)
        print(f"\nAccuracy\n--------\n{self.accs}")
        
    
    def get_sent(self, num):
        """Produces a formatted printout of the given sentence number,
           comparing the correct POS tags with the predicted POS tags."""
        
        assert num in self.res.sent_num.values, "Sentence index is out of range for this dataset"
        this_sentence = self.res[self.res.sent_num == num]
        
        star_if_true = lambda boolean: '*' if boolean else ''
        check_if_true = lambda boolean: '✓' if boolean else ''
        printout = pd.DataFrame({'true': self.tagset[this_sentence.y_true],
                                 'predict': self.tagset[this_sentence.y_predict],
                                 'correct?': (this_sentence.y_true == this_sentence.y_predict) \
                                             .map(check_if_true).values,
                                 'oov?': this_sentence.oov.map(star_if_true).values,
                                 'ambiguous?': this_sentence.ambig.map(star_if_true).values},
                                index = this_sentence.token,)
        print(printout)
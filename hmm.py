import numpy as np
import pandas as pd


### UTILITY FUNCTIONS

def separate_tagged_sents(tagged_sents):
    """Given a list of list of (token, TAG) pairs from Brown corpus,
       returns a parallel list of tokens and a list of tags."""
    
    # Combine the sentences together
    tagged_tokens = [pair for sent in tagged_sents for pair in sent]
    return tuple(map(list, zip(*tagged_tokens)))


def split_tagged_sents(tagged_sents):
    """Given an iterable of iterable of (token, POS) pairs,
       returns parallel iterable of iterables for tokens and POS.
       Keeps the underlying sentence structures."""
    
    untuple = lambda it: tuple(zip(*it))
    split_sents = tuple(map(untuple, tagged_sents))
    return untuple(split_sents)


def get_loc(index, entry, default):
    """Overloaded `get_loc` method from pd.Series.Index.
       If entry is in the `index`, returns the integer location.
       If not, returns the location of the default value (default must be in `index`)."""
    
    try:
        return index.get_loc(entry)
    except KeyError:
        return index.get_loc(default)

    
def index_tagged_sents(tagged_sents, tagset, vocab):
    """Given a list of list of (token, TAG) pairs from Brown corpus,
       returns a list of list of tokens and tags as indexes
       on the corresponding tagset and vocabulary."""
    
    tokens, tags = [], []
    
    for i, sentence in enumerate(tagged_sents):
        sub_tokens, sub_tags = tuple(map(list, zip(*sentence)))
        tags.append([get_loc(tagset, tag, "X") for tag in sub_tags])
        tokens.append([get_loc(vocab, token.lower(), "<UNK>") for token in sub_tokens])
        
    return tokens, tags


### COMPUTATION FUNCTIONS

def estimate_hmm(tagged_sents, tagset=None, vocab=None):
    """
    Estimates parameters of a Hidden Markov Model by MLE.  The vocabulary (size V)
    of words are the observations and the tags (size N) are the hidden states.
    Uses an OOV token and Laplace smoothing to avoid zero probabilities/likelihoods.
    
    ---Parameters---
    - tagged_sents: list of list of (token, TAG) pairs; each sublist represents one sentence
    - tagset: (N,) vector of unique tags to consider; inferred from data if None
    - vocab: (V,) vector of unique words to consider; inferred from data if None
    
    ---Returns---
    - A: (N, N) Markov transition matrix between states (rows sum to 1);
      entry i,j = Pr(state i -> state j)
    - B: (V+1, N) emission probability matrix; entry i,j = Pr(obs i | state j)
    - pi: (N,) initial distribution over the N states
    """
    
    tokens, tags = separate_tagged_sents(tagged_sents)
    assert len(tokens) == len(tags), "Observations and states must be same length."
    if tagset is None:
        tagset = sorted(set(tags))
    if vocab is None:
        vocab = sorted(set(map(str.lower, tokens)))
    vocab.append("<UNK>")   # add an "out of vocabulary" token
    
    # Begin with unnormalized count matrices
    A_counts = pd.DataFrame(0, dtype='int', index=tagset, columns=tagset)
    B_counts = pd.DataFrame(0, dtype='int', index=tagset, columns=vocab)
    pi_counts = pd.Series(0, dtype='int', index=tagset)
    
    # Loop over each (obs, state) pair, update corresponding counts
    for sent in tagged_sents:
        start_token, start_tag = sent[0]
        pi_counts[start_tag] += 1
        
        for i in range(len(sent)):
            this_token, this_tag = sent[i]
            B_counts.loc[this_tag, this_token.lower()] += 1
            
            # On the last token in the sentence, nothing follows, so don't update A
            if i != len(sent)-1:
                next_token, next_tag = sent[i+1]
                A_counts.loc[this_tag, next_tag] += 1
    
    # Apply add-k smoothing for everything (just add k to all counts, then marginalize)
    # Laplace is too extreme for this case, anecdotally
    smooth = 1e-4
    A = (A_counts + smooth) / np.sum(A_counts + smooth, axis=1)[:, None]
    B = (B_counts + smooth) / np.sum(B_counts + smooth, axis=1)[:, None]
    pi = (pi_counts + smooth) / np.sum(pi_counts + smooth)
    return A, B, pi


def viterbi(obs, pi, A, B):
    """
    Implements the Viterbi dynamic programming algorithm to decode
    (i.e., assign parts of speech) to a series of observations.
    
    ---Parameters---
    - obs: a (T,) list of list of observations; stored as ints which index the vocabulary;
      each sublist is a sentence
    - pi, A, B: the initial state distribution, transmission probability matrix,
      and emission probability matrix produced by `estimate_hmm`
    
    ---Returns---
    - states: a (T,) list of list of inferred states; stored as ints which index the tagset;
      each sublist is a sentence
    """
    
    all_states = []
    # Convert pi, A, and B to ndarrays and convert to log space to avoid underflow
    pi = np.log(pi.to_numpy())
    A = np.log(A.to_numpy())
    B = np.log(B.to_numpy())    
    
    for sentence_obs in obs:
        viterbi = np.zeros((len(pi), len(sentence_obs)))     # probability matrix (also log space)
        backpointer = np.zeros(viterbi.shape, dtype='int')   # int entries encoding path's last location
    
        # Treat the initial state separately
        viterbi[:, 0] = pi + B[:, sentence_obs[0]]
        backpointer[:, 0] = -1
        #print(f"\n{pd.Series(viterbi[:, 0].round(2), index=tagset).sort_values()}")
        
        # Compute the Viterbi probs / backpointers dynamically, based only on previous column
        for t in range(1, len(sentence_obs)):
            for s in range(len(pi)):
                # Figure out the optimal path to state 's' at observation 't'
                path_likes = viterbi[:, t-1] + A[:, s] + B[s, sentence_obs[t]]
                viterbi[s, t] = np.max(path_likes)
                backpointer[s, t] = np.argmax(path_likes)
            #print(f"\n{pd.Series(viterbi[:, t].round(2), index=tagset).sort_values()}")
        
        # Final path is the max of last column - follow path back to get all states
        states = [np.argmax(viterbi[:, -1])]
        for t in range(1, len(sentence_obs))[::-1]:
            states.append(backpointer[states[-1], t])
        states = states[::-1]
        all_states.append(states)
    
    return all_states


### WRAPPER CLASS

class HMM():
    """Class which encapsulates a Hidden Markov Model, calling the subroutines
       `estimate_hmm` and `viterbi` defined above to fit and predict datasets.
       
       Initialization parameters:
       - tagset = iterable of unique tags, in some order
       - vocab = iterable of unique tokens, in some order
         (if either of these is not provided, they will be inferred after .fit() is called)
       - default_tag = how to encode unknown token (X by default)
       - default_token = how to encode OOV token (<UNK> by default)"""
    
    def __init__(self, tagset=None, vocab=None, default_tag="X", default_token="<UNK>"):
        self.tagset = tagset
        self.vocab = vocab
        self.default_tag = default_tag
        self.default_token = default_token
        
    def fit(self, tagged_sents):
        """Given an iterable of iterable of (token, POS) pairs,
           infers parameters of HMM, storing as attributes of this object."""
        
        A, B, pi = estimate_hmm(tagged_sents, self.tagset, self.vocab)
        # Update the tagset and vocab
        self.tagset = B.index
        self.vocab = B.columns
        self.A = A
        self.B = B
        self.pi = pi
    
    def split_predict(self, tagged_sents):
        """Given an iterable of iterable of (token, POS) pairs,
           splits into [...[...token]], [...[...POS]] iterables and
           returns parallel true POS indices, predicted POS indices"""
        
        # Cannot generate until the HMM parameters are inferred
        assert hasattr(self, "pi"), "Must call .fit() before .split_predict()"
        
        sent_list, tag_lists = split_tagged_sents(tagged_sents)
        
        # Compute the token indices, then run through Viterbi to predict
        X = [[get_loc(self.vocab, token.lower(), self.default_token) for token in sent]
             for sent in sent_list]
        y_predict = viterbi(X, self.pi, self.A, self.B)
        # Compute the true tag indices, to compare with Viterbi result
        y_true = [[get_loc(self.tagset, tag, self.default_tag) for tag in tag_list]
                  for tag_list in tag_lists]
        
        return y_true, y_predict
    
    def generate(self, size, max_len=100):
        """Generates an artificial labeled dataset from fitted HMM parameters.
           Produces `size` number of sentences, stopping either when end punctuation
           [.!?] is reached or the maximum sentence length is reached.
           Returns an iterable of iterable of (token, POS) pairs."""
        
        # Cannot generate until the HMM parameters are inferred
        assert hasattr(self, "pi"), "Must call .fit() before .generate()"
        
        tagged_sents = []
        for i in range(size):
            sent = []
            stop_condition = False
            
            # Draw the starting state from initial distribution
            pos = np.random.choice(self.tagset, p=self.pi)
            # For each POS: (1) draw token from emission probability dist P(token | POS)
            #               (2) wrap POS and token together, add to current sentence
            #               (3) if not stop condition, draw a new POS from
            #                   transition probability dist P(POS_t+1 | POS_t)
            while not stop_condition:
                token = np.random.choice(self.vocab, p=self.B.loc[pos, :])
                sent.append((token, pos))
                if token in {'.', '!', '?'} or len(sent) >= max_len:
                    stop_condition = True
                else:
                    pos = np.random.choice(self.tagset, p=self.A.loc[pos, :])
            
            tagged_sents.append(sent)
        return tagged_sents
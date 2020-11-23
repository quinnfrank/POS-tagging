import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from gensim.models import Word2Vec, FastText


### UTILITY FUNCTIONS

def get_loc(index, entry, default=None):
    """Overloaded `get_loc` method from pd.Series.Index.
       If entry is in the `index`, returns the integer location.
       If not, returns the location of the default value (default must be in `index`)."""
    
    try:
        return index.get_loc(entry)
    except KeyError:
        return index.get_loc(default)
    

def split_tagged_sents(tagged_sents):
    """Given an iterable of iterable of (token, POS) pairs,
       returns parallel iterable of iterables for tokens and POS.
       Keeps the underlying sentence structures."""
    
    untuple = lambda it: tuple(zip(*it))
    split_sents = tuple(map(untuple, tagged_sents))
    return untuple(split_sents)


def train_fasttext(untagged_sents, *args, **kwargs):
    """Trains a FastText word embedder to convert words to dense vector form.
       Input must be an iterable of iterable of tokens (UNTAGGED)."""
    
    embedder = FastText(untagged_train, *args, **kwargs)
    return embedder


### DATA LOADING AND ITERATION CLASSES

class POSDataset(torch.utils.data.Dataset):
    """Custom Dataset class which digests an NLTK tagged dataset
       and converts sentences to torch tensors dynamically.
       
       > dataset = POSDataset(train, embedder, tagset)
       dataset[batch_idx] returns a tuple of the form:
       - tokens = float tensor of size (seq_len, batch_size, input_size)
         NOTE: if sequences are different lengths, the sequences which are too short
         will have padding along dim=2; dataset is sorted by sentence length to minimize this
       - tags = int tensor of size (seq_len, batch_size, 1)
       - mask = boolean tensor of size (seq_len * batch_size) indicating non-padded values
       
       Initialization parameters:
       - sentences = an iterable of iterable (token, POS) pairs
       - embedder = a TRAINED gensim embedder network like Word2Vec or FastText
       - tagset = an iterable of tags, in some fixed order"""
    
    def __init__(self, sentences, embedder, tagset):
        self.sentences = sentences
        self.embedder = embedder
        self.tagset = tagset
    
    def shuffle(self):
        # Shuffles around sentences, while keeping similar-length sentences together
        self.sentences = sorted(self.sentences,
                                key=lambda sent: (len(sent), np.random.random()))
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, index):
        these_sentences = self.sentences[index]
        # If an integer is passed in, this could break, need to wrap in another list
        if type(index) is int:
            print(these_sentences)
            these_sentences = [these_sentences]
            
        sent_list, tag_lists = split_tagged_sents(these_sentences)
        
        # Convert both the words and tags to numerical form
        embedded_list = [torch.tensor(embedder.wv[sent]) for sent in sent_list]
        label_list = [torch.tensor([get_loc(self.tagset, tag, "X") for tag in tag_list])
                      for tag_list in tag_lists]
        
        # If sequence lengths are difference,
        # pad the end of the embeddings and labels for the shorter sentences
        null_pad = get_loc(self.tagset, "X", "-NONE-")
        X_pad = pad_sequence(embedded_list)
        y_pad = pad_sequence(label_list, padding_value=-1)
        
        # Find where the null-padded values are, so the loss function can ignore them
        mask = y_pad != -1
        y_pad[y_pad == -1] = null_pad
        
        return X_pad, y_pad, mask
    

class POSDataLoader():
    """Custom DataLoader-like object that works with POSDataset."""
    
    def __init__(self, dataset, batch_size, shuffle=True):
        assert type(dataset) is POSDataset, "Only supports POSDataset objects"
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __iter__(self):
        if self.shuffle: self.dataset.shuffle()
        return POSDataIterator(self.dataset, self.batch_size)
    

class POSDataIterator():
    """Iterator object which returns (X, y) pairs from a POSDataset of specified
       batch size until it reaches the end."""
    
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        # Keep track of the pointer's location on the batch index
        self._idx = 0
        
    def __next__(self):
        # Returns the next (X, y) pair
        if self._idx >= len(self.dataset):
            raise StopIteration
        else:
            start_idx = self._idx
            end_idx = start_idx + self.batch_size
            self._idx = end_idx
            return self.dataset[start_idx:end_idx]
        

### ARCHITECTURE

class BLSTM(nn.Module):
    """Bidirectional, stacked LSTM RNN for sequence to sequence translation.
       Given a variable-length time-indexed sequence of word embeddings, produces a
       probability distribution over a finite tagset, for supervised learning.
       
       input shape:  (seq_len, batch_size, input_size),
       output shape: (seq_len, batch_size, output_size), softmax along dim=2"""
    
    def __init__(self, embed_size, hidden_size, num_layers, output_size):
        super(BLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=True,
                            dropout=0.25)
        self.classify = nn.Sequential(nn.Linear(2 * hidden_size, output_size),
                                      nn.Dropout(0.25))
        
    def forward(self, X):
        output, (h, c) = self.lstm(X)
        return self.classify(output)
    
    def predict(self, X):
        """Once model is TRAINED, call to predict y labels on input.
           
           input shape: (seq_len, batch_size, input_size),
           output shape: (seq_len, batch_size)"""
        
        # WARNING: Due to post-padding, it could be very memory-intensive to run
        # a large batch of embedded vectors; recommended to use .split_predict()
        return torch.argmax(self.forward(X), dim=2)
    
    def split_predict(self, dataset, batch_size=64):
        """Given a POSDataset object, returns parallel true POS indices,
           predicted POS indices (list of tensors, without padding).
           Made to be compatible with HMM .split_predict() method"""
        
        y_true, y_predict = [], []
        loader = POSDataLoader(dataset, batch_size, shuffle=False)
        for X, y, mask in loader:
            yhat = self.predict(X)
            
            for i in range(X.shape[1]):
                y_true.append(y[:, i][mask[:, i]].tolist())
                y_predict.append(yhat[:, i][mask[:, i]].tolist())
                
        return y_true, y_predict
    

### TRAINING, EVALUATION, AND PLOTTING RESULTS

def train_BLSTM(model, train_set, test_set,
                num_epochs=100, batch_size=16,
                use_cuda=True, train_summ=None, print_every=1):
    """Trains a BLSTM model for some fixed number of epochs.
       Loads data in batches, shuffling after each full epoch
       (if sentence lengths are difference, post-padding is added and ignored in loss calculation).
       Prints logs of train/test accuracy after each epoch and returns
       plottable dictionary of training results.
       
       Parameters
       - num_epochs: the number of epochs over train_set
       - batch_size: how many sentences from train_set to run before updating weights
                     (if sentence lengths are uneven, padding occurs at the end)
       - train_summ: summary dictionary produced by previous function call;
                     use to continue training while maintaining previous history
       - print_every: how often loss function statistics should be printed, in epochs
    """
    
    # Move to GPU if possible; batches get moved as needed to save on memory
    device = "cuda:0" if use_cuda and torch.cuda.is_available() else "cpu"
    
    # model(X) and y come out as 3d tensors
    # Need to combine seq_len and batch_size along dim=0 to calculate loss
    flatten = nn.Flatten(0, 1)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
    
    # Initialize a dictionary summarizing training, if one not provided
    if train_summ is None:
        train_summ = {'loss': [],        # one per batch
                      'train_acc': [],   # one per epoch
                      'test_acc': [],    # one per epoch
                      'num_epochs': num_epochs}
    else: train_summ['num_epochs'] += num_epochs
    
    train_loader = POSDataLoader(train_set, batch_size, shuffle=True)
    test_loader = POSDataLoader(test_set, batch_size)
    
    for epoch in range(num_epochs):
        # Only let the model be in training mode in this block
        model.train()
        
        for X, y, mask in train_loader:
            X = X.clone().to(device)
            y = y.clone().to(device)
            
            optimizer.zero_grad()
            
            # Apply a Boolean mask to ignore the post-padding values
            # This also flattens to (N, #classes) format for CrossEntropyLoss
            #print(sum(mask))
            loss = criterion(model(X)[mask, :], y[mask])
            loss.backward()
            optimizer.step()
            train_summ['loss'].append(loss.detach().cpu().item())
        
        # Evaluate how the model is performing on the entire training/testing sets
        train_acc, test_acc = eval_BLSTM(model, train_loader, test_loader, device)  
        train_summ['train_acc'].append(train_acc)
        train_summ['test_acc'].append(test_acc)
        
        if (epoch+1) % print_every == 0:
            print(f"- EPOCH {epoch+1}:" + 
                  f"\n  train loss = {train_summ['loss'][-1]}" +
                  f"\n  accuracy   = " +
                  f"{round(train_acc, 4)} (train) / {round(test_acc, 4)} (test)" +
                  "\n----------------------------")
    
    assert not model.training
    return train_summ


def eval_BLSTM(model, train_loader, test_loader, device):
    """Evaluates model on provided training and testing set.
       Due to memory limits, must take in data loaders and predict in batches.
       Returns float values (train_acc, test_acc)"""
    
    # Set the model to evaluation mode to ignore non-deterministic effects like dropout
    model.eval()
    train_checks, test_checks = [], []
    
    # Train predict
    for X, y, mask in train_loader:
        X = X.clone().to(device)
        y = y.clone().to(device)
        y_predict = model.predict(X)
        train_checks.append(y[mask] == y_predict[mask])
    train_checks = torch.cat(train_checks).float()
    train_acc = torch.mean(train_checks).cpu().item()
    
    # Test predict
    for X, y, mask in test_loader:
        X = X.clone().to(device)
        y = y.clone().to(device)
        y_predict = model.predict(X)
        test_checks.append(y[mask] == y_predict[mask])
    test_checks = torch.cat(test_checks).float()
    test_acc = torch.mean(test_checks).cpu().item()
    
    return train_acc, test_acc


def plot_train_summ(train_summ):
    """Produces plots that summarize and compare training of a model.
       Input should be a dictionary of the form:
       {loss: [...],
        train_acc: [...], test_acc: [...],
        num_epochs: ### } (dict returned by train_*) }"""
     
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = np.linspace(0, train_summ['num_epochs'], len(train_summ['loss']))
    ax.plot(epochs, train_summ['loss'], label="train")
    ax.legend()
    ax.set_title(f"Training Curve for BLSTM")
    ax.set_xlabel("epoch number")
    ax.set_ylabel("cross-entropy loss")
    plt.show()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(train_summ['num_epochs'])
    for dataset in ["train", "test"]:
        ax.plot(epochs, train_summ[f'{dataset}_acc'], label=dataset)
    ax.legend()
    ax.set_title(f"Classification Accuracy for BLSTM")
    ax.set_xlabel("epoch number")
    ax.set_ylabel("accuracy")
    ax.legend()
    plt.show()
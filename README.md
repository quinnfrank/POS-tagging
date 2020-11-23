# POS-tagging
Generative and discriminative modeling approaches to part of speech tagging.

## Contents

* `hmm.py` contains my Hidden Markov Model implementation
* `blstm.py` contains my Bidirectional LSTM implementation in PyTorch, plus some other utility functions
* `tagging_results.py` contains functions for computing accuracy of models from either of the above

* `Analysis.ipynb` is a Jupyter Notebook which walks through the process of loading the training data, fitting both the HMM and BLSTM model, and evaluating their accuracy.  It also creates an artificial dataset and prepares the pre-downloaded data to test these models.

* The JSON files in the `data/` directory contain a collection of Trump tweets from August-October 2020, manually downloaded from the <a href="https://www.thetrumparchive.com/">Trump Twitter Archive</a>.  For reference, I only used entries with were *not* retweets and which were sent from an iPhone.

## Dependencies

The following Python packages are required to run the notebook and all modules:

* `numpy`
* `pandas`
* `matplotlib`
* `torch`
* `gensim`
* `nltk` with "tagset", "brown", "punkt", and "averaged_perceptron_tagger" installed (the notebook does this automatically)

# NN4ABSA
Neural Network based models for Aspect-Based Sentiment Analysis

# Model 1
* Word embeddings: [stanford GloVe](https://nlp.stanford.edu/projects/glove/)
* Ctx Feat Extractor: CNN + Multi-Channel
* Target Feat Extractor: Weighted sum of word vectors making up the target phrase

# Performance (accuracy & macro-F1)
| | 14semval-restaurant | 14semeval-laptop | Twitter |
|---|---|---|---|
|ATAE-LSTM [1] | 77.2/- | 68.7/ | - |
|MemNet [2] | 78.16/65.83 | 70.33/64.09 | 68.50/66.91 |
|IAN [3] | 78.6/- | 72.1/- | - |
|RAM [4] | 80.23/70.80 | 74.49/71.35 | 69.36/67.30 |
|Model 1 | 79.43/69.49 | 74.65/69.27 | 70.38/68.00 |

# References
1. Attention-based LSTM for Aspect-level Sentiment Classification. EMNLP 2016
2. Aspect Level Sentiment Classification with Deep Memory Network. EMNLP 2016
3. Interactive Attention Networks for Aspect-Level Sentiment Classification. IJCAI 2017
4. Recurrent Attention Network on Memory for Aspect Sentiment Analysis. EMNLP 2017



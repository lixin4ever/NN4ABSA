# NN4ABSA
Neural Network based models for Aspect-Based Sentiment Analysis

# Model 1
* Word embeddings: [stanford GloVe](https://nlp.stanford.edu/projects/glove/)
* Ctx Feat Extractor: CNN + Multi-Channel
* Target Feat Extractor: Weighted sum of word vectors making up the target phrase

# Performance (accuracy & macro-F1)
| | 14semval-restaurant | 14semeval-laptop | Twitter |
|---|---|---|---|
|ATAE-LSTM [1] | x | x | x |
|IAN [2] | x | x | x |
|RAM [3] | x | x | x |
|Model 1 | 73.71/68.00 | 78.89/67.03 | 70.38/68.00 |

# References
1. Attention-based LSTM for Aspect-level Sentiment Classification. EMNLP 2016
2. Interactive Attention Networks for Aspect-Level Sentiment Classification. IJCAI 2017
3. Recurrent Attention Network on Memory for Aspect Sentiment Analysis. EMNLP 2017



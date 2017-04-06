# Quora
Python code to solve Quora questions pairs duplication on Kaggle

Tensorflow Model uses lstm on each question then the resulting two vectors are given to a FC layers to produce the final classification.

Keras Model uses TimeDistributed to apply a dense layer for each vector then select the sum or max and give the combined results to FC layers to produce the final classification.

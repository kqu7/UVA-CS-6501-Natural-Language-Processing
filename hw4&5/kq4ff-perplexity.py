import math
import numpy as np
# predictions are the predictions computed by the model for all sentences in the corpus
def perplexity(predictions):
    num_sentence = len(predictions)
    predictions = [item for sublist in predictions for item in sublist]
    predictions = np.asarray(predictions)
    denominator = len(predictions) + num_sentence
    numerator = np.sum(np.log(predictions))
    return math.exp(-numerator/denominator)
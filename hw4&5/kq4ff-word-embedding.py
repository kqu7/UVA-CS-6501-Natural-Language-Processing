# Load packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Load data
trn_texts = open("trn-reviews.txt").read().strip().split("\n")
trn_labels = open("trn-labels.txt").read().strip().split("\n")
print("Training data ...")
print("%d, %d" % (len(trn_texts), len(trn_labels)))

dev_texts = open("dev-reviews.txt").read().strip().split("\n")
dev_labels = open("dev-labels.txt").read().strip().split("\n")
print("Development data ...")
print("%d, %d" % (len(dev_texts), len(dev_labels)))

# Convert training text and dev text into lower-case form
trn_texts = [x.lower() for x in trn_texts]
dev_texts = [x.lower() for x in dev_texts]

# Tokenize
from nltk.tokenize import WordPunctTokenizer
trn_token = [WordPunctTokenizer().tokenize(x) for x in trn_texts]
dev_token = [WordPunctTokenizer().tokenize(x) for x in dev_texts]

# Load word embedding
import numpy as np
glove = {}
with open('glove.6B.50d.txt', 'rb') as f:
    lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i].decode().split()
        glove[line[0]] = np.array(line[1:]).astype(np.float)

# Get the word embedding 
word_dict = {}
gk = glove.keys()
for t in trn_token: 
    for tt in t:
        if tt in gk: 
            word_dict[tt] = glove[tt]
        else: 
            word_dict[tt] = glove["unk"]
for v in dev_token:
    for vv in v: 
        if vv in gk:
            word_dict[vv] = glove[vv]
        else:
            word_dict[vv] = glove["unk"]

# Get the sentence embedding
trn_sent_embed = np.zeros((40000, 50))
for i, t in enumerate(trn_token):
    if len(t) == 0:
        continue
    for tt in t:
        trn_sent_embed[i] += word_dict[tt]
    trn_sent_embed[i] /= len(t)
dev_sent_embed = np.zeros((5000, 50))
for i, v in enumerate(dev_token):
    if len(v) == 0:
        continue
    for vv in v:
        dev_sent_embed[i] += word_dict[vv]
    dev_sent_embed[i] /= len(v)

classifier = LogisticRegression()
classifier.fit(trn_sent_embed, trn_labels)
print("Training accuracy = %f" % classifier.score(trn_sent_embed, trn_labels))
print("Dev accuracy = %f" % classifier.score(dev_sent_embed, dev_labels))

# Get new features
vectorizer = CountVectorizer(lowercase=True, min_df=7e-5, ngram_range=(1, 2), max_features=10000)
trn_data = vectorizer.fit_transform(trn_texts).toarray()
dev_data = vectorizer.transform(dev_texts).toarray()
trn_new_features = np.concatenate((trn_sent_embed, trn_data), axis = 1)
dev_new_features = np.concatenate((dev_sent_embed, dev_data), axis = 1)
print(trn_new_features.shape)
print(dev_new_features.shape)

classifier2 = LogisticRegression()
classifier2.fit(trn_new_features, trn_labels)
print("Training accuracy = %f" % classifier2.score(trn_new_features, trn_labels))
print("Dev accuracy = %f" % classifier2.score(dev_new_features, dev_labels))

classifier3 = LogisticRegression()
grid_values = {'C':[0.01, 0.005, 0.001， 0.1], 'solver': ['lbfgs', 'newton-cg', 'sag']}
grid_clf_acc = GridSearchCV(classifier3, param_grid = grid_values,scoring = 'accuracy', verbose=10)
result = grid_clf_acc.fit(trn_new_features, trn_labels)

#Predict values based on new parameters
y_pred_acc = grid_clf_acc.predict(dev_new_features)

print('Accuracy Score : ' + str(accuracy_score(dev_labels,y_pred_acc)))

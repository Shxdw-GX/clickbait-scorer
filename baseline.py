import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

#load the data
train = pd.read_csv("data/clean/train.csv")
val = pd.read_csv("data/clean/validation.csv")
test = pd.read_csv("data/clean/test.csv")

#turn the headlines into numbers
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
x_train = tfidf.fit_transform(train["headline"])
x_val = tfidf.transform(val["headline"])
x_test = tfidf.transform(test["headline"])

#train the model
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(x_train, train["label"])


def evaluate(name, x, y):
    pred = model.predict(x)
    prob = model.predict_proba(x)[:, 1]

    print(name)
    print("Accuracy:", round(accuracy_score(y, pred), 4))
    print("Precision:", round(precision_score(y, pred), 4))
    print("Recall:", round(recall_score(y, pred), 4))
    print("F1:", round(f1_score(y, pred), 4))
    print("ROC AUC:", round(roc_auc_score(y, prob), 4))
    print("Confusion matrix:")
    print(confusion_matrix(y, pred))
    print()


evaluate("Validation", x_val, val["label"])
evaluate("Test", x_test, test["label"])

#save the model for the demo
file = open("baseline.pkl", "wb")
pickle.dump((tfidf, model), file)
file.close()


def get_score(headline):
    x = tfidf.transform([headline])
    prob = model.predict_proba(x)
    return prob[0][1]


#try a few headlines
headlines = [
    "You Won't Believe What This Dog Did Next",
    "Senate passes budget bill after late night vote",
    "17 Things Only 90s Kids Will Understand",
    "City council approves new road repair plan",
]

for h in headlines:
    print(round(get_score(h), 3), h)
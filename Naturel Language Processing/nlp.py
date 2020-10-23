import numpy as np
import pandas as pd
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix


comments = pd.read_csv('Restaurant_Reviews.csv')
print(comments)


ps = PorterStemmer()

nltk.download('stopwords')

# Preprocessing
comp = []
for i in range(1000):
    comment = re.sub('[^a-zA-Z]', ' ', comments['Review'][i])
    comment = comment.lower()
    comment = comment.split()
    comment = [ps.stem(word) for word in comment if not word in set(stopwords.words('english'))]
    comment = ' '.join(comment)
    comp.append(comment)

# Feautre Extraction
# Bag of Words (BOW)

cv = CountVectorizer(max_features=2000)
X = cv.fit_transform(comp).toarray()
y = comments.iloc[:, 1].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)


cm = confusion_matrix(y_test, y_pred)
print(cm)
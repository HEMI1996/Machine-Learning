

```python
# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```


```python
# importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3) #quoting=3 to ignore quotes(' & ") in the dataset
```


```python
dataset.shape
```




    (1000, 2)




```python
dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Review</th>
      <th>Liked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wow... Loved this place.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Crust is not good.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Not tasty and the texture was just nasty.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Stopped by during the late May bank holiday of...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The selection on the menu was great and so wer...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



In the dataset given reviews of a restaurant. We need to distinguish wheather review is good '1' or bad '0'


```python
# Cleaning the tests
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
```

    [nltk_data] Error loading stopwords: <urlopen error [WinError 10060] A
    [nltk_data]     connection attempt failed because the connected party
    [nltk_data]     did not properly respond after a period of time, or
    [nltk_data]     established connection failed because connected host
    [nltk_data]     has failed to respond>
    


```python
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values
```


```python
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```


```python
# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
```




    GaussianNB(priors=None)




```python
# Predicting the Test set results
y_pred = classifier.predict(X_test)
```


```python
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
```


```python
cm
```




    array([[55, 42],
           [12, 91]], dtype=int64)




```python
(55+91)/(55+42+12+91)
```




    0.73



we got 73% accuracy

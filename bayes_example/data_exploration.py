import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, roc_auc_score, log_loss
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

import xgboost as xgb

random_seed = 2000


def load_data():
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    return train_data, test_data


def clean_text(data):
    ps = PorterStemmer()
    corpus = []
    for i in range(data.shape[0]):
        text = data['text'][i]
        review = re.sub('[^A-Za-z0-9]', ' ', text)
        review = word_tokenize(review)
        review = [word for word in str(review).lower().split() if word not in set(stopwords.words('english'))]
        review = [ps.stem(word) for word in review]
        review = ' '.join(review)
        corpus.append(review)
    return corpus


def text_len(df):
    df['num_words'] = df['text'].apply(lambda x: len(str(x).split()))
    df['num_uniq_words'] = df['text'].apply(lambda x: len(set(str(x).split())))
    df['num_chars'] = df['text'].apply(lambda x: len(str(x)))
    df['num_stop_words'] = df['text'].apply(
        lambda x: len([word for word in str(x).lower().split() if word in set(stopwords.words('english'))]))
    df['num_punctuations'] = df['text'].apply(lambda x: len([word for word in str(x) if word in string.punctuation]))
    df['num_words_upper'] = df['text'].apply(lambda x: len([word for word in str(x) if word is word.upper()]))
    df['num_words_title'] = df['text'].apply(lambda x: len([word for word in str(x) if word is word.istitle()]))
    df['mean_word_len'] = df['text'].apply(lambda x: np.mean([len(word) for word in str(x).lower().split()]))


print('starting load data...')
train_data, test_data = load_data()

print('starting clean text column...')
train_clean = clean_text(train_data)
test_clean = clean_text(test_data)
train_data['clean_data'] = train_clean
test_data['clean_data'] = test_clean
del train_clean, test_clean

print('explore text info...')
text_len(train_data)
text_len(test_data)

print('word to vector')
# cv = CountVectorizer(max_features=2000, ngram_range=(1, 3), dtype=np.int8, stop_words='english')
# x_train = cv.fit_transform(train_data['clean_data']).toarray()
# x_test = cv.fit_transform(test_data['clean_data']).toarray()
tfidf = TfidfVectorizer(max_features=2000,dtype=np.float32,analyzer='word',
                        ngram_range=(1, 3),use_idf=True, smooth_idf=True,
                        sublinear_tf=True)
x_train = tfidf.fit_transform(train_data['clean_data']).toarray()
x_test = tfidf.fit_transform(test_data['clean_data']).toarray()
## map author name
name_ecode = {'EAP': 0, 'HPL': 1, 'MWS': 2}
y = train_data['author'].map(name_ecode)

kf = KFold(n_splits=10, shuffle=True, random_state=random_seed)
mNB = MultinomialNB()
predict_full_prob = 0
predict_score = []
count = 1
for train_index, test_index in kf.split(x_train):
    print('{} of KFlod {}'.format(count, kf.n_splits))
    x1, x2 = x_train[train_index], x_train[test_index]
    y1, y2 = y[train_index], y[test_index]
    mNB.fit(x1, y1)
    y_predict = mNB.predict(x2)
    predict_score.append(log_loss(y2, mNB.predict_proba(x2)))
    predict_full_prob += mNB.predict_proba(x_test)
    count += 1

print(predict_score)
print('mean of predict score:{}'.format(np.mean(predict_score)))
print('confusion matrix:\n',confusion_matrix(y2,y_predict))

y_pred = predict_full_prob/10
submit = pd.DataFrame(test_data['id'])
submit = submit.join(pd.DataFrame(y_pred))
submit.columns = ['id','EAP','HPL','MWS']
#submit.to_csv('spooky_pred1.csv.gz',index=False,compression='gzip')
submit.to_csv('data/spooky_pred2.csv',index=False)

#
# print(train_data.shape)
# print(test_data.shape)

# plt.figure(figsize=(14,6))
# plt.subplot(211)
# sns.heatmap(pd.crosstab(train_data['author'],train_data['num_words']),cmap='gist_earth',xticklabels=False)
# plt.xlabel('Original text word count')
# plt.ylabel('Author')
#
# plt.subplot(212)
# sns.heatmap(pd.crosstab(train_data['author'],train_data['num_uniq_words']),cmap='gist_heat',xticklabels=False)
# plt.xlabel('Unique text word count')
# plt.ylabel('Author')
# plt.tight_layout()
# plt.show()


# print(train_data.head())
# print(train_data['author'].value_counts())

# plt.figure(figsize=(14,5))
# sns.countplot(train_data['author'])
# plt.xlabel('Author')
# plt.title('Target variable distribution')
# plt.show()
# text = train_data['text'][0]
# review = re.sub('[^A-Za-z0-9]',' ',text)
# review = word_tokenize(text)
# review = [word for word in str(train_data['text'][0]).lower().split() if  word not in set(stopwords.words('english'))]
# ps = PorterStemmer()
# review = [ps.stem(word) for word in str(text).lower().split()]
# print(review)



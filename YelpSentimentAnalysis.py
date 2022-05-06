import numpy as np
import pandas as pd
import neattext.functions as nfx
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")
import re
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize
import gensim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print("Load the review dataset, text cleaning and vectorization")
df = pd.read_json('C:/Users/nican/Desktop/IAPROJECT/YelpReviewReduced.json',lines=True)
df['text'] = df['text'].apply(nfx.remove_stopwords)
df['text'] = df['text'].apply(nfx.remove_special_characters)
df['text'] = df['text'].str.lower() 
df.loc[df['stars'] < 4, 'sentiment'] = 'negative' 
df.loc[df['stars'] > 3, 'sentiment'] = 'positive'
Xfeatures=df['text']
Ylabels=df['sentiment']
tfidf_vec = TfidfVectorizer()
X = tfidf_vec.fit_transform(Xfeatures)

print("Text recognition on the whole YELP dataset on the negative/positive reviews")
modelLR = LogisticRegression(max_iter=300)
x_train,x_test,y_train,y_test = train_test_split(X,Ylabels,test_size=0.2,random_state=4)
modelLR.fit(x_train,y_train)
predictionLR = modelLR.predict(x_test)
labels=np.unique(y_test)
a=confusion_matrix(y_test,predictionLR,labels=labels)
print(pd.DataFrame(a,index=labels,columns=labels))
print(accuracy_score(y_test, predictionLR))
print(classification_report(y_test, predictionLR))

print("Load the business dataset, filter for McDonald's, merge and delete useless columns")
dfBusiness = pd.read_json('C:/Users/nican/Desktop/IAPROJECT/yelp2021/yelp_academic_dataset_business.json',lines=True)
MCdonald=dfBusiness[dfBusiness['name']=="McDonald's"]
joinedDf=pd.merge(MCdonald, df, on='business_id')
joinedDf = joinedDf.drop(['date','address','city','postal_code','latitude', 'review_count', 'hours','categories', 'attributes', 'state'],  1)

from wordcloud import WordCloud
#concatenate all the reviews into one single string 
full_text = ' '.join(joinedDf['text'])
cloud_no_stopword = WordCloud(background_color='white').generate(full_text)
plt.imshow(cloud_no_stopword, interpolation='bilinear')
plt.axis('off')
plt.show()

print("Analyze McDonald's data using the Word2Vec model")
#positive and service works, negative and service works,
df_good=joinedDf[(joinedDf['sentiment']=='positive')]
df_good=df_good.text.apply(gensim.utils.simple_preprocess)
model = gensim.models.Word2Vec(sentences=df_good, vector_size=100, window=10, min_count=5)
model.train(df_good,total_examples=model.corpus_count,epochs=50)
print(model.wv.most_similar("service",topn=10))

print("Sentiment recognition of the McDonald's historic reviews, using Logistic Regression and SVC")
vect = CountVectorizer(max_features=300)
vect.fit(joinedDf.text)
X = vect.transform(joinedDf.text)
X_df = pd.DataFrame(X.toarray(), columns=vect.get_feature_names())
y = joinedDf.sentiment
X = X_df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4, stratify=y)
modelLR2 = LogisticRegression().fit(X_train, y_train)
pred = modelLR2.predict(X_test)
print(accuracy_score(y_test,pred))
labels=np.unique(y_test)
a=confusion_matrix(y_test,pred,labels=labels)
print(pd.DataFrame(a,index=labels,columns=labels))
print("Most informative words for the recognition")
log_odds = modelLR2.coef_[0]
coeff = pd.DataFrame(log_odds, X.columns, columns=['coef'])\
            .sort_values(by='coef', ascending=False)
print(coeff)
print("Comparison with SVC algorithm")
modelSVC=SVC()
modelSVC.fit(X_train, y_train)
predictionSVC = modelSVC.predict(X_test)
print(accuracy_score(y_test, predictionSVC))
a=confusion_matrix(y_test,predictionSVC,labels=labels)
print(pd.DataFrame(a,index=labels,columns=labels))
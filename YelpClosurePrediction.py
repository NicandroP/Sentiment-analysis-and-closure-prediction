import pandas as pd
import numpy as nm
import warnings
from matplotlib import pyplot
warnings.filterwarnings("ignore")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from textblob import TextBlob
import neattext.functions as nfx
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

print("Load the review dataset, remove useles columns and preprocess the text")
dfReview = pd.read_json('C:/Users/nican/Desktop/IAPROJECT/yelp_training_set2013/yelp_training_set_review.json',lines=True)
dfReview["date"]=  pd.to_datetime(dfReview["date"], format='%Y-%m-%d')
dfReview['year'] = dfReview.date.dt.year
dfReview = dfReview.drop(['date','type','votes','user_id','review_id'], 1)
dfReview['text'] = dfReview['text'].apply(nfx.remove_stopwords)
dfReview['text'] = dfReview['text'].apply(nfx.remove_special_characters)
dfReview['text'] = dfReview['text'].str.lower()

dfReview=dfReview[(dfReview['year'] > 2006)]

#create a dataset for year that cointains only business with at least 5 reviews
df2007=dfReview[(dfReview['year']==2007)]
df2008=dfReview[(dfReview['year']==2008)]
df2009=dfReview[(dfReview['year']==2009)]
df2010=dfReview[(dfReview['year']==2010)]
df2011=dfReview[(dfReview['year']==2011)]

df2007=df2007.groupby("business_id").filter(lambda x: len(x) >= 8)
df2008=df2008.groupby("business_id").filter(lambda x: len(x) >= 8)
df2009=df2009.groupby("business_id").filter(lambda x: len(x) >= 8)
df2010=df2010.groupby("business_id").filter(lambda x: len(x) >= 8)
df2011=df2011.groupby("business_id").filter(lambda x: len(x) >= 8)


#add the polarity column
df2007['polarity']=0
df2008['polarity']=0
df2009['polarity']=0
df2010['polarity']=0
df2011['polarity']=0


#compute the polarity for each dataset
i=0
for text in df2007.text:
    blob=TextBlob(text)
    df2007.iloc[i,df2007.columns.get_loc('polarity')]=blob.sentiment.polarity
    i+=1

h=0
for text in df2008.text:
    blob=TextBlob(text)
    df2008.iloc[h,df2008.columns.get_loc('polarity')]=blob.sentiment.polarity
    h+=1

k=0
for text in df2009.text:
    blob=TextBlob(text)
    df2009.iloc[k,df2009.columns.get_loc('polarity')]=blob.sentiment.polarity
    k+=1

y=0
for text in df2010.text:
    blob=TextBlob(text)
    df2010.iloc[y,df2010.columns.get_loc('polarity')]=blob.sentiment.polarity
    y+=1

j=0
for text in df2011.text:
    blob=TextBlob(text)
    df2011.iloc[j,df2011.columns.get_loc('polarity')]=blob.sentiment.polarity
    j+=1



#concat all the dataset
dfConcat=pd.concat([df2007,df2008, df2009,df2010,df2011], axis=0)


#importing business dataset
dfBusiness=pd.read_json('C:/Users/nican/Desktop/IAPROJECT/yelp_training_set2013/yelp_training_set_business.json',lines=True)

dfBusiness = dfBusiness.drop(['full_address','categories','city','review_count','name','neighborhoods','longitude','latitude','state','type'], 1)

#merge of the two dataset and mean of the values
joinedDf=pd.merge(dfConcat, dfBusiness, on='business_id')
joinedDf=joinedDf.groupby("business_id").filter(lambda x: len(x) >= 40)
finalDf=joinedDf.groupby(['business_id'],as_index=False).mean()#boolean values are trasformed to int, True=1 False=0
finalDf = finalDf.drop('stars_y', 1)
finalDf.rename(columns={'stars_x': 'rating'}, inplace=True)
finalDf=finalDf.drop('year',1)

#start of the ML
print("Start the ML function")
dfClosed=finalDf[finalDf['open']==0]
dfOpen=finalDf[finalDf['open']==1]
print(len(dfOpen))
print(len(dfClosed))

accuracy=0
accuracyLR=0

for item in nm.array_split(dfOpen, 6):
    print(len(item))
    union=pd.concat([item, dfClosed], axis=0)
    Xfeatures=union.loc[:, ~union.columns.isin(['business_id', 'open'])]
    Ylabels=union['open']
    x_train,x_test,y_train,y_test = train_test_split(Xfeatures,Ylabels,test_size=0.2,random_state=4)
    """ st_x= StandardScaler()
    st_x.fit(x_train) 
    x_train= st_x.fit_transform(x_train)    
    x_test= st_x.transform(x_test) """
    error = []
    for i in range(1, 41):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train, y_train)
        pred_i = knn.predict(x_test)
        error.append(nm.mean(pred_i != y_test))
    pyplot.figure(figsize=(12, 6))
    pyplot.plot(range(1, 41), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
    pyplot.title('Error Rate K Value')
    pyplot.xlabel('K Value')
    pyplot.ylabel('Mean Error')
    #pyplot.show() 
    min_index = error.index(min(error))+1
    model=KNeighborsClassifier(n_neighbors=min_index)
    model.fit(x_train,y_train)
    prediction = model.predict(x_test)
    modelLR=LogisticRegression()
    modelLR.fit(x_train,y_train)
    predictionLR = modelLR.predict(x_test)
    accuracyLR+=accuracy_score(y_test, predictionLR)
    labels=nm.unique(y_test)
    a=confusion_matrix(y_test,prediction,labels=labels)
    print(pd.DataFrame(a,index=labels,columns=labels))
    print(accuracy_score(y_test, prediction))
    #print(classification_report(y_test, prediction))
    accuracy+=accuracy_score(y_test, prediction)
print("Accuracy mean of KNN: ",accuracy/6)
print("Accuracy mean of LR: ",accuracyLR/6)

import pickle
pickle.dump(model, open('model.pkl', 'wb'))
print('Model is saved into to disk successfully')
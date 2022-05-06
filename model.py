import neattext.functions as nfx
from textblob import TextBlob
import pickle
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

df= pd.read_csv("C:/Users/nican/Desktop/IAPROJECT/testreviews.csv",sep=';',on_bad_lines='skip')

df['polarity']=0
df['text']=df['text'].apply(nfx.remove_stopwords)
df['text']=df['text'].apply(nfx.remove_special_characters)
df['text']=df['text'].str.lower()
h=0
for text in df.text:
    blob=TextBlob(text)
    df.iloc[h,df.columns.get_loc('polarity')]=blob.sentiment.polarity
    h=+1
df=df.drop('text',1)
df=df.mean()
df=[df]
pickled_model=pickle.load(open('C:/Users/nican/Desktop/IAPROJECT/model.pkl','rb'))
result=pickled_model.predict(df)
print(int(result))
#Tweet Analysis Code...
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pprint import pprint
from pyspark.sql import Row
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)
from bs4 import BeautifulSoup as bs
import re
from nltk.tokenize import WordPunctTokenizer as tokenizer


dataLines=sc.textFile("training.1600000.processed.noemoticon.csv")

dataLines.count()
dataLines.first()
dataLines.cache()
dataLines.take(5)


#df=pd.read_csv('training.1600000.processed.noemoticon.csv')
#words=dataLines.flatMap(lambda line: line.split(","))
#words.take(20)
#words=tweetData.filter(lambda line: line.split(","))
#words.take(20)
#header=['sentiment','id','date','query_string','user','text']

parts=dataLines.map(lambda l:l.split(","))
tweetMap=parts.map(lambda p:Row(sentimentId=p[1],sentiment=p[0],user=p[4],text=p[5]))
tweetMap.collect()

tweetDf=sqlContext.createDataFrame(tweetMap)
tweetDf.registerTempTable("tweets")

tweetPandas=tweetDf.toPandas()
print(tweetPandas.sentiment[279])

#Convert the 'sentiment' datatype from string to float type.

tweetPandas['sentiment']=tweetPandas['sentiment'].replace('"',' ', regex=True).astype(float)
tweetPandas['sentimentId']=tweetPandas['sentimentId'].replace('"',' ', regex=True).astype(int)
#tweetDf.head()
#tweetDf.select("text").distinct().show()


tok = tokenizer()
patttern1 = r'@[A-Za-z0-9_]+'
patttern2 = r'https?://[^ ]+'
combined_pat = r'|'.join((patttern1, patttern2))

www_pat = r'www.[^ ]+'
negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')

def TweetCleaner(text):
    soup = bs(text, 'lxml')
    souped = soup.get_text()
    try:
        bom_removed = souped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        bom_removed = souped
    stripped = re.sub(combined_pat, '', bom_removed)
    stripped = re.sub(www_pat, '', stripped)
    lower_case = stripped.lower()
    neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case)
    letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = [x for x  in tok.tokenize(letters_only) if len(x) > 1]
    return (" ".join(words)).strip()




num_segments = [0,400000,800000,1200000,1600000]
print ("Cleaning and parsing the tweets...\n")
clean_tweet_texts = []
for i in range(num_segments[0],num_segments[1]):
    if( (i+1)%10000 == 0 ):
        print ("Tweets %d of %d has been processed" % ( i+1, num_segments[1] ))                                                                    
    clean_tweet_texts.append(TweetCleaner(tweetPandas['text'][i]))
    
for i in range(num_segments[1],num_segments[2]):
    if( (i+1)%10000 == 0 ):
        print ("Tweets %d of %d has been processed" % ( i+1, num_segments[2] ))                                                                    
    clean_tweet_texts.append(TweetCleaner(tweetPandas['text'][i]))
    
for i in range(num_segments[2],num_segments[3]):
    if( (i+1)%10000 == 0 ):
        print ("Tweets %d of %d has been processed" % ( i+1, num_segments[3] ))                                                                    
    clean_tweet_texts.append(TweetCleaner(tweetPandas['text'][i]))
    
for i in range(num_segments[3],num_segments[4]):
    if( (i+1)%10000 == 0 ):
        print ("Tweets %d of %d has been processed" % ( i+1, num_segments[4] ))                                                                    
    clean_tweet_texts.append(TweetCleaner(tweetPandas['text'][i]))

clean_df = pd.DataFrame(clean_tweet_texts,columns=['text'])
clean_df['target'] = tweetPandas.sentiment
clean_df.head()

#CHecking the cleaned dataset 

clean_df.info()
clean_df[clean_df.isnull().any(axis=1)].head()
np.sum(clean_df.isnull().any(axis=1))


df = pd.read_csv("training.1600000.processed.noemoticon.csv",header=None)
df.iloc[clean_df[clean_df.isnull().any(axis=1)].index,:].head()

clean_df.dropna(inplace=True)
clean_df.reset_index(drop=True,inplace=True)
clean_df.info()

clean_df.to_csv('clean_tweet.csv',encoding='utf-8')


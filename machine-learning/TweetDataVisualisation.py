df=pd.read_csv('clean_tweetfloat.csv')

neg_tweets=df[df['target']==0.0]


neg_string=[]
for t in neg_tweets.text:
    neg_string.append(t)
print(neg_string)
neg_string = pd.Series(neg_string).str.cat(sep=' ')

from wordcloud import WordCloud

wordcloud1 = WordCloud(width=1600, height=800,max_font_size=200).generate(neg_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud1, interpolation="bilinear")
plt.axis("off")
plt.show()

pos_tweets=df[df['target']==1.0]

pos_string=[]
for t in pos_tweets.text:
    pos_string.append(t)
pos_string = pd.Series(pos_string).str.cat(sep=' ')

wordcloud2=WordCloud(width=1600, height=800,max_font_size=200).generate(neg_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud2, interpolation="bilinear")
plt.axis("off")
plt.show()

#CountVectorizer.
df=pd.read_csv('clean_tweetfloat.csv')

tweets=df['text']

from sklearn.feature_extraction.text import CountVectorizer
#from pyspark.ml.feature import CountVectorizer
cv = CountVectorizer()
cv.fit(df['text'].values.astype('U')) 

len(cv.get_feature_names())

neg_doc_matrix = cv.transform(df[df.target == 0].text)
pos_doc_matrix = cv.transform(df[df.target == 1].text)

neg_tf = np.sum(neg_doc_matrix,axis=0)
pos_tf = np.sum(pos_doc_matrix,axis=0)

neg = np.squeeze(np.asarray(neg_tf))
pos = np.squeeze(np.asarray(pos_tf))
term_freq_df = pd.DataFrame([neg,pos],columns=cv.get_feature_names()).transpose()


doc_matrix = cv.transform(df.text)
df[df.target == 0.0].tail()


neg_batches = np.linspace(0,298179,100).astype(int)
i=0
neg_tf = []
while i < len(neg_batches)-1:
    batch_result = np.sum(doc_matrix[neg_batches[i]:neg_batches[i+1]].toarray(),axis=0)
    neg_tf.append(batch_result)
    if (i % 10 == 0) | (i == len(neg_batches)-2):
        print (neg_batches[i+1],"entries' term freuquency calculated")
    i += 1
neg_batches = np.linspace(298180,498179,100).astype(int)
i=0
while i < len(neg_batches)-1:
    batch_result = np.sum(doc_matrix[neg_batches[i]:neg_batches[i+1]].toarray(),axis=0)
    neg_tf.append(batch_result)
    if (i % 10 == 0) | (i == len(neg_batches)-2):
        print (neg_batches[i+1],"entries' term freuquency calculated")
    i += 1
neg_batches = np.linspace(498180,698179,100).astype(int)
i=0
while i < len(neg_batches)-1:
    batch_result = np.sum(doc_matrix[neg_batches[i]:neg_batches[i+1]].toarray(),axis=0)
    neg_tf.append(batch_result)
    if (i % 10 == 0) | (i == len(neg_batches)-2):
        print (neg_batches[i+1],"entries' term freuquency calculated")
    i += 1

neg_batches = np.linspace(698180,798179,100).astype(int)
i=0
while i < len(neg_batches)-1:
    batch_result = np.sum(doc_matrix[neg_batches[i]:neg_batches[i+1]].toarray(),axis=0)
    neg_tf.append(batch_result)
    if (i % 10 == 0) | (i == len(neg_batches)-2):
        print (neg_batches[i+1],"entries' term freuquency calculated")
    i += 1

pos_batches = np.linspace(798179,996019,100).astype(int)
i=0
pos_tf = []
while i < len(pos_batches)-1:
    batch_result = np.sum(doc_matrix[pos_batches[i]:pos_batches[i+1]].toarray(),axis=0)
    pos_tf.append(batch_result)
    if (i % 10 == 0) | (i == len(pos_batches)-2):
        print (pos_batches[i+1],"entries' term freuquency calculated")
    i += 1
    
pos_batches = np.linspace(996020,1196019,100).astype(int)
i=0
while i < len(pos_batches)-1:
    batch_result = np.sum(doc_matrix[pos_batches[i]:pos_batches[i+1]].toarray(),axis=0)
    pos_tf.append(batch_result)
    if (i % 10 == 0) | (i == len(pos_batches)-2):
        print (pos_batches[i+1],"entries' term freuquency calculated")
    i += 1

pos_batches = np.linspace(1196020,1398179,100).astype(int)
i=0
while i < len(pos_batches)-1:
    batch_result = np.sum(doc_matrix[pos_batches[i]:pos_batches[i+1]].toarray(),axis=0)
    pos_tf.append(batch_result)
    if (i % 10 == 0) | (i == len(pos_batches)-2):
        print (pos_batches[i+1],"entries' term freuquency calculated")
    i += 1

pos_batches = np.linspace(1398180,1596019,100).astype(int)
i=0
while i < len(pos_batches)-1:
    batch_result = np.sum(doc_matrix[pos_batches[i]:pos_batches[i+1]].toarray(),axis=0)
    pos_tf.append(batch_result)
    if (i % 10 == 0) | (i == len(pos_batches)-2):
        print (pos_batches[i+1],"entries' term freuquency calculated")
    i += 1

neg = np.sum(neg_tf,axis=0)
pos = np.sum(pos_tf,axis=0)
#term_freq_df = pd.DataFrame([neg,pos],columns=cv.get_feature_names()).transpose()
term_freq_df.head()

term_freq_df.columns = ['negative', 'positive']
term_freq_df['total'] = term_freq_df['negative'] + term_freq_df['positive']
term_freq_df.sort_values(by='total', ascending=False).iloc[:10]

term_freq_df.to_csv('term_freq_df.csv',encoding='utf-8')

#Lets see how the tokens look on a GRaph

y_pos = np.arange(500)
plt.figure(figsize=(10,8))
s = 1
expected_zipf = [term_freq_df.sort_values(by='total', ascending=False)['total'][0]/(i+1)**s for i in y_pos]
plt.bar(y_pos, term_freq_df.sort_values(by='total', ascending=False)['total'][:500], align='center', alpha=0.5)
plt.plot(y_pos, expected_zipf, color='r', linestyle='--',linewidth=2,alpha=0.5)
plt.ylabel('Frequency')
plt.title('Top 500 tokens in tweets')

#the top 50 words in negative tweets on a bar chart.

y_pos = np.arange(50)
plt.figure(figsize=(12,10))
plt.bar(y_pos, term_freq_df.sort_values(by='negative', ascending=False)['negative'][:50], align='center', alpha=0.5)
plt.xticks(y_pos, term_freq_df.sort_values(by='negative', ascending=False)['negative'][:50].index,rotation='vertical')
plt.ylabel('Frequency')
plt.xlabel('Top 50 negative tokens')
plt.title('Top 50 tokens in negative tweets')


#top 50 positive tokens on a bar chart.

y_pos = np.arange(50)
plt.figure(figsize=(12,10))
plt.bar(y_pos, term_freq_df.sort_values(by='positive', ascending=False)['positive'][:50], align='center', alpha=0.5)
plt.xticks(y_pos, term_freq_df.sort_values(by='positive', ascending=False)['positive'][:50].index,rotation='vertical')
plt.ylabel('Frequency')
plt.xlabel('Top 50 positive tokens')
plt.title('Top 50 tokens in positive tweets')
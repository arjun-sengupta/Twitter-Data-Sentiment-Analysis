import pickle
with open('tweetsSentiModel2', 'rb') as f:
    loadedModel = pickle.load(f)
    
import pandas as pd
df = pd.read_csv("testData.csv")

tweetText = df.text

def predictSentiment(tweetText):
    prediction = loadedModel.predict(tweetText)
    d = {'Tweet-Text':tweetText , 'Prediction': prediction}
    prediction_df = pd.DataFrame(data = d)
    prediction_df.to_csv('prediction_phrase_success_df.csv')
    print("Predictions are....")
    print(prediction_df)
    
#Predict
        
predictSentiment(tweetText)

phrase = input('Enter a word/phrase to search in the corpus of tweet data:')

new_df_tweetText = df[df['text'].str.contains(phrase)].text

predictSentiment(new_df_tweetText)

prediction_phrase_fail_df = pd.read_csv("prediction_phrase_fail_df.csv")
import matplotlib.pyplot as plt
import numpy as np

def scatterplot(x_data, y_data, x_label="", y_label="", title="", yscale_log=False):

    # Create the plot object
    _, ax = plt.subplots()

    # Plot the data, set the size (s), color and transparency (alpha)
    # of the points
    col=[]
    for i in range(0,len(y_data)):
        if y_data[i] == 0.0:
            col.append('red')
        else:
            col.append('green')
    
    ax.scatter(x_data, y_data, s = 20, color = col, alpha = 0.75)
    if yscale_log == True:
        ax.set_yscale('log')
    
    red_dot, = plt.plot(3, "ro", markersize=5)
    green_dot, = plt.plot(3,"go",markersize=5)
    plt.plot(3,"wo",markersize=5)
    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.legend([red_dot,green_dot] , ["Negative","Positive"])
    
x_data = list(prediction_phrase_fail_df['Unnamed: 0'])
y_data = list(prediction_phrase_fail_df['Prediction'])

scatterplot(x_data,y_data,"Tweet ID","Predictions","Tweet ID vs Prediction Graph(Phrase='fail')")

prediction_phrase_success_df = pd.read_csv("prediction_phrase_success_df.csv")
x_data = list(prediction_phrase_success_df['Unnamed: 0'])
y_data = list(prediction_phrase_success_df['Prediction'])

scatterplot(x_data,y_data,"Tweet ID","Predictions","Tweet ID vs Prediction Graph(Phrase='success')")

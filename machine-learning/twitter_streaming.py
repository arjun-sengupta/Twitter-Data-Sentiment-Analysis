#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 23:22:04 2018

@author: arjun
"""

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

#Variables that contains the user credentials to access Twitter API 
access_token = "154056702-Z5Okhp9xgGn8gqwninz0mPlK0CiIf4d3hP9BZ1vH"
access_token_secret = "Sn02l1V1pcK4EwPHickabksfCMC2zaLSNSmSaz1EKjaLK"
consumer_key = "UBbexQYbYVHeKGoHeBUzskMVZ"
consumer_secret = "53t6hUpt26DBQTQyCANKq2VAZJEXulTdjyzhgfF8S3J84QCdqu"


#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):

    def on_data(self, data):
        print (data)
        return True

    def on_error(self, status):
        print (status)


if __name__ == '__main__':

    #This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)

    #This line filter Twitter Streams to capture data by the keywords: 'python', 'javascript', 'ruby'
    stream.filter(track=['sports','movies','politics'])

#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis on BlackLivesMatter Movement
# by Saida Muktar

# ### Abstract
#  * Police shootings is something that we all see in the news very regularly now.
#  * We have previously analyzed how police shootings differ between races. According to our analysis Black people are being disproportionately killed by Police than other races. This has risen a Black Lives Matter Movement around the country. Why wouldn't you support the Black Lives Movement? I mean it is a good cause, right? However, there are many around the country that argue saying all lives matter or blue lives matter. We will be analyzing how people feel by using Twitter API to do a sentiment analysis on this issue.
#  * In the first section the report will go over how the data was obtained for the analysis
#  * In the second section the report will go over the data analysis
#  * Finally, in the last section the report will go over the conclusions that came about from this analysis answering our research questions.
# 

# ### Obtaining Data
#  * The data we will be using was obtaining from Twitter
#  * In order to get Data from Twitter we needed to use API.
#  * To use the Twitter API you would need to have an:
#  
#      * API Key
#      
#      * API Secret Key
#      
#      * Access Token
#      
#      * Access Token Secret
#      
#    You can obtaining by first creating a Twitter account and then heading over to [Twitter Developer] (https://developer.twitter.com/en) and signing in. Once you sign it you would have to fill out some uestions before you are given your uniue keys.
#    
#  * Now that you have your unie keys we would have to install and import tweepy.
#  * Once tweepy is installed, you would need to use your Keys to be authorized to use the twitter API, once you put that in you can go ahead and call the api.
#  * This process is shown below:

# In[5]:


# tweepy installation

import sys
get_ipython().system('conda install -c conda-forge --yes --prefix {sys.prefix} tweepy')

# accessing Twitter API

import tweepy
auth = tweepy.OAuthHandler('API KEY', 'API SECRET KEY')
auth.set_access_token('ACCESS TOKEN', 'ACCESS TOKEN SECRET')

api = tweepy.API(auth)


# In[8]:


import pandas as pd
import re
import matplotlib.pyplot as plt

#Data Fetching
#This searched the first 100 tweets with the text "blacklivesmatter", "alllivesmatter", "bluelivesmatter", "whitelivesmatter"

blm = api.search('blacklivesmatter', count = 100, lang = "en")
alllives = api.search('alllivesmatter', count = 100, lang = "en")
bluelives = api.search('bluelivesmatter', count = 100, lang = "en")
whitelives = api.search('whitelivesmatter', count = 100, lang = "en")

#Once we have the tweets we will put them in a DataFrame so this could be easier when we clean the text 

blmdata = pd.DataFrame([tweet.text for tweet in blm], columns= ['Tweets on BLM'])
alllivesdata = pd.DataFrame([tweet.text for tweet in alllives], columns= ['Tweets on alllives'])
bluelivesdata = pd.DataFrame([tweet.text for tweet in bluelives], columns= ['Tweets on bluelives'])
whitelivesdata = pd.DataFrame([tweet.text for tweet in whitelives], columns= ['Tweets on whitelives'])

#To clean the tweets we are using the remove re, we are removing words that contain @ or # or links and retweets

def cleanTxt(text):
    text = re.sub(r'@[A-Za-z0-9]+','',text)
    text = re.sub(r'#','',text)
    text = re.sub(r'https?:\/\/\S+','',text)
    text = re.sub(r'RT[\s]+','',text)
    return text

whitelivesdata['Tweets on whitelives'] = whitelivesdata['Tweets on whitelives'].apply(cleanTxt)
bluelivesdata['Tweets on bluelives'] = bluelivesdata['Tweets on bluelives'].apply(cleanTxt)
alllivesdata['Tweets on alllives'] = alllivesdata['Tweets on alllives'].apply(cleanTxt)
blmdata['Tweets on BLM'] = blmdata['Tweets on BLM'].apply(cleanTxt)


# These are the first five datas for each word we searched twitter that we get after we clean:
# ![image](https://user-images.githubusercontent.com/70491460/95353817-b7b05980-0891-11eb-9e54-bda7f401db98.png)
# ![image](https://user-images.githubusercontent.com/70491460/95354022-f47c5080-0891-11eb-9f88-a46f727826f8.png)
# ![image](https://user-images.githubusercontent.com/70491460/95354139-170e6980-0892-11eb-857c-118d12a5a5c0.png)
# ![image](https://user-images.githubusercontent.com/70491460/95354278-3a391900-0892-11eb-8f06-453102324eb8.png)
# 
# 
# 

# In[10]:


# Save these as csv file 
whitelivesdata.to_csv('Tweets_on_BLM.csv')
blmdata.to_csv('Tweets_on_BLM.csv')
bluelivesdata.to_csv('Tweets_on_BLM.csv')
alllivesdata.to_csv('Tweets_on_BLM.csv')


# ### Data Analysis
# 
# * Now that we have obtained the data we will be analyzing whether these tweets on the four keywords chosen are positive or negative, using TextBlob as our basis of what is positive, negative or neutral.
# * TextBlob is a Python library for processing textual data. It provides a simple API for diving into common natural language processing (NLP) tasks such as part-of-speech tagging, noun phrase extraction, sentiment analysis, classification, translation, and more.
# * In order to do this we are going to find the subjectivity and polarity of each keyword.
# * Once we find that we are going to check if the polarity is positive, negative, or neutral to see how people who are tweeting these feel about them.

# * In the first part of the analysis we are going to import TextBlob 
# * We will then create a function to find the subjectivity and polarity using textblob
# * We will then apply these functions to the datas that we have obtained from Twitter creating three new columns of Subjectivity, Polarity and Analysis
# * The Analysis section will check if the score of the Polarity obtained for tweet is positive, negative, or neutral.

# After obtaining the Subjectivity and Polarity these graphs below were plotted to show the relationship between the Subjectivity and Polarity. There are four graphs to show the relathionship of the four keywords.
# 
# ![image](https://user-images.githubusercontent.com/70491460/95507062-db050280-097e-11eb-8cd5-ad1f1c9ccd6d.png)
# ![image](https://user-images.githubusercontent.com/70491460/95507953-3be10a80-0980-11eb-905f-32fb094faf98.png)
# ![image](https://user-images.githubusercontent.com/70491460/95508022-59ae6f80-0980-11eb-9e8f-f166a689d67f.png)
# ![image](https://user-images.githubusercontent.com/70491460/95508074-6f239980-0980-11eb-92f8-09630aae2d2b.png)
# 

# * Next we will be plotting a graph to show the amount of times they keywords were mentioned based of our analysis
# * In order to do that we will be using valuecounts() 
# 
# ![image](https://user-images.githubusercontent.com/70491460/95512961-93cf3f80-0987-11eb-8b90-efea665d57ba.png)
# 

# * The next step to the Data Analysis was to see the perecentage difference of the Positive, Negative, and Neutral tweets about Black Lives Matter, White Lives Matter, Blue Lives Matter, and All Lives Matter. 
# * The graph below will show the percentage of tweets that are Positive, Negative, and Neutral for each of the kwywords based on the TextBlob definition of what is Positive, Negative, and Neutral
# 
# ![image](https://user-images.githubusercontent.com/70491460/95513286-1952ef80-0988-11eb-9a0c-e0a12629df76.png)
# 

# ### Conclusion
#  * We have obtained data from Twitter looking by looking for the keywords BlackLivesMatter, WhiteLivesMatter, BlueLivesMatter, and AllLivesMatter.
#  * We have used sentiment analysis in order to see if the tweets with these key words were Positive, Negative or Neutral based on TextBlob.
#  * Finally, we found the percent of Positive, Negative, and Neutral tweets of each keyword and compared them.
#  * According to TextBlob:
#      * ~ 40% of people tweeting about Black Lives Matter are tweeting about it Negatively
#      * ~ 30% of people tweeting about White Lives Matter are tweeting about it Negatively
#      * ~ 70% of people tweeting about Blue Lives Matter are tweeting about it Negatively
#      * ~ 15% of people tweeting about All Lives Matter are tweeting about it Negatively
#      
#      
#  

# * According to TextBlob:
#     * ~ 25% of people tweeting about Black Lives Matter are tweeting about it Positvely
#     * ~ 35% of people tweeting about White Lives Matter are tweeting about it Positvely
#     * ~ 15% of people tweeting about Blue Lives Matter are tweeting about it Positively
#     * ~ 40% of people tweeting about All Lives Matter are tweeting about it Positively

# This report has shown that according to TextBlob Black Lives Matter is more negatively talked about than White Lives Matter and All Lives Matter. 
# Something that has come to a suprise is the amount of negativity that Blue Lives Matter has brought about on Twitter according to TextBlob which was not something I was expecting but shows to shine a light on the fact that people agree that cop lives yes matter but it's your choic to become a cop, and its different than a skin color which you can't change.

# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[3]:


import nltk
#nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import bs4 as bs
import numpy as np
import pandas as pd
import requests
from pandas import DataFrame
import csv
from datetime import datetime
import time
import pathlib

sid = SentimentIntensityAnalyzer()

#user input
url_link=input("Enter URL: ").rstrip() 
file_name=input("Name the file: ").rstrip() 

#create the csv with headers if csv does not already exist
csv_file_path = pathlib.Path('file_path.csv')
if csv_file_path.exists():
    input("This csv already exists. Press Enter to append existing csv.")
else:
    csv_header_df = pd.DataFrame(columns = ['Name', 'Date', 'Time', 'Score', 'Comment'])
    csv_header_df.to_csv(csv_file_path, index = False, mode = "a", header=True, encoding='utf-8-sig')


forum_thread_page_num = 1

while True:
    time.sleep(1)
    url = url_link + 'page-' + str(forum_thread_page_num)
    print(forum_thread_page_num)
    request = requests.get(url)
    response = request.text 
    soup = bs.BeautifulSoup(response)

    #create list of usernames so that they can later be matched to comments 
    username_list = []
    username = soup.findAll("a", {"itemprop": "name"})
    for name in username:
        new_name = name.get_text()
        username_list.append(new_name)
        
    #find the date & time
    date_soup = soup.findAll('div', {'class': 'message-attribution-main'})
    date_list = []
    time_list = []
    #date and time is found in the '<time>' tag. In the time tage there is a variable called "datetime" that is = to 
    #date of the post and formatted as '2020-09-24T18:28:56-0400', I store this datetime as a list object and slice it
    #so that I get only '2020-09-24' for date and 18:28:56 for time.
    for item in date_soup:
        date = item.find('time').attrs['datetime'][0:10]
        time_ = item.find('time').attrs['datetime'][11:19]
        date_list.append(date)
        time_list.append(time_)

    #find the comments 
    comments = soup.findAll("div", {"class": "bbWrapper"})

    #return only the text from the comments. 'recursive=False' prevents parsing any sub-tags. All needed text is a 
    #direct child of -> "div", {"class": "bbWrapper"}
    comment_list = []
    for comment in comments:
        comment=comment.find_all(text=True, recursive=False)
        comment = ''.join(comment) #convert list to string
        comment = comment.replace('\n', '') #remove new lines for paragraphs(Combines multiple paragraphs to one)
        if not comment: #if comment is empty
            comment = 'N/A'
        comment_list.append(comment)
    
    #use sentimentAnalyzer for each comment and create list of the 'compound' score for each comment
    compound_result_list = []
    for comment in comment_list:
        sentiment_result_dict = sid.polarity_scores(comment)
        compound_result = sentiment_result_dict.get('compound')
        compound_result_list.append(compound_result)
    
    #combine five lists and convert to DataFrame
    new_dict = zip(username_list, date_list, time_list, compound_result_list, comment_list)
    df = DataFrame(new_dict)
    
    #Write dataFrame to csv in append mode with the header removed
    df.to_csv(csv_file_path, index = False, mode = "a", header=False, encoding='utf-8-sig')
    
    #if there are less than 50 usernames it means it is the last page and should break
    if len(username_list) < 50:
        print('*FINISHED*')
        break
    else:
        forum_thread_page_num = forum_thread_page_num + 1

#positive sentiment: compound score >= 0.05
#neutral sentiment: (compound score > -0.05) and (compound score < 0.05)
#negative sentiment: compound score <= -0.05


# In[ ]:





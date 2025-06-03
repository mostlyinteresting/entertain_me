#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 21:02:31 2025

@author: ethan
"""

import pandas as pd

genome = pd.read_csv('/Users/ethan/entertain_me/backend_helpers/ml_25m/genome-scores.csv')

genome.info()
genome['new_column'] = genome['movieId']+genome['tagId']

tags = pd.read_csv('/Users/ethan/entertain_me/backend_helpers/ml_25m/tags.csv')

tags.info()
tags['tag'].nunique()
tags.groupby('userId')['tag'].nunique()
tags.head(1).T

tags.groupby('movieId')['tag'].nunique()

tags.groupby('movieId')['userId'].nunique()

rating = pd.read_csv('/Users/ethan/entertain_me/backend_helpers/ml_25m/ratings.csv')
rating.info()

tag_ratings= rating.merge(tags,how='inner',on=['userId','movieId'])
tag_ratings.info()


tag_ratings.groupby('tag')['rating'].mean()
# math operations with columns
    # sum movieId and tagId in genome df
    # subtract movieId and tagId in genome df
    # divide movieId and tagId in genome df
    # multiply movieId and tagId in genome df

# group by
    # how many tags by user
    # how many tags by movie
    # how many users by movie
    
# join/merge
    # join / merge ratings with tags on MovieID and get the average rating by 
    # tag

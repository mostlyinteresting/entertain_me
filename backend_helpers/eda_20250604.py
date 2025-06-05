#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 20:34:58 2025

@author: ethan
"""

import pandas as pd
ratings = pd.read_csv('/Users/ethan/entertain_me/backend_helpers/ml_25m/ratings.csv')

movie_ratings=ratings.groupby('movieId')['rating'].nunique()

avg_movie_rating = movie_ratings.mean()

def rating_function(x):
    return x > avg_movie_rating

rating_function(x=5)
rating_function(x=4)

movie_ratings.apply(rating_function)


# load ratings file
# get the number of ratings per movie
# get the average number of ratings per movie
# write a function to say if the number of ratings for a given movie
# is more or less than the average number


# apply this function to the whole data set and store these
# values in a new column called 'many_reviews'
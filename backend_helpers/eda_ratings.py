#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 19:34:45 2025

@author: ethan
"""

import pandas as pd
ratings = pd.read_csv("/Users/ethan/entertain_me/backend_helpers/ml_25m/ratings.csv")

ratings.info()

ratings.head(1).T
ratings['rating']
ratings[['rating','timestamp']]
ratings['rating'].describe()
ratings['rating'].mean()
ratings['rating'].max()
ratings['rating'].min()
ratings['rating'].median()
ratings['rating'].hist()


# math operations with columns
# group by
# join/merge
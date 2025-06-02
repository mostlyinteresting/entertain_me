#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 31 20:08:57 2025

Data from https://datasets.imdbws.com/

@author: ethan
"""

import pandas as pd
import csv

file_path = "TMDB_movie_dataset_v11.csv" # https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies?utm_source=chatgpt.com
working_dir = "/Users/ethan/entertain_me/backend_helpers/ml_25m"

# ------------- MovieLens 25M -------------
ml_movies   = pd.read_csv(f"{working_dir}/movies.csv")        # movieId, title, genres, tmdbId
ratings     = pd.read_csv(f"{working_dir}/ratings.csv", usecols=["movieId", "rating"])
genome      = pd.read_csv(f"{working_dir}/genome-scores.csv") # movieId, tagId, relevance
tags        = pd.read_csv(f"{working_dir}/tags.csv")          # user tags (optional)


ml_movies.shape
ml_movies.info()
ml_movies.head(5)
ml_movies.head(1).T
ml_movies.tail(5)
ml_movies.tail(1).T
ml_movies.loc[10]
ml_movies.iloc[10]


"""
df.describe() — Summary statistics

df.dtypes — Data types of each column

df.columns / df.index — Get column/index names

df.sample(n) — Random sample of rows

🧼 Data Cleaning
df.isnull() / df.notnull() — Check for missing values

df.dropna() — Drop missing values

df.fillna(value) — Fill missing values

df.duplicated() / df.drop_duplicates() — Check/drop duplicates

df.replace(old, new) — Replace values

df.astype(type) — Convert data types

df.rename(columns={}) — Rename columns

🔍 Filtering and Selection
df['col'] / df[['col1', 'col2']] — Select columns

df.loc[row_label, col_label] — Label-based selection

df.iloc[row_index, col_index] — Integer position selection

df.query('col > 5') — SQL-like filtering

Boolean indexing: df[df['col'] > 5]

🔄 Sorting and Reordering
df.sort_values(by='col', ascending=True)

df.sort_index()

🔧 Transformations
df.apply(func) — Apply function row/column-wise

df.map(func) — Element-wise transformation (Series only)

df.assign(new_col=lambda df: ...) — Add columns in chain

df.drop(columns=['col1', 'col2']) — Drop columns

🧮 Aggregations and Grouping
df.groupby('col') — Group by column(s)

.agg({'col1': 'mean', 'col2': 'sum'}) — Aggregate

df.pivot_table(index=..., columns=..., values=..., aggfunc=...)

📊 Combining DataFrames
pd.concat([df1, df2], axis=0) — Stack vertically

pd.merge(df1, df2, on='col') — SQL-style join

df.join(other_df) — Join on index or key

🧠 Time Series (if applicable)
pd.to_datetime(df['date']) — Convert to datetime

df.set_index('date') — Set datetime index

df.resample('W').mean() — Resample time series

df.shift(1) — Lag

df.rolling(window=3).mean() — Rolling window

📤 Exporting
df.to_csv('file.csv', index=False)

"""



# pre-aggregate ratings so we don’t carry 25 M rows around
rating_stats = ratings.groupby("movieId").agg(
    ml_mean  = ("rating", "mean"),
    ml_cnt   = ("rating", "size")
)

# ------------- TMDB bulk dump -------------
tmdb = pd.read_csv(
    file_path,              # e.g. "tmdb_movies.csv"  or  ".csv.gz"
    engine="pyarrow",       # fast & memory-friendly
    compression="infer",    # handles the .gz automatically
    na_values=["", "null"], # TMDB uses empty strings for null
    dtype_backend="pyarrow" # optional: keeps RAM low
)

ml_movies["tmdbId"] = (
    pd.to_numeric(ml_movies["movieId"], errors="coerce")   # some are empty = NaN
      .astype("Int64")                                   # pandas nullable int
)

tmdb = tmdb.rename(columns={"id": "tmdbId"})              # now both frames agree
tmdb["tmdbId"] = tmdb["tmdbId"].astype("Int64")

# optional: keep one row per TMDB id if the dump has duplicates
tmdb = tmdb.drop_duplicates(subset="tmdbId")

# ─────────── ratings + metadata merge ───────────
df = (ml_movies
      .merge(rating_stats,  on="movieId", how="left")
      .merge(tmdb,          on="tmdbId",  how="inner")    # <- single key now
)


from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

df = df.drop(columns=['genres_y']).rename(columns={'genres_x':'genres'})
df = df.drop(columns=['title_y']).rename(columns={'title_x':'title'})

# text to embed = overview + genres
texts = (df["overview"].fillna("") + " Genres: " + 
         df["genres"].fillna("").str.replace("|", " ")).tolist()

model = SentenceTransformer("all-MiniLM-L6-v2", device="mps")

emb = model.encode(texts, batch_size=256, show_progress_bar=True,
                   normalize_embeddings=True).astype("float32")  # (N,768)

# Optional: append Tag-Genome mood vector (≈1 129 dims) scaled down
def make_genome_vec(movie_id):
    sub = genome[genome.movieId == movie_id]
    vec = np.zeros(1129, dtype="float32")
    vec[sub.tagId.values] = sub.relevance.values
    return vec

genome_mat = np.vstack([make_genome_vec(mid) for mid in df.movieId])
genome_mat = (genome_mat / np.linalg.norm(genome_mat, axis=1, keepdims=True))

# final item vector = [txt | 0.2 * genome]
item_vecs = np.hstack([emb, 0.2 * genome_mat]).astype("float32")


def recommend(sentence_mood:str,
              want_tv:bool=False,
              k:int=10,
              rating_floor:float=3.2,
              alpha:float=0.7):
    """
    alpha balances mood similarity (content) vs quality.
    """
    # 1. get candidate set of matching type + rating
    # mask = ((df["media_type"] == ("tv" if want_tv else "movie")) &
    #         (df["weighted_rating"] >= rating_floor))
    mask = (df["weighted_rating"] >= rating_floor)
    cand_ids  = np.where(mask)[0]
    if not len(cand_ids):
        return []

    # 2. embed the mood sentence
    q_vec = model.encode(sentence_mood, normalize_embeddings=True)

    # 3. similarity search inside candidate subset
    sub_index = faiss.IndexFlatIP(item_vecs.shape[1])
    sub_index.add(item_vecs[cand_ids])
    sims, idxs = sub_index.search(q_vec[None], 500)   # top-500 raw

    # 4. re-score with rating boost
    sims = sims.flatten()
    rating_norm = (df.loc[cand_ids, "weighted_rating"] - 2.5) / 2.5  # 0-1
    final_score = alpha * sims + (1-alpha) * rating_norm.values
    top = final_score.argsort()[::-1][:k]

    return df.iloc[cand_ids[top]][[
        "title", "first_air_date", "overview",
        "poster_path", "weighted_rating"
    ]]

recommend(
    sentence_mood="I feel nostalgic and need something light-hearted",
    want_tv=False, k=5
)

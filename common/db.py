from typing import List

import pandas as pd
from pymongo import MongoClient

from common.constants import *

client = MongoClient(MONGODB_CONNECTION)
db = client[MONGODB_PROJECT]
complaints_collection = db['complaints']


def find_complaints(keywords: List[str] = []):
  filtering = {C_KEYWORDS: {'$in': keywords}} if keywords else {}
  data = complaints_collection.find(filtering)
  df = pd.DataFrame.from_dict(list(data))
  df[[C_LATITUDE, C_LONGITUDE]] = df[[C_LATITUDE, C_LONGITUDE]].apply(pd.to_numeric)
  df = df.rename(columns={'_id': C_REPORT_ID}).set_index(C_REPORT_ID)
  return df


def get_complaints_from_ids(ids: List[str]):
  data = complaints_collection.find({'_id': {'$in': ids}})
  df = pd.DataFrame.from_dict(list(data))
  df[[C_LATITUDE, C_LONGITUDE]] = df[[C_LATITUDE, C_LONGITUDE]].apply(pd.to_numeric)
  df = df.rename(columns={'_id': C_REPORT_ID}).set_index(C_REPORT_ID)
  return df


def get_clustering(threshold: str, percentages: List[str]):
  collection = db['clustering']
  query = {'threshold': threshold, 'weights': weights}
  documents = list(collection.find(query))
  if documents:
    return documents

  similarities = get_similarities(df, weights)
  df['cluster'] = autocluster(similarities, threshold)
  documents = []
  for _, row in df_cluster.iterrows():
    document = {'threshold': threshold, 'weights': weights, 'report_id': row['report_id'], 'cluster': row['cluster']}
    documents.append(document)
  collection.insert_many(documents)
  return cluster

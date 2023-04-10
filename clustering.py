import io
from typing import List

import more_itertools
import pinecone
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from functools import partial
import gridfs

from common.utils import *
from common.gpt_actions import *
from common.constants import *
from common.db import db

THRESHOLD = 2
WEIGHTS = [0.45, 0.45, 0.1]
MAX_IDS_PER_REQUEST = 1000
FILE_SIMILARITIES = 'similarities_{}.npy'


def fetch_pinecone(ids: List[str]) -> pd.DataFrame:
  index = pinecone.Index(PINECONE_INDEX_NAME)
  contents = []
  for id_chunk in more_itertools.chunked(ids, MAX_IDS_PER_REQUEST):
    content = index.fetch(id_chunk, namespace='complaints')
    contents.extend(content['vectors'].values())
  vector_data = [
      {C_REPORT_ID: x['id'], "vector": np.array(x['values'])}
      for x in contents if x and x['values']
  ]
  df_vectors = pd.DataFrame(vector_data).set_index(C_REPORT_ID)
  return df_vectors


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
  """
  Remove outliers from a DataFrame based on latitude and longitude using the IQR method.
  """
  # Calculate the IQR of the latitude column
  lat_Q1 = df[C_LATITUDE].quantile(0.25)
  lat_Q3 = df[C_LATITUDE].quantile(0.75)
  lat_IQR = lat_Q3 - lat_Q1

  # Calculate the IQR of the longitude column
  lon_Q1 = df[C_LONGITUDE].quantile(0.25)
  lon_Q3 = df[C_LONGITUDE].quantile(0.75)
  lon_IQR = lon_Q3 - lon_Q1

  # Define the upper and lower bounds for outlier detection
  lat_upper_bound = lat_Q3 + 1.5 * lat_IQR
  lat_lower_bound = lat_Q1 - 1.5 * lat_IQR
  lon_upper_bound = lon_Q3 + 1.5 * lon_IQR
  lon_lower_bound = lon_Q1 - 1.5 * lon_IQR

  # Remove outliers based on latitude and longitude
  df = df[(df[C_LATITUDE] > lat_lower_bound) & (df[C_LATITUDE] < lat_upper_bound) &
          (df[C_LONGITUDE] > lon_lower_bound) & (df[C_LONGITUDE] < lon_upper_bound)]

  return df


def coordinates_similarity(coordinates: np.ndarray) -> np.ndarray:
  # Compute Euclidean distances between all pairs of coordinates
  coord_distances = np.sqrt(
      ((coordinates[:, None, :] - coordinates) ** 2).sum(axis=2))
  return 1 - coord_distances / np.max(coord_distances)


def time_similarity(dates: np.ndarray) -> np.ndarray:
  # Compute the time differences between all pairs of timestamps
  timestamps = np.array([x.timestamp() for x in dates])
  time_diffs = np.abs(timestamps[:, None] - timestamps)
  return 1 - time_diffs / np.max(time_diffs)


def combine_similarities(similarities: list, weights: list) -> list:
  if len(similarities) != len(weights):
    raise Exception('Similarities and weights must be of the same length')

  return sum([x * y for x, y in zip(weights, similarities)])


def autocluster(similarity: np.ndarray, threshold: float) -> np.ndarray:
  # Create AgglomerativeClustering instance
  model = AgglomerativeClustering(
      n_clusters=None, linkage='ward', distance_threshold=threshold)
  # Fit the model to the data
  model.fit(similarity)
  # Extract cluster labels
  return model.labels_.astype(str)


def subcluster_by_embed(df: pd.DataFrame) -> pd.DataFrame:
  df_temp = groupby(df.reset_index(), 'cluster_balance', dict(), list)
  df_temp['subcluster'] = df_temp.apply(
      lambda x: autocluster(x['vector'], np.stack(
          (x['latitude'], x['longitude'])).T, THRESHOLD, 0),
      axis=1)
  df_temp = df_temp.reset_index()
  df_temp['report_id'] = df_temp['report_id'].apply(np.array)
  clustering_data = df_temp.apply(
      lambda x: [
          dict(report_id=id, final=f"{x['cluster_balance']}-{str(a)}")
          for id, a in zip(x['report_id'], x['subcluster'])
      ], axis=1).to_numpy()
  clustering_data = flatten(clustering_data)
  df_temp = pd.DataFrame(clustering_data).set_index('report_id')
  df = df.join(df_temp)
  return df


def compute_similarities(
    df: pd.DataFrame,
    weights: List[float]
) -> np.ndarray:
  ids = df.index.to_list()
  df_vectors = fetch_pinecone(ids)
  df = df.join(df_vectors)
  vectors = df['vector'].apply(np.array)
  embeddings = vectors.to_list()
  coordinates = df[['latitude', 'longitude']].to_numpy()
  dates = pd.to_datetime(df[C_DATE], format='%Y-%m-%d %H:%M:%S')
  similarities = [
      cosine_similarity(embeddings),
      coordinates_similarity(coordinates),
      time_similarity(dates)
  ]
  return combine_similarities(similarities, weights)


def get_similarities(df: pd.DataFrame, weights: List[float]) -> np.ndarray:
  similarities = np.array([])
  weights_name = '_'.join([str(int(x * 100)) for x in weights])
  filename = FILE_SIMILARITIES.format(weights_name)
  compute_array = partial(compute_similarities, df, weights)
  similarities = save_npy_to_gridfs(compute_array, 'similarities', filename)
  return similarities


def save_npy_to_gridfs(compute_array, collection_name, file_name):
  fs = gridfs.GridFS(db, collection=collection_name)

  file_doc = fs.find_one({'filename': file_name})
  if file_doc is None:
    npy_array = compute_array()
    bytes_io = io.BytesIO()
    np.save(bytes_io, npy_array)
    npy_bytes = bytes_io.getvalue()
    with fs.new_file(filename=file_name) as fp:
      fp.write(npy_bytes)
      file_doc = fp
  contents = fs.get(file_doc._id).read()
  npy_array = np.load(io.BytesIO(contents))
  return npy_array


def df_clustering(df: pd.DataFrame, threshold: float, weights: List[float]) -> pd.DataFrame:
  collection = db['clustering']
  query = {'threshold': threshold, 'weights': weights}
  documents = list(collection.find(query))
  if documents:
    new_df = pd.DataFrame(documents).set_index('report_id')
    df['cluster'] = new_df.loc[df.index]['cluster']
    return df

  similarities = get_similarities(df, weights)
  df['cluster'] = autocluster(similarities, threshold)
  documents = []
  for index, row in df.iterrows():
    document = {'threshold': threshold, 'weights': weights, 'report_id': index, 'cluster': row['cluster']}
    documents.append(document)
  collection.insert_many(documents)
  return df


def plot(df: pd.DataFrame, similarities: List, dates: List):
  tsne = TSNE(n_components=2, perplexity=15, random_state=42,
              init='random', learning_rate=200)

  ids = df.index.to_list()
  df_vectors = fetch_pinecone(ids)
  matrix = np.vstack(df_vectors['vector'])
  timestamps = np.array([x.timestamp() for x in dates])
  time_diffs = np.abs(timestamps[:, None] - timestamps)

  vis_dims = tsne.fit_transform(similarities)
  vis_dims2 = tsne.fit_transform(matrix)
  vis_dims3 = tsne.fit_transform(time_diffs)

  df_temp = df.reset_index()
  fig, axes = plt.subplots(2, 2, figsize=(12, 12))
  le = LabelEncoder()
  x1 = [x for x, y in vis_dims]
  y1 = [y for x, y in vis_dims]
  x2 = df_temp['latitude'].to_list()
  y2 = df_temp['longitude'].to_list()
  x3 = [x for x, y in vis_dims2]
  y3 = [y for x, y in vis_dims2]
  x4 = [x for x, y in vis_dims3]
  y4 = [y for x, y in vis_dims3]
  cluster = df_temp['cluster'].to_list()
  cluster = list(map(int, cluster))

  axes[0, 0].scatter(x1, y1, c=cluster, cmap='tab10')
  axes[0, 0].set_title('Cluster (Balance)')

  axes[0, 1].scatter(x2, y2, c=cluster, cmap='tab10')
  axes[0, 1].set_title('Cluster (Distance)')

  axes[1, 0].scatter(x3, y3, c=cluster, cmap='tab10')
  axes[1, 0].set_title('Cluster (Embedding)')

  axes[1, 1].scatter(x4, y4, c=cluster, cmap='tab10')
  axes[1, 1].set_title('Cluster (Time)')

  plt.tight_layout()
  plt.show()


if __name__ == '__main__':
  df = file_to_df(FILE_CITIZEN_REPORTS).set_index(C_REPORT_ID)
  df = remove_outliers(df)
  similarities = get_similarities(WEIGHTS)
  df['cluster'] = autocluster(similarities, THRESHOLD)
  dates = pd.to_datetime(df[C_DATE], format='%Y-%m-%d %H:%M:%S')

  plot(df, similarities, dates)

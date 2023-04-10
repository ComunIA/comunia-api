from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

from common.constants import *
from common.utils import *
from common.db import find_complaints, get_complaints_from_ids, complaints_collection, reports_collection
from common.gpt_actions import summarize_complaints, question_complaints, complaints_db
from clustering import df_clustering

NUM_KNOWLEDGE_RESULTS = 3
CLUSTER_PERCENTAGES = [0.4, 0.6, 0]
INITIAL_THRESHOLD = 2


class Filtering:
  problem: Optional[str]
  key_words: List[str]
  threshold: float
  percentages: List[float]

  def __init__(self, problem=None, key_words=[], threshold=INITIAL_THRESHOLD, percentages=CLUSTER_PERCENTAGES):
    self.problem = problem
    self.key_words = key_words
    self.threshold = threshold
    self.percentages = percentages


messages: Dict[str, List[Tuple[str, str]]] = {}


def get_complaints_score(query: str, amount: int, vector_filter=None) -> pd.DataFrame:
  docs = complaints_db.similarity_search_with_score(query, amount, vector_filter, 'complaints')
  results = [{C_REPORT_ID: doc.metadata[C_REPORT_ID], "score": score} for doc, score in docs]
  df_score = pd.DataFrame(results).set_index(C_REPORT_ID)
  df_score[C_SCORE] = normalize(df_score[C_SCORE])
  return df_score


def score_complaints(df: pd.DataFrame, filtering: Filtering) -> pd.DataFrame:
  if filtering.problem:
    vector_filter = {C_KEYWORDS: {'$in': filtering.key_words}} if filtering.key_words else None
    df_score = get_complaints_score(filtering.problem, len(df), vector_filter)
    valid_indices = df_score.index.intersection(df.index)
    df = df.loc[df_score.loc[valid_indices].index]
    df[C_SCORE] = df_score[C_SCORE]
  else:
    df[C_SCORE] = None
  return df


def grouping_community(df: pd.DataFrame) -> pd.DataFrame:
  df_temp = groupby(df.reset_index(), 'cluster', {
      C_LATITUDE: np.mean,
      C_LONGITUDE: np.mean,
      C_KEYWORDS: lambda x: list(set(x)),
  })
  df_temp['num_reports'] = df_temp[C_COMPLAINT].map(len)
  return df_temp


def get_answer(question: str, report_ids: List[str]) -> str:
  id = ids_to_hash(report_ids)
  vector_filter = {C_REPORT_ID: {'$in': report_ids}}
  if id not in messages:
    messages[id] = []
  answer_complaints = question_complaints(question, messages[id], vector_filter)
  messages[id].append((question, answer_complaints))
  return answer_complaints.strip()


def generate_report(report_ids: List[str]) -> str:
  id = ids_to_hash(report_ids)
  document = reports_collection.find_one({'_id': id})
  if document:
    return document['report']
  complaints = get_complaints_from_ids(report_ids)['complaint']
  report = summarize_complaints(complaints)
  reports_collection.insert_one({'_id': id, 'report': report, 'report_ids': report_ids})
  return report

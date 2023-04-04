import numpy as np
import pandas as pd
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from typing import Dict, Tuple, List, Optional

from common.constants import *
from common.gpt_actions import summarize_complaints, question_complaints, complaints_db
from common.utils import normalize, file_to_df, groupby, to_text_list, save_yaml, ids_to_hash
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
filtering = Filtering()
df_complaints = file_to_df(FILE_CITIZEN_REPORTS, C_REPORT_ID)
df_processed = df_clustering(df_complaints, filtering.threshold, CLUSTER_PERCENTAGES)


def get_complaints_score(query: str, amount: int, vector_filter=None) -> pd.DataFrame:
  docs = complaints_db.similarity_search_with_score(query, amount, vector_filter, 'complaints')
  results = [dict(report_id=doc.metadata['report_id'], score=score) for doc, score in docs]
  df_score = pd.DataFrame(results).set_index('report_id')
  df_score[C_SCORE] = normalize(df_score[C_SCORE])
  return df_score


def filter_complaints(new_filtering: Filtering) -> pd.DataFrame:
  global df_processed
  global filtering
  prev_threshold = filtering.threshold
  filtering = new_filtering
  if prev_threshold != filtering.threshold:
    df_processed = df_clustering(df_complaints, filtering.threshold, filtering.percentages)
  df_temp = df_processed
  vector_filter = None
  if filtering.key_words:
    key_words = [np.nan if x == 'Otro' else x for x in filtering.key_words]
    df_temp = df_temp[df_temp[C_KEYWORDS].isin(key_words)]
    vector_filter = {C_KEYWORDS: {'$in': key_words}}
  if filtering.problem:
    df_score = get_complaints_score(filtering.problem, len(df_temp), vector_filter)
    df_temp = df_temp.loc[df_score.index]
    df_temp[C_SCORE] = df_score[C_SCORE]
  else:
    df_temp[C_SCORE] = None
  return df_temp


def grouping_community(df: pd.DataFrame) -> pd.DataFrame:
  df_temp = groupby(df.reset_index(), 'cluster', dict(
      latitude=np.mean,
      longitude=np.mean,
      alias=lambda x: list(set(x)),
      issue=lambda x: list(set(x))
  ))
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
  df_community_reports = file_to_df(FILE_NEIGHBORHOOD_REPORTS, 'id')
  if not df_community_reports.empty and id in df_community_reports.index:
    return df_community_reports.loc[id]['report']

  complaints = df_complaints.loc[report_ids]['complaint']
  report = summarize_complaints(complaints)
  reports = []
  if not df_community_reports.empty:
    reports = df_community_reports.reset_index().to_dict('records')
  reports.append({'id': id, 'report': report, 'report_ids': report_ids})
  save_yaml(FILE_NEIGHBORHOOD_REPORTS, reports)
  return report

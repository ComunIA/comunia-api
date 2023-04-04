import os
import hashlib
import numpy as np
import pandas as pd
from yaml import dump, safe_load
from pathlib import Path
import json
from typing import List

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


def ids_to_hash(ids: List[str]) -> str:
  ids = ','.join(list(sorted(set(ids))))
  return hashlib.sha256(ids.encode('utf-8')).hexdigest()


def read_file(filepath: str) -> str:
  with open(filepath, 'r', encoding='utf-8') as f:
    return f.read()


def read_yaml(filepath: str) -> dict:
  with open(filepath, 'r', encoding='utf-8') as f:
    return safe_load(f.read())


def save_file(filepath: str, content: dict):
  with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)


def save_yaml(filepath: str, data: dict):
  with open(filepath, 'w', encoding='utf-8') as f:
    dump(data, f, allow_unicode=True)


def similarity(v1, v2):  # return dot product of two vectors
  return np.dot(v1, v2)


def flatten(a_list):
  return [item for sublist in a_list for item in sublist]


def get_embedded(file, column_name='vector'):
  df = pd.read_csv(file)
  df[column_name] = df[column_name].apply(eval).apply(np.array)
  return df


def normalize(series):
  return (series - series.min()) / (series.max() - series.min())


def rename_columns(df, column_dict):
  selected_cols = [col for col in df.columns if col in column_dict.keys()]
  renamed_cols = {col: column_dict[col] for col in selected_cols}
  return df[selected_cols].rename(columns=renamed_cols)


def file_to_df(file_path, index_name=None):
  file_path = Path(file_path)
  file_extension = file_path.suffix.lower()

  if file_extension == '.csv':
    df = pd.read_csv(file_path)
  elif file_extension == '.json':
    with open(file_path, 'r') as f:
      data = json.load(f)
    df = pd.DataFrame.from_dict(data)
  elif file_extension in ('.yaml', '.yml'):
    with open(file_path, 'r') as f:
      data = safe_load(f)
    df = pd.DataFrame.from_dict(data)
  else:
    raise ValueError(f"Unsupported file format: {file_extension}")
  if not df.empty and index_name:
    df = df.set_index(index_name)
  return df


def groupby(df, column_name, columns_group, default_func=list):
  aggregate_grouping = dict(
      **columns_group,
      **{x: default_func for x in set(df.columns) - set(list(columns_group.keys()) + [column_name])}
  )
  df = df.groupby(column_name).agg(aggregate_grouping)
  return df


def to_text_list(list_a):
  return '\n'.join([f'- {x}' for i, x in enumerate(list_a)])


def as_messages(chat_prompts, **data):
  if len(chat_prompts) == 0:
    raise Exception("No chat messages provided.")
  if len(chat_prompts) % 2 != 0:
    raise Exception("The list of chat messages must be even.")

  role_types = {
      'system': SystemMessage,
      'user': HumanMessage,
      'assistant': AIMessage,
  }
  messages = []
  messages.append({"role": "system", "content": chat_prompts[0]})
  for i, x in enumerate(chat_prompts[1:]):
    role = "user" if (i % 2) == 0 else "assistant"
    messages.append({"role": role, "content": x})
  messages = [{'role': y['role'], 'content': y['content'].format(**data)} for y in messages]
  messages = [role_types[x['role']](content=x['content']) for x in messages]
  return messages


def replace_dict(my_dict, replacements):
  return {
      k2: my_dict.get(k1)
      for k1, k2 in replacements.items()
  }


def chunk_list(lst: list, chunk_size: int) -> list:
  result = []
  sublist = []
  for item in lst:
    if sum([len(x) for x in sublist]) + len(item) > chunk_size:
      result.append(sublist)
      sublist = []
    sublist.append(item)
  if sublist:
    result.append(sublist)
  return result

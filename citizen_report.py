import os
import warnings

import requests
import pandas as pd
from yaml import safe_load

from common.gpt_actions import chat_completion
from common.utils import rename_columns, save_yaml, replace_dict, groupby
from common.constants import *

warnings.filterwarnings('ignore')

COLUMNS_REPLACE = {
    'reporteId': C_REPORT_ID,
    'fecha': C_DATE,
    'sector': C_SECTOR,
    'calle': C_LOCATION,
    'colonia': C_NEIGHBORHOOD,
    'explicacion': C_COMPLAINT,
    'latitud': 'latitude',
    'longitud': 'longitude',
    'alias': 'alias',
    'asunto': 'issue',
}
PROCESSED_REPLACEMENTS = dict(
    problemas_ciudadano='citizen_problems',
    propuestas_ciudadano='citizen_proposals',
    tipo_transporte='transport_type',
    afectados='affected',
    ubicaciones='location',
    horario='time',
    frecuencia='frequency',
)


def cleanup_neighborhood(row):
  cleaned_neighborhood = row.replace(
      '(ASENTAMIENTO IRREGULAR)', '').strip().title()
  return f'{cleaned_neighborhood}, San Pedro Garza García, Nuevo Leon, Mexico'


def get_location(address):
  params = dict(
      address=address,
      sensor='false',
      key=GOOGLE_API_KEY
  )
  req = requests.get(GOOGLE_MAPS_API_URL, params=params)
  result = req.json()['results'][0]

  return dict(
      latitude=result['geometry']['location']['lat'],
      longitude=result['geometry']['location']['lng'],
      address=result['formatted_address'],
  )


def add_neighborhood_info(df):
  df = df[df[C_NEIGHBORHOOD].notnull()]
  df[C_NEIGHBORHOOD] = df[C_NEIGHBORHOOD].apply(cleanup_neighborhood)
  # df = pd.concat([df, df[C_NEIGHBORHOOD].apply(
  #     get_location).apply(pd.Series)], axis=1)
  return df


with open("data/aliases.txt", 'r', encoding='utf-8') as f:
  aliases = f.read().splitlines()

if __name__ == '__main__':
  df = pd.read_csv(FILE_CITIZEN_COMPLAINTS)
  df = rename_columns(df, COLUMNS_REPLACE)
  df[C_REPORT_ID] = df[C_REPORT_ID].astype(str)
  df = add_neighborhood_info(df)
  df = df[(df[C_COMPLAINT] != '') & (~df[C_COMPLAINT].isna())]
  df = df.fillna('')

  df = df[df['alias'].isin(aliases)]
  df[C_DATE] = pd.to_datetime(df[C_DATE], format='%d/%m/%Y %H:%M:%S')
  df = df.sort_values(C_DATE, ascending=False)
  df = df[df[C_DATE].dt.strftime('%Y-%m-%d') >= "2023-01-01"]
  df[C_DATE] = df[C_DATE].astype(str)

  # df = groupby(df, 'alias', dict()).reset_index()
  # df['size'] = df[C_COMPLAINT].apply(len)
  # df = df.sort_values('size', ascending=False)
  # a = df[['alias', 'size']].to_dict('records')
  # print(', '.join([x['alias'] for x in a]))
  # for x in a:
  #   print(x['alias'], x['size'])
  # df[list(PROCESSED_REPLACEMENTS.values())] = df[C_COMPLAINT].apply(
  #     get_complaint_data).apply(pd.Series)
  report_data = df.to_dict(orient='records')

  save_yaml(FILE_CITIZEN_REPORTS, report_data)

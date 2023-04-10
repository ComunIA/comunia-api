from bs4 import BeautifulSoup
import pandas as pd

import requests
from geopy.geocoders import Nominatim

from common.constants import *
from common.utils import rename_columns, save_yaml

FOLDER = 'data/csvs/CIAC'
COLUMNS = {
    "reporteId": C_REPORT_ID,
    "fecha": C_DATE,
    "latitud": C_LATITUDE,
    "longitud": C_LONGITUDE,
    "explicacion": C_COMPLAINT,
    "alias": C_KEYWORDS,
    "sector": C_SECTOR,
    "calle": C_LOCATION,
    "colonia": C_NEIGHBORHOOD,
}
KEYWORDS_FILTER = [
    'Luminarias',
    'Semáforo',
    'Rondines',
    'Violación al reglamento de tránsito',
    'Contaminación',
    'Baches',
    'Cables',
    'Banqueta',
    'Rotura',
    'Información de accidentes víales',
    'Pintura vial',
    'Estacionamiento',
    'Postes',
    'Quejas',
    'Parquímetros',
    'Licencias',
    'Multas',
    'Movilidad',
    'Parabuses',
    'Vehículos dañados por baches',
    'Banquetas',
    'Waze',
    'Puentes',
]


def extract_data(file: str) -> pd.DataFrame:
  with open(file, 'rb') as f:
    html_doc = f.read()

  soup = BeautifulSoup(html_doc, 'html.parser')
  table = soup.find('table')
  headers = []
  for th in table.find_all('th'):
    headers.append(th.text.strip())

  data = []
  for tr in table.find_all('tr'):
    if not tr.find_all('th'):
      row = []
      for td in tr.find_all('td'):
        row.append(td.text.strip())
      data.append(row)
  return pd.DataFrame(data, columns=headers)


def get_address_from_coords(lat, lon):
  geolocator = Nominatim(user_agent="comunia", timeout=10)
  location = geolocator.reverse(f"{lat}, {lon}")
  return location.address


def process_data(df: pd.DataFrame, date: str = None):
  df = rename_columns(df, COLUMNS)
  df = df[(df[C_COMPLAINT] != "") & (~df[C_COMPLAINT].isna())]
  df = df[df[C_KEYWORDS].isin(KEYWORDS_FILTER)]
  df = df.fillna("")
  # df[C_ADDRESS] = df[[C_LATITUDE, C_LONGITUDE]].apply(lambda x: get_address_from_coords(*x), axis=1)

  df[C_DATE] = pd.to_datetime(df[C_DATE], format="%d/%m/%Y %H:%M:%S")
  df = df.sort_values(C_DATE, ascending=False)
  if date:
    df = df[df[C_DATE].dt.strftime("%Y-%m-%d") >= date]
  df[C_DATE] = df[C_DATE].astype(str)
  return df


if __name__ == "__main__":
  df_final = None
  for filename in os.listdir(FOLDER):
    file = os.path.join(FOLDER, filename)
    df_new = extract_data(file)
    df_new['reporteId'] = df_new['reporteId'].astype(str)
    df_new = df_new.set_index('reporteId')
    if df_final is None:
      df_final = df_new
    else:
      df_final = df_final.combine_first(df_new)
  print('Size total:', len(df_final))
  df_final = process_data(df_final.reset_index(), "2023-01-01")
  print('Size processed:', len(df_final))

  save_yaml(FILE_CITIZEN_REPORTS, df_final.to_dict('records'))

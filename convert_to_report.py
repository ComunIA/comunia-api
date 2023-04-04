from bs4 import BeautifulSoup
import pandas as pd

from common.constants import *

FOLDER = 'data/csvs/CIAC'


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

  df_final.to_csv(FILE_CITIZEN_COMPLAINTS)

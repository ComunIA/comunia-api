import os
import json

from common.pdf_converter import pdf_to_list
from common.utils import save_yaml
from common.constants import *


if __name__ == "__main__":
  data = []
  for filename in os.listdir(DIR_PDF):
    if not filename.endswith('.pdf'):
      continue
    file = os.path.join(DIR_PDF, filename)
    name = os.path.splitext(filename)[0]
    meta_file = os.path.join(DIR_PDF, f'{name}.meta.json')
    text_chunks = pdf_to_list(file)
    text_chunks = [x.strip() for x in text_chunks if x.strip()]
    with open(meta_file, 'r') as f:
      meta_info = json.load(f)
    data.extend(dict(content=x, **meta_info) for x in text_chunks)
  save_yaml(FILE_URBAN_DOCUMENTS, data)

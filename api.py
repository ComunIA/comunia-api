import os
import json
from functools import wraps
from typing import Optional

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

from common.utils import ids_to_hash
from functionalities import *
from clustering import df_clustering
from common.db import find_complaints

PORT = 8000
MIN_GENERATE_CLUSTER = 3


app = Flask(__name__)
cors = CORS(app, origins=os.getenv('CORS_ORIGIN').split(','))
app.config['JSON_AS_ASCII'] = False
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['SECRET_KEY'] = os.getenv('API_SECRET_KEY')

API_KEYS = json.loads(os.getenv('API_KEYS'))


def api_key_required(f):
  @wraps(f)
  def decorated(*args, **kwargs):
    api_key = None
    if 'X-API-KEY' in request.headers:
      api_key = request.headers['X-API-KEY']
    if not api_key:
      return jsonify({'message': 'API key is missing'}), 401
    if api_key not in API_KEYS:
      return jsonify({'message': 'API key is invalid'}), 401
    return f(*args, **kwargs)
  return decorated


@app.route('/complaints', methods=['POST'])
@api_key_required
def get_complaints():
  data = request.get_json()
  filtering = Filtering(**data['filtering'])
  df = find_complaints(filtering.key_words)
  df = score_complaints(df, filtering)
  df = df.reset_index()
  return jsonify(df.to_dict('records'))


@app.route('/reports', methods=['POST'])
@api_key_required
def get_reports():
  data = request.get_json()
  filtering = Filtering(**data['filtering'])
  df = find_complaints()
  df = df_clustering(df, filtering.threshold, filtering.percentages)
  if filtering.key_words:
    df = df[df[C_KEYWORDS].isin(filtering.key_words)]

  df = score_complaints(df, filtering)
  df_reports = grouping_community(df)
  df_reports = df_reports.reset_index()
  return jsonify(df_reports.to_dict('records'))


@app.route('/reports/generate', methods=['POST'])
@api_key_required
def api_generate_report():
  data = request.get_json()
  report_ids = data['report_ids']
  if len(report_ids) < MIN_GENERATE_CLUSTER:
    return f'At least {MIN_GENERATE_CLUSTER} report ids are required to generate a report', 400
  report = generate_report(report_ids)
  return jsonify(report)


@app.route('/chat/history', methods=['POST'])
@api_key_required
def api_chat_history():
  data = request.get_json()
  report_ids = data['report_ids']
  id = ids_to_hash(report_ids)
  return jsonify(messages.get(id, []))


@app.route('/chat/message', methods=['POST'])
@api_key_required
def api_chat():
  data = request.get_json()
  message = data['message']
  report_ids = data['report_ids']
  answer = get_answer(message, report_ids)
  return jsonify(answer)


@app.route('/chat/clear', methods=['POST'])
@api_key_required
def api_chat_reset():
  data = request.get_json()
  report_ids = data['report_ids']
  id = ids_to_hash(report_ids)
  messages[id] = []
  return jsonify(messages[id])


if __name__ == '__main__':
  app.run(port=PORT, debug=True)

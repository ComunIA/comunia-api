import os
import json

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from typing import Optional
from functools import wraps
from jwt import decode, InvalidTokenError

from common.utils import ids_to_hash
from functionalities import filter_complaints, grouping_community, messages, get_answer, generate_report, Filtering

PORT = 8000
MIN_GENERATE_CLUSTER = 3


app = Flask(__name__)
cors = CORS(app, support_credentials=True)
# app.secret_key = os.getenv('API_SECRET_KEY')
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
@cross_origin()
@api_key_required
def get_complaints():
  data = request.get_json()
  new_filtering = Filtering(**data['filtering'])
  df_temp = filter_complaints(new_filtering)
  df_temp = df_temp.reset_index()
  return jsonify(df_temp.to_dict('records'))


@app.route('/reports', methods=['POST'])
@cross_origin()
@api_key_required
def get_reports():
  data = request.get_json()
  new_filtering = Filtering(**data['filtering'])
  df_temp = filter_complaints(new_filtering)
  df_reports = grouping_community(df_temp)
  df_reports = df_reports.reset_index()
  return jsonify(df_reports.to_dict('records'))


@app.route('/reports/generate', methods=['POST'])
@cross_origin()
@api_key_required
def api_generate_report():
  data = request.get_json()
  report_ids = data['report_ids']
  if len(report_ids) < MIN_GENERATE_CLUSTER:
    return f'At least {MIN_GENERATE_CLUSTER} report ids are required to generate a report', 400
  report = generate_report(report_ids)
  return jsonify(report)


@app.route('/chat/history', methods=['POST'])
@cross_origin()
@api_key_required
def api_chat_history():
  data = request.get_json()
  report_ids = data['report_ids']
  id = ids_to_hash(report_ids)
  return jsonify(messages.get(id, []))


@app.route('/chat/message', methods=['POST'])
@cross_origin()
@api_key_required
def api_chat():
  data = request.get_json()
  message = data['message']
  report_ids = data['report_ids']
  answer = get_answer(message, report_ids)
  return jsonify(answer)


@app.route('/chat/clear', methods=['POST'])
@cross_origin()
@api_key_required
def api_chat_reset():
  data = request.get_json()
  report_ids = data['report_ids']
  id = ids_to_hash(report_ids)
  messages[id] = []
  return jsonify(messages[id])


if __name__ == '__main__':
  app.run(port=PORT, debug=True)

```sh
python -m convert_to_report
```
```sh
python -m citizen_report
```
This script uses data from `data/csvs/congestionamiento-vial.csv` process the information and generates a yaml file: `data/jsons/citizen_reports.yaml`

```sh
python -m pinecone_embed --index-name 'citizen-complaints' --file 'data/jsons/citizen_reports.yaml' --column 'complaint' --index 'report_id'
python -m pinecone_embed --index-name 'citizen-complaints-2023' --file 'data/jsons/citizen_reports_2023.yaml' --column 'complaint' --index 'report_id'
```
This script generates the embeddings and uploads them to pinecone.

```sh
python -m process_documents
```

```sh
python -m pinecone_embed --index-name 'urban-planning-documents' --file 'data/jsons/urban_planning_documents.yaml' --column 'content'
```

```sh
bokeh serve visualize_reports.py
```
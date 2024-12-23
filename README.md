```sh
uvicorn app:app --host 0.0.0.0 --port 8000
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"text": "松友美佐紀は、日本のバドミントン選手。"}'
```
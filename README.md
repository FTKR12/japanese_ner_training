# Japanese NER Training, Evaluation, and REST API
## Training and Evaluation
training
```sh
python train.py
```
training log is on [here](./output/train/cl-tohoku/bert-base-japanese-whole-word-masking/log.txt).
evaluation
```sh
python test.py
```
evaluation result is on [here](./output/test/cl-tohoku/bert-base-japanese-whole-word-masking/log.txt).

## REST API
run
```sh
uvicorn app:app --host 0.0.0.0 --port 8000
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"text": "松友美佐紀は、日本のバドミントン選手。"}'
```
response
```sh
{"result":[["人名","松友美佐紀"],["日本","地名"]]}
```

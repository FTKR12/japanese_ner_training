from fastapi import FastAPI
from pydantic import BaseModel
import json
import torch
from transformers import BertForTokenClassification

from src import NerTokenizerForTest, NerTester

label2id = json.load(open("./dataset/ner_config.json", "r"))
id2label = {value: key for key, value in label2id.items()}
model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
device = "cpu"
tokenizer = NerTokenizerForTest.from_pretrained(model_name)
model = BertForTokenClassification.from_pretrained(model_name, num_labels=9)
model.load_state_dict(torch.load("./model/model.bin", weights_only=True))
model.eval()

tester = NerTester(
    device=device,
    args=None,
    label2id=label2id,
    dataset=None,
    tokenizer=tokenizer,
    model=model
)

app = FastAPI()

class TextRequest(BaseModel):
    text: str
    max_length: int = 128

@app.get("/")
async def root():
    return {"message": "now on running"}

# エンドポイントの定義
@app.post("/predict")
async def predict(request: TextRequest):
    result = tester.predict(request.text)
    for item in result:
        item["type_name"] = id2label[item["type_id"]]
    result = [{item["name"], item["type_name"]} for item in result]
    return {"result": result}
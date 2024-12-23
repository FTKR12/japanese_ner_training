import logging
logger = logging.getLogger("Japanese NER Training")

import torch
import pandas as pd
from prettytable import PrettyTable


class NerTrainer():

    def __init__(self, args, model, loader, optimizer):
        self.args = args
        self.model = model
        self.loader = loader
        self.optimizer = optimizer

        self.best_loss_val = 9999

    def run(self):
        torch.backends.cudnn.benchmark = True
        for epoch in range(self.args.epochs):
            self.model.train()
            epoch_loss_train = self.train_val("train")

            self.model.eval()
            epoch_loss_val = self.train_val("val")

            logger.info(f"[Epoch] {epoch+1}/{self.args.epochs}    [Phase] train    [Train Loss] {epoch_loss_train}    [Val Loss] {epoch_loss_val}")

        return 
    
    def train_val(self, phase: str):
        epoch_loss = 0.0

        for iter, batch in enumerate(self.loader[phase]):
            input_ids = batch["input_ids"].to(self.args.device)
            attention_mask = batch["attention_mask"].to(self.args.device)
            labels = batch["labels"].to(self.args.device)

            with torch.set_grad_enabled(phase == 'train'):
                self.optimizer.zero_grad()
                loss, _ = self.model(
                        input_ids=input_ids, 
                        token_type_ids=None, 
                        attention_mask=attention_mask, 
                        labels=labels,
                        return_dict=False
                )

                if phase == "train":
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                if (iter % 10 == 0):
                    logger.info(f"    {phase}    iter {iter}    loss: {loss:.4f}")
                
                epoch_loss += loss.item() * self.args.batch_size
        
        epoch_loss = epoch_loss / len(self.loader[phase].dataset)

        if phase == "val":
            if self.best_loss_val > epoch_loss:
                self.best_loss_val = epoch_loss
                torch.save(self.model.state_dict(), self.args.model_save_path)
                logger.info("saved!")

        return epoch_loss


class NerTester():

    def __init__(self, device, args, label2id, dataset, tokenizer, model):
        self.device = device
        self.args = args
        self.label2id = label2id
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.model = model

    def run(self):
        entities_list = []
        entities_predicted_list = []
        for sample in self.dataset:
            text = sample['text']
            entities_predicted = self.predict(text)
            entities_list.append(sample['entities'])
            entities_predicted_list.append( entities_predicted )
        
        eval_df = pd.DataFrame()
        for k, v in self.label2id.items():
            eval_res = self.evaluate(entities_list, entities_predicted_list, type_id=v)
            eval_df[k] = eval_res.values()

        eval_res_all = self.evaluate(entities_list, entities_predicted_list, type_id=None)
        eval_df["ALL"] = eval_res_all.values()

        eval_df.index = eval_res_all.keys()

        table=PrettyTable()
        table.field_names = ["Index"] + eval_df.columns.tolist()
        for idx, row in eval_df.iterrows():
            table.add_row([idx] + row.tolist())
        
        logger.info(f"\n{table}")

        return


    def predict(self, text):

        encoding, spans = self.tokenizer.encode_plus_untagged(text)
        encoding = { k: v.to(self.device) for k, v in encoding.items() }

        with torch.no_grad():
            output = self.model(**encoding)
            scores = output.logits
            labels_predicted = scores[0].argmax(-1).cpu().numpy().tolist() 
        
        print(labels_predicted)

        entities = self.tokenizer.convert_bert_output_to_entities(
            text, labels_predicted, spans
        )

        return entities

    def evaluate(self, entities_list, entities_predicted_list, type_id=None):
        num_entities = 0 
        num_predictions = 0 
        num_correct = 0 

        for entities, entities_predicted in zip(entities_list, entities_predicted_list):

            if type_id:
                entities = [ e for e in entities if e['type_id'] == type_id ]
                entities_predicted = [ 
                    e for e in entities_predicted if e['type_id'] == type_id
                ]
                
            get_span_type = lambda e: (e['span'][0], e['span'][1], e['type_id'])
            set_entities = set( get_span_type(e) for e in entities )
            set_entities_predicted = set( get_span_type(e) for e in entities_predicted )

            num_entities += len(entities)
            num_predictions += len(entities_predicted)
            num_correct += len( set_entities & set_entities_predicted )

        precision = num_correct/num_predictions 
        recall = num_correct/num_entities 
        f_value = 2*precision*recall/(precision+recall) 

        result = {
            'num_entities': num_entities,
            'num_predictions': num_predictions,
            'num_correct': num_correct,
            'precision': precision,
            'recall': recall,
            'f_value': f_value
        }

        return result
from typing import List, Dict
import itertools

from transformers import BertJapaneseTokenizer

class NerToknizer(BertJapaneseTokenizer):

    def tokenize_and_labeling(self, splitted: List):
        """
        Args:
            splitted: splitted text
        Return:
            tokens: list of token by tokenizing splitted text
            labels: list of label for each token
        """
        tokens = []
        labels = []
        for s in splitted:
            text = s["text"]
            label = s["label"]
            tokens_splitted = self.tokenize(text)
            labels_splitted = [label] * len(tokens_splitted)
            tokens.extend(tokens_splitted)
            labels.extend(labels_splitted)
            
        return tokens, labels


    def encoding_for_bert(self, tokens: List, labels: List, max_length: int) -> Dict:
        """
        Args:
            tokens: list of token by tokenizing splitted text
            labels: list of label for each token
            max_length:
        Returns: 
            encoding: input for bert including 'input_ids', 'token_type_ids', 'attention_mask', 'labels'
        """
        encoding = self.encode_plus(
            tokens, 
            max_length=max_length, 
            padding='max_length', 
            truncation=True
        ) 
        labels = [0] + labels[:max_length-2] + [0] # labeling for [CLS] and [SEP]
        labels = labels + [0]*( max_length - len(labels) ) # padding
        encoding['labels'] = labels

        return encoding


    def encode_plus_tagged(self, text: str, entities: Dict, max_length:int) -> Dict:
        """
        Args:
            text: 
            entities: including span and label
        """
        entities = sorted(entities, key=lambda x: x['span'][0])
        splitted = []
        position = 0
        for entity in entities:
            start = entity['span'][0]
            end = entity['span'][1]
            label = entity['type_id']
            splitted.append({'text': text[position:start], 'label':0}) 
            splitted.append({'text': text[start:end], 'label':label}) 
            position = end

        splitted.append({'text': text[position:], 'label':0})
        splitted = [ s for s in splitted if s['text'] ] 
        tokens, labels = self.tokenize_and_labeling(splitted)
        encoding = self.encoding_for_bert(tokens, labels, max_length)

        return encoding
    
class NerTokenizerForTest(BertJapaneseTokenizer):

    def encoding_for_bert(self, tokens, max_length):

        encoding = self.encode_plus(
            tokens, 
            max_length=max_length, 
            padding='max_length', 
            truncation=True,
            return_tensors = "pt"
        ) 

        return encoding


    def create_spans_of_token(self, text, tokens_original, encoding):    
        position = 0
        spans = []
        for token in tokens_original:
            l = len(token)
            while 1:
                if token != text[position:position+l]:
                    position += 1
                else:
                    spans.append([position, position+l])
                    position += l
                    break

        sequence_length = len(encoding['input_ids'])
        spans = [[-1, -1]] + spans[:sequence_length-2] 
        spans = spans + [[-1, -1]] * ( sequence_length - len(spans) ) 

        return spans


    def encode_plus_untagged(self, text, max_length=None):
        tokens = []
        tokens_original = []
        words = self.word_tokenizer.tokenize(text)
        for word in words:
            tokens_word = self.subword_tokenizer.tokenize(word) 
            tokens.extend(tokens_word)
            if tokens_word[0] == '[UNK]':
                tokens_original.append(word)
            else:
                tokens_original.extend([
                    token.replace('##','') for token in tokens_word
                ])

        encoding = self.encoding_for_bert(tokens, max_length)
        spans = self.create_spans_of_token(text, tokens_original, encoding)

        return encoding, spans


    def convert_bert_output_to_entities(self, text, labels, spans):
        labels = [label for label, span in zip(labels, spans) if span[0] != -1]
        spans = [span for span in spans if span[0] != -1]

        entities = []
        position = 0
        for label, group in itertools.groupby(labels):
            start_idx = position
            end_idx = position + len(list(group)) - 1
            
            start = spans[start_idx][0] 
            end = spans[end_idx][1]
            
            position = end_idx + 1

            if label != 0:
                entity = {
                    "name": text[start:end],
                    "span": [start, end],
                    "type_id": label
                }
                entities.append(entity)

        return entities
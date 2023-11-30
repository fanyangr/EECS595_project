from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch
import torch.nn as nn
import numpy as np
    

if __name__ == "__main__":
    """Note: I am using the ClozeTrain dataset for example, but we should decide which one to use [Cloze, Order]
    It seems like OrderTrain set is corrupeted, which means it doesn't have the correct label for conflict sentences
    """
    model_checkpoint = "bert-base-uncased"
    batch_size=32


    dataset = load_dataset("sled-umich/TRIP")

    # only consider soties with 5 sentences
    dataset.filter(lambda example: (len(example['stories'][0]['sentences'])==5 and len(example['stories'][1]['sentences'])==5))
    # for i, pair in enumerate(dataset['ClozeTrain']['confl_pairs']):
    #     if pair:
    #         print(pair)
    # training_set = dataset['OrderTrain']
    # eval_set = dataset['OrderDev']
    # test_set = dataset['OrderTest']
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    def sentence_preprocess_function(examples):
        """This preprocess function combines all the sentences together and outputs the story that is plausible"""
        # unflatten
        story_1 = [stories[0]['sentences'] for stories in examples['stories']]
        story_2 = [stories[1]['sentences'] for stories in examples['stories']]
        num_sentences_1 = [len(this_story) for this_story in story_1]
        num_sentences_2 = [len(this_story) for this_story in story_2]
        story_1 = sum(story_1, [])
        story_2 = sum(story_2, [])
        num_data = len(examples['stories'])
        total_num_sentences_1 = len(story_1)
        tokenized_story = tokenizer(story_1 + story_2)

        # flatten        
        flattened_results = {}
        for key in tokenized_story.keys():
            ctr_1 = 0
            ctr_2 = 0
            values = []
            for num_1, num_2 in zip(num_sentences_1, num_sentences_2):
                values.append([tokenized_story[key][ctr_1: ctr_1+num_1], \
                tokenized_story[key][total_num_sentences_1+ctr_2: total_num_sentences_1+ctr_2+num_2]])
                ctr_1 += num_1
                ctr_2 += num_2

            flattened_results[key] = values
        
        return flattened_results

        
    "example of encoding the dataset"
    tokenized_story = sentence_preprocess_function(dataset['ClozeTrain'][:5])
    print(tokenizer.decode(tokenized_story['input_ids'][0][0][0]))
    



    encoded_dataset = dataset.map(sentence_preprocess_function, batched=True)

    model = AutoModelForMultipleChoice.from_pretrained(model_checkpoint)
    model_name = model_checkpoint.split("/")[-1]
    args = TrainingArguments(
        f"{model_name}-finetuned-swag",
        evaluation_strategy = "epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=6,
        weight_decay=0.01,
        push_to_hub=True,
    )

    @dataclass
    class SentenceDataCollator:
        """
        Data collator that will dynamically pad the inputs for multiple choice received.
        """

        tokenizer: PreTrainedTokenizerBase
        padding: Union[bool, str, PaddingStrategy] = True
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None

        def __call__(self, features):
            story_labels = [feature.pop('label') for feature in features]
            confl_pairs = [feature.pop('confl_pairs') for feature in features]
            # flattened_features = [{k: self.flatten_sentences(v) for k, v in feature.items()} for feature in features]
            flattened_features = [self.flatten_feature(feature) for feature in features]
            
            # fine the label
            sentence_labels = []
            # flattened_feature_for_comparison = [{k: self.flatten_sentences(v) for k, v in feature.items()} for feature in features]
            for i, (story_label, confl_pair, feature, flattened_feature) in enumerate(zip(story_labels, confl_pairs, features, flattened_features)):
                confl_sent_pair = feature['input_ids'][story_label][confl_pair[0][0]] + feature['input_ids'][story_label][confl_pair[0][1]]
                for j in range(len(flattened_feature)):
                    if len(flattened_feature[j]['input_ids']) == len(confl_sent_pair):
                        if (np.array(flattened_feature[j]['input_ids']) == np.array(confl_sent_pair)).all():
                            if story_label == 0:
                                assert j < 10
                            else:
                                assert j >= 10
                            sentence_labels.append(j)


            batch_size = len(features)
            num_choices = (4+3+2+1) * 2
            flattened_features = sum(flattened_features, [])
            
            batch = self.tokenizer.pad(
                flattened_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            
            # Un-flatten
            batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
            # Add back labels
            batch["labels"] = torch.tensor(sentence_labels, dtype=torch.int64)
            return batch
        def pair2label(self, pairs, num_sentences):
            # returns the index of the conflict pair
            assert pair[0] < pair[1]
            num = 0
            for i in range(1, pair[0] + 1):
                num += (num_sentences - i)
            num += (pair[1] - pair[0] - 1) 
            return num
        def flatten_feature(self, feature):
            keys = feature.keys()
            results = []

            input_ids, attention_mask = feature['input_ids'], feature['attention_mask']
            for i in range(2): # loop over stories
                for j in range(len(input_ids[i])): 
                    for k in range(j+1, len(input_ids[i])):
                        results.append({'input_ids': input_ids[i][j] + input_ids[i][k], 'attention_mask': attention_mask[i][j] + attention_mask[i][k]})
            
            return results
        # def flatten_sentences(self, sentences):
        #     result = []
        #     for story in sentences:
        #         for i, sentence in enumerate(story):
        #             for j in range(i, len(story)):
        #                 result.append([story[i] + story[j]])
        #     return result


    accepted_keys = ["input_ids", "attention_mask", "input_ids", "attention_mask", 'label', "confl_pairs"]
    features = [{k: v for k, v in encoded_dataset['ClozeTrain'][i].items() if k in accepted_keys} for i in range(200)]
    batch = SentenceDataCollator(tokenizer)(features)
    print(batch)
    def compute_metrics(eval_predictions):
        predictions, label_ids = eval_predictions
        preds = np.argmax(predictions, axis=1)
        return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}
    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["ClozeTrain"],
        eval_dataset=encoded_dataset["ClozeTest"],
        tokenizer=tokenizer,
        data_collator=SentenceDataCollator(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()

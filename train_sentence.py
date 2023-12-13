import functools
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch
import torch.nn as nn
import numpy as np
from data_loader_utils import sentence_preprocess_function

if __name__ == "__main__":
    """Note: I am using the ClozeTrain dataset for example, but we should decide which one to use [Cloze, Order]
    It seems like OrderTrain set is corrupeted, which means it doesn't have the correct label for conflict sentences
    """
    # model_checkpoint = "roberta-base"
    model_checkpoint = "bert-base-uncased"
    batch_size=12


    dataset = load_dataset("sled-umich/TRIP")
    del dataset['OrderTrain']
    del dataset['OrderDev']
    del dataset['OrderTest']
    dataset.cleanup_cache_files() # this step is important, ortherwise it won't execute any preprocess functions, such as filtering, proprocessing.

    # only consider soties with 5 sentences
    dataset = dataset.filter(lambda example: (len(example['stories'][0]['sentences'])==5 and len(example['stories'][1]['sentences'])==5))
    dataset = dataset.filter(lambda example: len(example['confl_pairs'])==1)
    # for i, pair in enumerate(dataset['ClozeTrain']['confl_pairs']):
    #     if pair:
    #         print(pair)
    # training_set = dataset['OrderTrain']
    # eval_set = dataset['OrderDev']
    # test_set = dataset['OrderTest']
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    "example of encoding the dataset"
    tokenized_story = sentence_preprocess_function(dataset['ClozeTrain'][4:10], tokenizer=tokenizer)
    print(tokenizer.decode(tokenized_story['input_ids'][0][0][0]))
    



    encoded_dataset = dataset.map(functools.partial(sentence_preprocess_function, tokenizer=tokenizer), batched=True)
    if 'token_type_ids' in encoded_dataset.keys():
        encoded_dataset = encoded_dataset.remove_columns('token_type_ids')
    # encoded_dataset = encoded_dataset.rename_column('t','label')


    model = AutoModelForMultipleChoice.from_pretrained(model_checkpoint)
    model_name = model_checkpoint.split("/")[-1]
    args = TrainingArguments(
        f"{model_name}-finetuned-confl_sentence",
        evaluation_strategy = "epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=8,
        weight_decay=0.01,
        push_to_hub=False,
        lr_scheduler_type='linear',
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
            label_name = "label" if "label" in features[0].keys() else "labels"
            labels = [feature.pop(label_name) for feature in features]
            batch_size = len(features)
            num_choices = len(features[0]["input_ids"])
            assert num_choices == (4+3+2+1) * 2
            for feature in features:
                for k, v in feature.items():
                    assert len(v) == 20
            flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features]
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
            batch["labels"] = torch.tensor(labels, dtype=torch.int64)
            return batch
        
    encoded_dataset = encoded_dataset.remove_columns('label')
    encoded_dataset = encoded_dataset.rename_column('arranged_confl_labels','label')
    # accepted_keys = ["input_ids", "attention_mask", 'label']
    # features = [{k: v for k, v in encoded_dataset['ClozeTrain'][i].items() if k in accepted_keys} for i in range(len(encoded_dataset['ClozeTrain']))]
    # batch = SentenceDataCollator(tokenizer)(features)
    # print(batch)
    def compute_metrics(eval_predictions):
        " given the gt sotry label, predict the conflict sentences"
        predictions, label_ids = eval_predictions
        # preds = np.argmax(predictions, axis=1)
        prediction_1 = predictions[:, :10]
        prediction_2 = predictions[:, 10:20]
        preds_1 = np.argmax(prediction_1, axis=1)
        preds_2 = np.argmax(prediction_2, axis=1) + 10
        accuracy_1 = (preds_1 == label_ids).astype(np.float32).mean().item()
        accuracy_2 = (preds_2 == label_ids).astype(np.float32).mean().item()
        return {"accuracy": accuracy_1 + accuracy_2}
        # return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}
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

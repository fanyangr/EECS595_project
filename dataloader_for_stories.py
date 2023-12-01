import functools
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch
import numpy as np
from data_loader_utils import story_preprocess_function
    

if __name__ == "__main__":
    """Note: I am using the OrderTrain dataset for example, but we should decide which one to use [Cloze, Order]"""
    model_checkpoint = "bert-base-uncased"
    batch_size=32


    dataset = load_dataset("sled-umich/TRIP")
    dataset.cleanup_cache_files()

    # training_set = dataset['OrderTrain']
    # eval_set = dataset['OrderDev']
    # test_set = dataset['OrderTest']
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        
    "example of encoding the dataset"
    tokenized_story = story_preprocess_function(dataset['OrderTrain'][:5], tokenizer=tokenizer)
    print(tokenizer.decode(tokenized_story['input_ids'][0][0]))
    

    encoded_dataset = dataset.map(functools.partial(story_preprocess_function, tokenizer=tokenizer), batched=True)

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
    class StoryDataCollator:
        """
        Data collator that will dynamically pad the inputs for multiple choice received.
        """

        tokenizer: PreTrainedTokenizerBase
        padding: Union[bool, str, PaddingStrategy] = True
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None

        def __call__(self, features):
            label_name = "label" if "label" in features[0].keys() else "labels"
            # 1 means plausible, 0 means not
            labels = [feature.pop(label_name) for feature in features]
            batch_size = len(features)
            num_choices = len(features[0]["input_ids"])
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
    accepted_keys = ["input_ids", "attention_mask", "input_ids", "label"]
    # features = [{k: v for k, v in encoded_dataset['OrderTrain'][i].items() if k in accepted_keys} for i in range(10)]
    # batch = StoryDataCollator(tokenizer)(features)

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
        data_collator=StoryDataCollator(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
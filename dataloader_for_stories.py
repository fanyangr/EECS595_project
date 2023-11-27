from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch
    

if __name__ == "__main__":
    """Note: I am using the OrderTrain dataset for example, but we should decide which one to use [Cloze, Order]"""
    model_checkpoint = "bert-base-uncased"
    batch_size=4


    dataset = load_dataset("sled-umich/TRIP")
    # training_set = dataset['OrderTrain']
    # eval_set = dataset['OrderDev']
    # test_set = dataset['OrderTest']
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    def story_preprocess_function(examples):
        """This preprocess function combines all the sentences together and outputs the story that is plausible"""
        story_1 = [' '.join(stories[0]['sentences']) for stories in examples['stories']]
        story_2 = [' '.join(stories[1]['sentences']) for stories in examples['stories']]
        num_data = len(examples['stories'])
        tokenized_story = tokenizer(story_1 + story_2)
        return {k: [[v[i], v[i+num_data]] for i in range(num_data)] for k, v in tokenized_story.items()}

        
    "example of encoding the dataset"
    tokenized_story = story_preprocess_function(dataset['OrderTrain'][:5])
    print(tokenizer.decode(tokenized_story['input_ids'][0][0]))
    



    encoded_dataset = dataset.map(story_preprocess_function, batched=True)

    model = AutoModelForMultipleChoice.from_pretrained(model_checkpoint)
    model_name = model_checkpoint.split("/")[-1]
    args = TrainingArguments(
        f"{model_name}-finetuned-swag",
        evaluation_strategy = "epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
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
    accepted_keys = ["input_ids", "attention_mask", "input_ids", "attention_mask", "label"]
    features = [{k: v for k, v in encoded_dataset['OrderTrain'][i].items() if k in accepted_keys} for i in range(10)]
    batch = StoryDataCollator(tokenizer)(features)

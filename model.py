from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"

# Load the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-large')
model = T5ForConditionalGeneration.from_pretrained('t5-large')

# Load the SNLI dataset
dataset = load_dataset('snli')

# Filter out examples with positive entailment
dataset = dataset.filter(lambda example: example['label'] == 0)  # label 0 usually stands for entailment in SNLI dataset

# Preprocess the dataset for T5
def preprocess(examples):
    # T5 expects the task to be in the input so prepend 'entailment:' to the hypothesis
    src_texts = ['entailment: ' + hypothesis for hypothesis in examples['hypothesis']]
    tgt_texts = examples['premise']

    input_encodings = tokenizer(src_texts, truncation=True, padding="max_length", max_length=300)
    target_encodings = tokenizer(tgt_texts, truncation=True, padding="max_length", max_length=300)

    return {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['input_ids'],
    }


# Map the preprocess function to all examples in the dataset
dataset = dataset.map(preprocess, batched=True)

# Define your training arguments
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=500,
    weight_decay=0.01,
)

# Define your trainer
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
)

# Train the model
trainer.train()


# Save the model
model_path = "./t5_snli_model"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

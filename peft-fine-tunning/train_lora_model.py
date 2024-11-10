# Imports
from datasets import load_dataset
from transformers import (
    DistilBertTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, AutoPeftModelForSequenceClassification
import numpy as np

# Load and split dataset
dataset = load_dataset("sms_spam", split="train").train_test_split(test_size=0.2, shuffle=True, seed=23)
splits = ["train", "test"]

# Tokenization
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
tokenized_dataset = {}
for split in splits:
    tokenized_dataset[split] = dataset[split].map(
        lambda x: tokenizer(x["sms"], truncation=False, padding='max_length'), batched=True
    )

# Load base model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
    id2label={0: "not spam", 1: "spam"},
    label2id={"not spam": 0, "spam": 1}
)

# Set model parameters to require gradients
for param in model.parameters():
    param.requires_grad = True

# Define compute metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}

# Train base model
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./data/spam_not_spam",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=2,
        weight_decay=0.01,
        load_best_model_at_end=True
    ),
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

base_eval_results = trainer.evaluate()

# Configure LoRA
config = LoraConfig(
    r=8, 
    lora_alpha=16,
    target_modules=["q_lin", "v_lin"],
    lora_dropout=0.1,
    bias="none",
    task_type='SEQ_CLS'
)

# Apply LoRA model
lora_model = get_peft_model(model, config)
lora_model.print_trainable_parameters()

train_data = tokenized_dataset['train'].rename_column('label', 'labels')
test_data = tokenized_dataset['test'].rename_column('label', 'labels')

# Train LoRA model
lora_trainer = Trainer(
    model=lora_model,
    args=TrainingArguments(
        output_dir="./data/lora_spam_not_spam",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=2,
        weight_decay=0.01,
        load_best_model_at_end=True
    ),
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics
)
lora_trainer.train()
lora_trainer.evaluate()

# Save LoRA model
lora_model.save_pretrained("sms-lora")

# Load and evaluate LoRA model
loaded_lora_model = AutoPeftModelForSequenceClassification.from_pretrained("sms-lora")
loaded_lora_trainer = Trainer(
    model=loaded_lora_model,
    args=TrainingArguments(
        output_dir="./data/loaded_lora_spam_not_spam",
        per_device_eval_batch_size=16,
    ),
    eval_dataset=test_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    train_dataset=train_data
)
loaded_lora_eval_results = loaded_lora_trainer.evaluate()

# Display evaluation results
print("Base model evaluation results:", base_eval_results)
print("Loaded LoRA model evaluation results:", loaded_lora_eval_results)

improvement = loaded_lora_eval_results["eval_accuracy"] - base_eval_results["eval_accuracy"]
print(f"PEFT fine Tuning improved results by: {improvement * 100:.2f}%")

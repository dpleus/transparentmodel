from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from transparentmodel.huggingface.training import train_with_memory_tracking

# Load Model
model = AutoModelForCausalLM.from_pretrained("gpt2")
config = model.config
config.use_cache = False
model = AutoModelForCausalLM.from_pretrained("gpt2", config=config)
print(model.config)


# Prepare Data
dataset = load_dataset("vicgalle/alpaca-gpt4")

tokenizer = AutoTokenizer.from_pretrained("gpt2")


def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")

tokenizer.pad_token = tokenizer.eos_token
tokenized_datasets = dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch", per_device_train_batch_size=1)

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

train_with_memory_tracking(trainer)
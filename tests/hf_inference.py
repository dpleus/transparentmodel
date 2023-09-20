from transformers import AutoModelForCausalLM, AutoTokenizer

from transparentmodel.huggingface.inference import (
    generate_with_memory_tracking,
)

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Encode a text inputs
text = (
    "Who was Jim Henson ? Jim Henson was a Who was Jim Henson ? "
    "Jim Henson was a Who was Jim Henson ? Jim Henson "
    "was aWho was Jim Henson ? Jim Henson was a Who was Jim Henson ? Jim "
    "Henson was a Who was Jim Henson ? Jim"
    "Henson was a Who was Jim Henson ? Jim Henson was aWho was Jim Henson ? "
    "Jim Henson was aWho was Jim Henson ? "
    "Jim Henson was a Who was Jim Henson ? Jim Henson was a Who was Jim "
    "Henson ? Jim Henson was aWho was Jim"
    "Henson ? Jim Henson was a"
)
tokenizer.pad_token = tokenizer.eos_token
indexed_tokens = tokenizer(text, return_tensors="pt")

# Generate next tokens and decode them to text
tokens_tensor = generate_with_memory_tracking(
    model,
    realtime=False,
    input_ids=indexed_tokens["input_ids"],
    min_new_tokens=50,
    max_new_tokens=70,
    num_beams=30,
)
tokens = tokenizer.decode(tokens_tensor[0], skip_special_tokens=True)

print(tokens)

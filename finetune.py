# %%
!pip install huggingface_hub
!pip install transformers
!pip install accelerate  peft  bitsandbytes  trl
!pip install sentencepiece

# %%
from huggingface_hub import login

login()


# %%
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# %%
model_id = "meta-llama/Llama-2-7b-chat-hf"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# %%
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, device_map={"": 0})

# %%
from peft import prepare_model_for_kbit_training
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# %%
def print_trainable_parameters(model):
    """

    Prints the number of trainable parameters in the model.

    """

    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f"trainable params: {trainable_params} | | all params: {all_param} | | trainable % : {100 * trainable_params / all_param}")


# %%
from peft import LoraConfig, get_peft_model
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# %%
model = get_peft_model(model, config)

print_trainable_parameters(model)

# %%
import transformers
from trl import SFTTrainer

# needed for llama tokenizer
tokenizer.pad_token = tokenizer.eos_token


trainer = SFTTrainer(
  model=model,
  train_dataset=dataset,
  dataset_text_field="text",
  args=transformers.TrainingArguments(
  per_device_train_batch_size=1,
  gradient_accumulation_steps=4,
  warmup_steps=2,
  max_steps=10,          #change accordingly
  learning_rate=2e-4,
  fp16=True,
  logging_steps=1,
  output_dir="outputs",
  optim="paged_adamw_8bit"
),
data_collator=transformers.DataCollatorForLanguageModeling(
    tokenizer, mlm=False),

)

model.config.use_cache=False 

# %%
trainer.train()

# %%
from transformers import pipeline
prompt = """### Human: Write a short story about two students trapped in a haunted house in Montana. ### Assistant:"""


pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
    )

sequences = pipe(
            prompt,
            max_length=1000,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
    )

for seq in sequences:
    print(seq['generated_text'])

# %%
from transformers import AutoTokenizer
import transformers
import torch

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)
sequences = pipeline(
    'I liked batman. Do you have any recommendations of other shows or movie I might like?\n',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    max_length=500,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")



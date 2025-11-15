import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

model_name = "Qwen/Qwen3-0.6B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA device is required to run this script.")

device = torch.device("cuda")

model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto").to(device)

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,  # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

# conduct text completion
if os.getenv("ENABLE_TORCH_COMPILE", "0") == "1":
    print("Compiling the model with torch.compile")
    torch.compiler.reset()
    torch._dynamo.reset()
    model.forward = torch.compile(model.forward, fullgraph=True)

    past_key_values = StaticCache(
        model.config,
        max_batch_size=1,
        device=device,
        dtype=torch.bfloat16,
        max_cache_len=model_inputs.input_ids.shape[1] + 1,
    )

    # import depyf
    # with depyf.prepare_debug("depyf_qwen3"):
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1,
        past_key_values=past_key_values,
    )
else:
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=128,
    )
output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip(
    "\n"
)
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)

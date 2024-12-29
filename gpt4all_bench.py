import time
from gpt4all import GPT4All

results = []
devices = ["gpu", "cuda"]
models = ["Meta-Llama-3-8B-Instruct.Q4_0.gguf",
          "Phi-3-mini-4k-instruct.Q4_0.gguf",
          "orca-mini-3b-gguf2-q4_0.gguf"]

def generate(model_name, device):
    print("available devices:")
    print(GPT4All.list_gpus())
    model = GPT4All(model_name, device=device)
    print(f"model created for {model_name}/{device}")
    start_time = time.time()
    response = model.generate("Who was Jerry Garcia?")
    end_time = time.time()
    
    latency = end_time - start_time
    print(f"response generated in {latency}")
    result = {"model":model_name, "device":device, "response":response,"latency":latency}
    results.append(result)

for model in models:
    for device in devices:
        generate(model, device)

print(f"model       device      latency")
for result in results:
    print(f"{result['model'][:10]}      {result['device']}      {result['latency']}")




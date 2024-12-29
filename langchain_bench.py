from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import time

results = []
devices = ["cpu", "gpu"]
models = ["mistralai/Mistral-7B-Instruct-v0.3"]

def generate(model_name, device):
    d = -1 if device == "cpu" else 0
    model = HuggingFacePipeline(pipeline=pipeline("text-generation", model=model_name, device=d))
    print(f"model created for {model_name}/{device} using {model.pipeline.device}")
    start_time = time.time()
    response = model.invoke("Who was Jerry Garcia?")
    print(response)
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

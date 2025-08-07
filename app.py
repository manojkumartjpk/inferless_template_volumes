import json
import os
import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from datetime import datetime


class InferlessPythonModel:

    def initialize(self):
        print("started load model", flush=True)
        # Get custom cache path from environment
        folder_path = os.getenv("NFS_PATH")
        
        # Set environment variable for Hugging Face cache
        os.environ["TRANSFORMERS_CACHE"] = folder_path
        os.environ["HF_HOME"] = folder_path

        # (Optional) set Torch hub cache too, if needed
        os.environ["TORCH_HOME"] = folder_path

        # Load the model using the cache path
        self.generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M", device=0)


    def infer(self, inputs):
        print("started inference", flush=True)
        prompt = inputs["prompt"]
        pipeline_output = self.generator(prompt, do_sample=True, min_length=20)
        generated_txt = pipeline_output[0]["generated_text"]
        return {"generated_text": generated_txt }

    def finalize(self, args):
        self.generator = None

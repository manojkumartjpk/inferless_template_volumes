import json
import os
import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from datetime import datetime


class InferlessPythonModel:

    def initialize(self):
        print("Started model loading", flush=True)

        # Get custom cache path from environment
        folder_path = os.getenv("NFS_PATH")
        if not folder_path:
            raise EnvironmentError("NFS_PATH environment variable not set")

        print(f"Using cache directory: {folder_path}", flush=True)

        # Load tokenizer and model using the specified cache directory
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neo-125M",
            cache_dir=folder_path
        )

        model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-neo-125M",
            cache_dir=folder_path
        )

        # Create the pipeline with the loaded model and tokenizer
        self.generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0  # Set to -1 for CPU, or 0 for GPU
        )

        print("Model loaded successfully", flush=True)

    def infer(self, inputs):
        print("Started inference", flush=True)

        prompt = inputs["prompt"]

        # Perform text generation
        pipeline_output = self.generator(
            prompt,
            do_sample=True,
            min_length=20,
            max_length=100
        )

        generated_txt = pipeline_output[0]["generated_text"]
        return {"generated_text": generated_txt}

    def finalize(self, args):
        print("Cleaning up", flush=True)
        self.generator = None

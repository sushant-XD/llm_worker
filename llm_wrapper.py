import logging
import os
import torch
from transformers import (
    AutoTokenizer, GenerationConfig,
    AutoModelForCausalLM
)

from config import Config

class LLMWrapper:
    def __init__(self):
        self.client = None
        if (Config.options or {}).get("provider") == "openrouter":
            from openai import OpenAI
            self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
            return
        self.model = AutoModelForCausalLM.from_pretrained(Config.model_str, torch_dtype=torch.bfloat16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(Config.model_str)

    def _openrouter(self, prompt: str, temperature: float) -> str:
        completion = self.client.chat.completions.create(
            model=Config.model_str,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1024,
        )
        return (completion.choices[0].message.content or "").strip()

    def _generate_response(self, prompt: str) -> str:
        if self.client:
            return self._openrouter(prompt, 0)
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        generation_config = GenerationConfig(max_new_tokens=1024, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        with torch.no_grad():
            output_tokens = self.model.generate(**inputs, generation_config=generation_config)
        input_length = inputs['input_ids'].shape[1]
        newly_generated_tokens = output_tokens[0, input_length:]
        return self.tokenizer.decode(newly_generated_tokens, skip_special_tokens=True).strip()

    def _generate_response_with_sampling(self, prompt: str) -> str:
        if self.client:
            return self._openrouter(prompt, 0.3)
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)

        generation_config = GenerationConfig(
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        )

        with torch.no_grad():
            output_tokens = self.model.generate(**inputs, generation_config=generation_config)

        input_length = inputs['input_ids'].shape[1]
        newly_generated_tokens = output_tokens[0, input_length:]

        return self.tokenizer.decode(newly_generated_tokens, skip_special_tokens=True).strip()


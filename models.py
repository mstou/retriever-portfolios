from abc import ABC, abstractmethod
import openai
import pickle
import os
import json

class LLMBase(ABC):
    """
    Abstract base class for any LLM wrapper.
    """
    @abstractmethod
    def answer(self, question, passages, system_prompt="You are a helpful assistant."):
        pass

    def judge(self, answers):
        raise NotImplementedError("Judge method not implemented.")

    def score(self, answer, reference):
        raise NotImplementedError("Score method not implemented.")


class LLMCache:
    def __init__(self, filename=None):
        self.filename = filename
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                self.data = pickle.load(f)
        else:
            self.data = {}
        self.cache_hits = 0
        self.cache_requests = 0
        self._since_last_write = 0
    
    def _make_key(self, question_id, passage_ids):
        return (question_id, tuple(sorted(passage_ids)))
    
    def get(self, question_id, passage_ids):
        self.cache_requests += 1
        key = self._make_key(question_id, passage_ids)
        if key in self.data:
            self.cache_hits += 1
            return self.data.get(key)
        return None

    def set(self, question_id, passage_ids, value):
        key = self._make_key(question_id, passage_ids)

        if key not in self.data:
            self._since_last_write += 1
            self.data[key] = value

            if self._since_last_write > 200:
                self._since_last_write = 0
                with open(self.filename, "wb") as f:
                    pickle.dump(self.data, f)
        
    
    def initialize(self):
        self.data = {}
        with open(self.filename, "wb") as f:
                pickle.dump(self.data, f)

    def close(self):
        with open(self.filename, "wb") as f:
                pickle.dump(self.data, f)

class OpenAI_LLM(LLMBase):
    """
    LLM wrapper for OpenAI API models (e.g. GPT-3.5, GPT-4o, and local models via vllm).
    """
    def __init__(self, model_name, api_key, base_url=None):
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)


    def answer(self, system_prompt, user_prompt, max_tokens=128, temperature=0.0, top_p=1.0):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        params = params = dict(
            model=self.model_name,
            messages=messages,
        )

        if "gpt-4.1" in self.model_name or "o3" in self.model_name or "o4" in self.model_name:
            params['max_completion_tokens'] = max_tokens
        else:
            params['max_tokens'] = max_tokens
        
        params['temperature'] = temperature
        params['top_p'] = top_p
        params['stop'] = '</final>'
        
        response = self.client.chat.completions.create(**params)

        return response.choices[0].message.content.strip()


    def judge(self, system_prompt, user_prompt, max_tokens=256, verbose=False, temperature=0.0, top_p=1.0):
        """
        - Given a question and a list of candidate answers (and optionally passages),
        ask the LLM to choose the best answer and return its index.
        - The instructions are given by the system/user prompts 
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        params = params = dict(
            model=self.model_name,
            messages=messages,
        )

        if verbose:
            # ============== Printing ==============
            print("\n[OpenAI_LLM.judge] --- SENDING TO API ---")
            print(f"Model: {self.model_name}")
            print(f"System prompt:\n{system_prompt}\n")
            print(f"User prompt:\n{user_prompt}\n")
            print(f"API params: max_tokens={max_tokens if 'max_tokens' in locals() else 'N/A'}")
            print("-"*50)
            # ==========================================

        if "gpt-4.1" in self.model_name or "o3" in self.model_name or "o4" in self.model_name:
            params['max_completion_tokens'] = max_tokens
        else:
            params['max_tokens'] = max_tokens
        
        params['temperature'] = temperature
        params['top_p'] = top_p
        params['stop'] = '</final>'

        response = self.client.chat.completions.create(**params)

        if verbose:
            print(f'Response from API: \n{response}\n\n{response.choices[0].message.content.strip()}')
        
        text = response.choices[0].message.content.strip()
        
        return text
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ChatGeneration:
    text: str

@dataclass
class ChatResult:
    generations: List[List[ChatGeneration]]
    llm_output: Optional[dict] = None


class LocalChatModel:
    def __init__(self, model_name, use_vllm=False) -> None:
        self.model_name = model_name
        self.use_vllm = use_vllm
        if use_vllm:
            from vllm import LLM
            self.model = LLM(model_name, tensor_parallel_size=torch.cuda.device_count())
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token_id = 0
            self.tokenizer.padding_side = "left"

    def generate(self, messages) -> ChatResult:
        pass

    def create_result(self, predictions: List[str]) -> ChatResult:
        return ChatResult(
            generations=[[ChatGeneration(p)] for p in predictions],
            llm_output={"token_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0
            }}
        )
from transformers import GenerationConfig
from local_chat_model import LocalChatModel


class Llama2Model(LocalChatModel):
    def __init__(self, model_name, use_vllm=False) -> None:
        super().__init__(model_name, use_vllm)

    def generate(self, messages):
        prompts = [m[0].content for m in messages]
        if self.use_vllm:
            from vllm import SamplingParams
            sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
            outputs = self.model.generate(prompts, sampling_params)
            predictions = [o.outputs[0].text.split("\n")[0].strip() for o in outputs]
        else:
            pass
            # prompt_length = len(self.tokenizer.decode(self.tokenizer(prompt).input_ids, skip_special_tokens=True))
            # inputs = self.tokenizer(prompt, return_tensors="pt")
            # input_ids = inputs.input_ids.cuda()
            # attn_mask = inputs.attention_mask.cuda()
            # gen_config = GenerationConfig(do_sample=False, max_new_tokens=256, pad_token_id=self.tokenizer.pad_token_id)
            # generate_ids = self.model.generate(input_ids, gen_config, attention_mask=attn_mask)
            # output = self.tokenizer.decode(generate_ids[0], skip_special_tokens=True)
            # prediction = output[prompt_length:].split("\n")[0].strip()

        return self.create_result(predictions)
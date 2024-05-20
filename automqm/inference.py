import pandas as pd
import torch
import json
import argparse
from vllm import LLM, SamplingParams

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", "-m", type=str, help="model name or path")
    parser.add_argument("--testset-path", "-t", type=str, default="data/wmt22_mqm_zh-en.pkl", help="path of the testset")
    parser.add_argument("--output-path", "-o", type=str)
    parser.add_argument("--input-mode", "-mode", type=str, help="input mode: st/rt/srt")
    parser.add_argument("--src", type=str, default="Chinese")
    parser.add_argument("--tgt", type=str, default="English")
    args = parser.parse_args()

    model_name = args.model_name
    testset_path = args.testset_path
    output_path = args.output_path
    input_mode = args.input_mode

    src_lang = args.src
    tgt_lang = args.tgt
    if input_mode == "srt":
        prompt_path = "prompts/prompt_alpaca_ref.txt"
    elif input_mode == "rt":
        prompt_path = "prompts/prompt_alpaca_no_src.txt"
    elif input_mode == "st":
        prompt_path = "prompts/prompt_alpaca_no_ref.txt"
    else:
        raise RuntimeError("input mode error")
    with open(prompt_path) as p:
        prompt_template = p.read()

    testset = pd.read_pickle(testset_path)
    model = LLM(model_name, tensor_parallel_size=torch.cuda.device_count(), dtype="float16")
    sampling_params = SamplingParams(temperature=0.0, max_tokens=512)

    prompts = [prompt_template.format(src_lang=src_lang, tgt_lang=tgt_lang, source=sample.source, reference=sample.reference, translation=sample.candidate) for sample in testset.itertuples()]
    outputs = model.generate(prompts, sampling_params)
    output_text = [o.outputs[0].text.strip() for o in outputs]
    with open(output_path, "w") as f:
        json.dump(output_text, f, indent=2)

if __name__ == "__main__":
    main()

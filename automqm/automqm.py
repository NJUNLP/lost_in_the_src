import argparse
import json
import torch
import os
import re
import pandas as pd
from example_selector import (
    StratifiedSampler,
    UniformSampler,
    ExampleRetriever,
    SemanticRetriever,
    StratifiedSemanticRetriever,
    EditDistanceRetriever,
)
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts.chat import HumanMessagePromptTemplate, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, load_prompt
from llama2_model import Llama2Model

LANG_CODEBOOK = {"zh": "Chinese", "en": "English", "de": "German"}

def add_translation(df, translation_path):
    from math import ceil
    with open(translation_path) as t:
        translations = t.readlines()
    translations = [t.strip() for t in translations if t != "<None>\n"]
    indices = df.index
    translations = translations * ceil((indices[-1] + 1) / len(translations))
    translations = pd.Series(translations).loc[indices]
    df["translation_m"] = translations

def format_examples(examples: pd.DataFrame, src_lang: str, tgt_lang: str, has_reference: bool, has_source: bool, translation_as_ref: bool):
    pattern = re.compile(r"<v>(.*)</v>")
    example_list = []
    for s in examples.itertuples():
        if s.category:
            error = []
            for span, category, severity in zip(s.span, s.category, s.severity):
                error_span = re.search(pattern, span).group(1)
                first_category = category.split("/")[0].lower()
                error.append(f"'{error_span}' - {severity}/{first_category}")
            errors = "; ".join(error)
        else:
            errors = "No error"
        if has_source:
            if has_reference:
                if translation_as_ref:
                    example_list.append({"src_lang": src_lang, "source": s.source, "tgt_lang": tgt_lang, "reference": s.translation_m, "translation": s.candidate, "error": errors})
                else:
                    example_list.append({"src_lang": src_lang, "source": s.source, "tgt_lang": tgt_lang, "reference": s.reference, "translation": s.candidate, "error": errors})
            else:
                example_list.append({"src_lang": src_lang, "source": s.source, "tgt_lang": tgt_lang, "translation": s.candidate, "error": errors})
        else:
            if has_reference:
                if translation_as_ref:
                    example_list.append({"tgt_lang": tgt_lang, "reference": s.translation_m, "translation": s.candidate, "error": errors})
                else:
                    example_list.append({"tgt_lang": tgt_lang, "reference": s.reference, "translation": s.candidate, "error": errors})
            else:
                example_list.append({"tgt_lang": tgt_lang, "translation": s.candidate, "error": errors})
    return example_list

def format_template(prompt_template, test_sample, select_demos, selector, example_selector, has_reference, has_source, translation_as_ref, src_lang, tgt_lang):
    reference = test_sample.translation_m if translation_as_ref else test_sample.reference
    if select_demos and selector is not None:
        if example_selector == "edit_distance" or example_selector == "stratified_semantic" or example_selector == "semantic":
            examples = selector.get_examples(test_sample.source, test_sample.candidate)
        else:
            examples = selector.get_examples()
        example_index = examples.index.to_list()
        formated_examples = format_examples(examples, LANG_CODEBOOK[src_lang], LANG_CODEBOOK[tgt_lang], has_reference, has_source, translation_as_ref)
        prompt_template.examples = formated_examples
        if has_source:
            if has_reference:
                prompt = prompt_template.format(
                        src_lang=LANG_CODEBOOK[src_lang],
                        source=test_sample.source,
                        tgt_lang=LANG_CODEBOOK[tgt_lang],
                        reference=reference,
                        translation=test_sample.candidate
                    )
            else:
                prompt = prompt_template.format(
                        src_lang=LANG_CODEBOOK[src_lang],
                        source=test_sample.source,
                        tgt_lang=LANG_CODEBOOK[tgt_lang],
                        translation=test_sample.candidate
                    )
        else:
            if has_reference:
                prompt = prompt_template.format(
                    tgt_lang=LANG_CODEBOOK[tgt_lang],
                    reference=reference,
                    translation=test_sample.candidate
                )
            else:
                prompt = prompt_template.format(
                    tgt_lang=LANG_CODEBOOK[tgt_lang],
                    translation=test_sample.candidate
                )
        p = [HumanMessage(content=prompt)]
    else:
        example_index = None
        if has_source:
            if has_reference:
                p = prompt_template.format_messages(
                        src_lang=LANG_CODEBOOK[src_lang],
                        source=test_sample.source,
                        tgt_lang=LANG_CODEBOOK[tgt_lang],
                        reference=reference,
                        translation=test_sample.candidate
                    )
            else:
                p = prompt_template.format_messages(
                        src_lang=LANG_CODEBOOK[src_lang],
                        source=test_sample.source,
                        tgt_lang=LANG_CODEBOOK[tgt_lang],
                        translation=test_sample.candidate
                    )
        else:
            if has_reference:
                p = prompt_template.format_messages(
                        tgt_lang=LANG_CODEBOOK[tgt_lang],
                        reference=reference,
                        translation=test_sample.candidate
                    )
            else:
                p = prompt_template.format_messages(
                        tgt_lang=LANG_CODEBOOK[tgt_lang],
                        translation=test_sample.candidate
                    )
            
    return p, example_index

def output_results(output_path, testset, results):
    with open(output_path + ".txt", "w") as txt:
        for s in pd.concat([testset[["source", "candidate", "reference", "span", "category", "severity"]], results], axis=1).itertuples():
            txt.write(f"Sample#{s.Index}:\n")
            txt.write(f"{s.source}\n")
            txt.write(f"{s.reference}\n")
            txt.write(f"{s.candidate}\n")
            if s.span:
                for span, cate, seve in zip(s.span, s.category, s.severity):
                    txt.write(f"{span}\t{cate}\t{seve}\n")
            else:
                txt.write("No error\n")
            txt.write(f"{s.output}\n\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-2-13b-hf")
    parser.add_argument("-l", "--lang-pair", type=str, default="en-de")
    parser.add_argument("-p", "--prefix", type=str, default="llama2_13b_no_src_ref_stratified_wmt22_ende_3200")
    parser.add_argument("--example-selector", type=str, default="stratified")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-src", "--has-source", action="store_true")
    parser.add_argument("-ref", "--has-reference", action="store_true")
    parser.add_argument("--prompt-path", type=str, default="prompts/prompt_no_src_ref_sample.json")
    args = parser.parse_args()
    print(args)

    model_name = args.model_name
    prefix = args.prefix
    example_selector = args.example_selector
    seed = args.seed
    has_source = args.has_source
    has_reference = args.has_reference
    translation_as_ref = False
    select_demos = True
    lang_pair = args.lang_pair
    prompt_path = args.prompt_path
    checkpoint = f"save/{prefix}.pt"
    output_path = f"results/{prefix}"
    test_path = f"data/wmt22_mqm_{lang_pair}_3200.pkl"
    pool_path = f"data/wmt21_mqm_{lang_pair}.pkl"
    src_lang, tgt_lang = lang_pair.split("-")

    pool = pd.read_pickle(pool_path)
    selector = None
    if select_demos:
        if example_selector == "uniform":
            selector = UniformSampler(num_shot=4, pool=pool, seed=seed)
        elif example_selector == "stratified":
            selector = StratifiedSampler(num_shot=4, pool=pool, seed=seed)
        elif example_selector == "retrieval":
            selector = ExampleRetriever(pool=pool, example_path="save/gpt3.5_no_ref_sampletest_500.pt")
        elif example_selector == "semantic":
            selector = SemanticRetriever(num_shot=4, pool=pool, src_lang=src_lang, tgt_lang=tgt_lang)
        elif example_selector == "stratified_semantic":
            selector = StratifiedSemanticRetriever(num_shot=4, pool=pool, src_lang=src_lang, tgt_lang=tgt_lang, seed=42)
        elif example_selector == "edit_distance":
            selector = EditDistanceRetriever(num_shot=4, pool=pool, src_lang=src_lang, tgt_lang=tgt_lang)
        else:
            raise ValueError(f"Unknown example selector {example_selector}")
        prompt_template = load_prompt(prompt_path)
    else:
        if has_source:
            if has_reference:
                human_message = HumanMessagePromptTemplate.from_template_file(prompt_path, input_variables=["src_lang", "tgt_lang", "source", "reference", "translation"])
            else:
                human_message = HumanMessagePromptTemplate.from_template_file(prompt_path, input_variables=["src_lang", "tgt_lang", "source", "translation"])
        else:
            if has_reference:
                human_message = HumanMessagePromptTemplate.from_template_file(prompt_path, input_variables=["tgt_lang", "reference", "translation"])
            else:
                human_message = HumanMessagePromptTemplate.from_template_file(prompt_path, input_variables=["tgt_lang", "translation"])
        prompt_template = ChatPromptTemplate.from_messages([human_message])

    if "gpt" in model_name:
        chat_model = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0, max_tokens=384)
        pass
    elif "llama" in model_name.lower():
        if "chat" in model_name:
            pass
        else:
            chat_model = Llama2Model(model_name, True)
    testset = pd.read_pickle(test_path)
    prompt_tokens = completion_tokens = 0
    results = []
    indices = []
    example_indices = []
    if checkpoint is not None and os.path.exists(checkpoint):
        pt = torch.load(checkpoint)
        results = pt["results"]
        indices = pt["indices"]
        example_indices = pt["example_indices"]
        random_state = pt["random_state"]
        if selector is not None:
            selector.set_rng_state(random_state)
        prompt_tokens = pt["prompt_tokens"]
        completion_tokens = pt["completion_tokens"]

    bz = 50
    messages = []
    for i, test_sample in enumerate(testset.itertuples(), start=1):
        if indices and test_sample.Index <= indices[-1]:
            continue
        print(f"sample#{test_sample.Index}")
        p, example_index = format_template(prompt_template, test_sample, select_demos, selector, example_selector, has_reference, has_source, translation_as_ref, src_lang, tgt_lang)
        messages.append(p)
        indices.append(test_sample.Index)
        if example_index is not None:
            example_indices.append(example_index)
        if len(messages) < bz and i < len(testset):
            continue
        r = chat_model.generate(messages)
        outputs = [o[0].text for o in r.generations]
        print(outputs)
        print()
        prompt_tokens += r.llm_output["token_usage"]["prompt_tokens"]
        completion_tokens += r.llm_output["token_usage"]["completion_tokens"]
        results.extend(outputs)
        torch.save(
            {
                "results": results,
                "indices": indices,
                "example_indices": example_indices if example_indices else None,
                "random_state": selector.get_rng_state() if selector is not None else None,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens
            },
            checkpoint
        )
        messages = []

    print(f"prompt tokens: {prompt_tokens}\ncompletion tokens: {completion_tokens}")
    print(f"cost: {prompt_tokens * 0.0000015 + completion_tokens * 0.000002:.4f}")
    results = pd.Series(results, index=indices, name="output", dtype="string")
    results.to_pickle(f"{output_path}.pkl")
    output_results(output_path, testset, results)

if __name__ == "__main__":
    main()

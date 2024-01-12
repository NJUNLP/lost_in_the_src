import os
import glob
import time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

LANG_CODES = {"en": "English", "zh": "Chinese", "de": "German", "ru": "Russian"}
t_ins_prefix = """\
Score the following translation from {source_lang} to {target_lang} on a continuous scale from 0 to 100 that starts on "No meaning preserved", \
goes through "Some meaning preserved", then "Most meaning preserved and few grammar mistakes", up to "Perfect meaning and grammar".\n\n{target_lang} translation: "\
"""

st_ins_prefix = """\
Score the following translation from {source_lang} to {target_lang} on a continuous scale from 0 to 100 that starts on "No meaning preserved", \
goes through "Some meaning preserved", then "Most meaning preserved and few grammar mistakes", up to "Perfect meaning and grammar".\n\n{source_lang} source: \
"{source}"\n{target_lang} translation: "\
"""

rt_ins_prefix = """\
Score the following machine translation from {source_lang} to {target_lang} with respect to the human reference on a continuous scale from 0 to \
100 that starts with "No meaning preserved", goes through "Some meaning preserved", then "Most meaning preserved and few grammar mistakes", up to \
"Perfect meaning and grammar".\n\n{target_lang} human reference: "{reference}"\n{target_lang} machine translation: "\
"""

srt_ins_prefix = """
Score the following machine translation from {source_lang} to {target_lang} with respect to the human reference on a continuous scale from 0 to \
100 that starts with "No meaning preserved", goes through "Some meaning preserved", then "Most meaning preserved and few grammar mistakes", up to \
"Perfect meaning and grammar".\n\n{source_lang} source: "{source}"\n{target_lang} human reference: "{reference}"\n{target_lang} machine translation: "\
"""

st_prefix = "{source} ="
rt_prefix = "{reference} ="
srt_prefix = "{source} = {reference} ="

def initialize(model_name):
    trust_remote_code = "baichuan" in model_name
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if "llama" in model_name:
        tokenizer.pad_token = tokenizer.unk_token
    elif "mistral" in model_name:
        tokenizer.pad_token_id = 2

    return model, tokenizer

def load_file(path):
    with open(path) as f:
        content = [l.strip() for l in f]
    return content

def format_prefix(prefix_type, lang_pair, tokenizer, sources, references, chat_tag=""):
    if prefix_type == "ins_T":
        prefix_template = t_ins_prefix
    elif prefix_type == "ins_S-T":
        prefix_template = st_ins_prefix
    elif prefix_type == "ins_R-T":
        prefix_template = rt_ins_prefix
    elif prefix_type == "ins_S-R-T":
        prefix_template = srt_ins_prefix
    elif prefix_type == "S-T":
        prefix_template = st_prefix
    elif prefix_type == "R-T":
        prefix_template = rt_prefix
    elif prefix_type == "S-R-T":
        prefix_template = srt_prefix
    else:
        prefix_template = ""
    src_lang, tgt_lang = lang_pair.split("-")
    src_lang, tgt_lang = LANG_CODES[src_lang], LANG_CODES[tgt_lang]
    prefix = [
        chat_tag + prefix_template.format(
            source_lang=src_lang, target_lang=tgt_lang, source=src, reference=ref
        ) for src, ref in zip(sources, references)
    ]
    prefix_length = [len(id) for id in tokenizer(prefix).input_ids]
    return prefix, prefix_length

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-name", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("-l", "--lang-pair", type=str, default="en-de")
    parser.add_argument("-o", "--output-file", type=str, default="mistral_7b_t_loss-src.seg.score")
    parser.add_argument("-b", "--batch-size", type=int, default=1)
    parser.add_argument("-p", "--prefix", type=str, default="")
    args = parser.parse_args()

    model_name = args.model_name
    lang_pair = args.lang_pair
    output_file = args.output_file
    batch_size = args.batch_size
    prefix_type = args.prefix

    model, tokenizer =initialize(model_name)
    system_outputs_dir = f"wmt22/system-outputs/{lang_pair}"
    source_file = f"wmt22/sources/{lang_pair}.txt"
    reference_file = f"wmt22/references/{lang_pair}.refA.txt"
    metric_score_dir = f"wmt22/metric-scores/{lang_pair}"
    sources = load_file(source_file)
    references = load_file(reference_file)
    chat_tag = "[INST] " if ("llama" in model_name or "mistral" in model_name) and ("chat" in model_name or "Instruct" in model_name) else ""

    sys_score = {}
    output = []
    for file in glob.glob(f"{system_outputs_dir}/*.txt"):
        sys_name = os.path.basename(file)[:-4]
        if sys_name in ["DLUT", "NiuTrans", "QUARTZ_TuneReranking", "chrf_bestmbr"]:
            continue
        loss = []
        start_time = time.time()
        sys_outputs = load_file(file)
        for i in range(0, len(sys_outputs), batch_size):
            translations = sys_outputs[i: i + batch_size]
            src = sources[i: i + batch_size]
            ref = references[i: i + batch_size]
            prefixes, prefix_lengths = format_prefix(prefix_type, lang_pair, tokenizer, src, ref, chat_tag)
            loss.extend(forward_loss(model, tokenizer, translations, prefixes, prefix_lengths))
        output.extend([f"{sys_name}\t-{p}\n" for p in loss])
        sys_score[sys_name] = -sum(loss) / len(loss)
        elapsed_time = time.time() - start_time
        print(f"processed {sys_name}'s outputs in {elapsed_time:.3f}s")
    
    with open(f"{metric_score_dir}/{output_file}", "w") as f:
        f.writelines(output)
    output_file = output_file.replace("seg.score", "sys.score")
    with open(f"{metric_score_dir}/{output_file}", "w") as f:
        for k, v in sys_score.items():
            f.write(f"{k}\t{v}\n")


def forward_loss(model, tokenizer, translations, prefix, prefix_length):
    translations = [p + s for p, s in zip(prefix, translations)]
    inputs = tokenizer(translations, padding=True, return_tensors="pt")
    input_ids = inputs.input_ids.cuda()
    attn_mask = inputs.attention_mask.cuda()
    with torch.no_grad():
        output = model(input_ids, attn_mask)
    logits = output.logits
    shift_logits = logits[:, :-1, :]
    input_ids[attn_mask == 0] = -100
    labels = input_ids.clone()
    for label, p_len in zip(labels, prefix_length):
        label[:p_len] = -100
    labels = labels[:, 1:]

    loss_fct = nn.CrossEntropyLoss(reduction="sum")
    loss = [loss_fct(logit, label) for logit, label in zip(shift_logits, labels)]
    loss = torch.tensor(loss, dtype=torch.float32)
    return loss.tolist()


if __name__ == "__main__":
    main()
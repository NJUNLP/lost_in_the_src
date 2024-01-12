import json
import re
import fire
import pandas as pd
from functools import partial
from sacremoses import MosesTokenizer

def sp_mr(data_path="data/wmt22_mqm_zh-en_500.pkl", result_path="results/gpt4_no_ref_stratified_sample_seed42_500.pkl", type=""):
    test_data = pd.read_pickle(data_path)
    results = preprocess_results(result_path, test_data, type)
    df = pd.concat([test_data[["source", "candidate", "span", "category", "severity"]], results], axis=1)

    num_predict = num_label = num_intersection = 0
    num_major = num_major_intersection = 0
    for sample in df.itertuples():
        span_labels, label_length = span_label(sample)
        major_label_spans, major_length = span_label(sample, major=True)
        num_major += sum(major_length)
        num_label += sum(label_length)

        for pred_span in sample.pred_span:
            s = sample.source + "\t" + sample.candidate
            pred_begin = s.find(pred_span)
            if pred_begin > len(sample.source):
                num_predict += len(pred_span.split(" "))
            else:
                num_predict += len(pred_span)
            pred_end = pred_begin + len(pred_span)
            num_intersection += intersection((pred_begin, pred_end), span_labels, s, len(sample.source))
            num_major_intersection += intersection((pred_begin, pred_end), major_label_spans, s, len(sample.source))

    span_precision = num_intersection / num_predict
    span_recall = num_intersection / num_label
    sf1 = (2 * num_intersection) / (num_predict + num_label)
    major_precision = num_major_intersection / num_predict
    major_recall = num_major_intersection / num_major
    mf1 = (2 * num_major_intersection) / (num_predict + num_major)

    print(f"SP: {num_intersection:>5} / {num_predict:>5} = {span_precision:.3f}")
    print(f"SR: {num_intersection:>5} / {num_label:>5} = {span_recall:.3f}")
    print(f"S-F1: {sf1:.3f}")
    print(f"MP: {num_major_intersection:>5} / {num_predict:>5} = {major_precision:.3f}")
    print(f"MR: {num_major_intersection:>5} / {num_major:>5} = {major_recall:.3f}")
    print(f"M-F1: {mf1:.3f}")

def mcc(data_path="../mqm-processed/wmt22_mqm_zh-en_500.pkl", result_path="results/gpt3.5_ref_stratified_sample_cate5_seed42_500.pkl", tgt="en"):

    def parse_gold(sample):
        translation_tok = tokenizer.tokenize(sample.candidate, escape=False)
        tag = [1] * len(translation_tok)
        if not sample.category:
            return tag
        pos = []
        p_begin = p_end = 0
        for token in translation_tok:
            p_begin = sample.candidate.find(token, p_begin)
            assert p_begin >= 0
            p_end = p_begin + len(token)
            pos.append((p_begin, p_end))
            p_begin = p_end
        for span, category in zip(sample.span, sample.category):
            if category == "Accuracy/Omission" or category == "Source error":
                continue
            begin = span.find("<v>")
            end = span.find("</v>") - 3
            for i, p in enumerate(pos):
                if p[0] < end and p[1] > begin:
                    tag[i] = 0
        return tag

    test_data = pd.read_pickle(data_path)
    results = preprocess_results(result_path, test_data, type)
    df = pd.concat([test_data[["source", "candidate", "span", "category", "severity"]], results], axis=1)

    tokenizer = MosesTokenizer(lang=tgt)
    for sample in df.itertuples():
        tag_gold = parse_gold(sample)

def preprocess_results(result_path, test_data, type):
    results = pd.read_pickle(result_path)
    if type:
        process_type_partial = partial(process_type, type=type)
        results = results.apply(process_type_partial)
    df = pd.concat([test_data[["source", "candidate"]], results], axis=1)
    pred_span, pred_category, pred_severity = [], [], []
    for sample in df.itertuples():
        p_span_list, p_category_list, p_severity_list = [], [], []
        if " - " in sample.output and ("major/" in sample.output or "minor/" in sample.output):
            # pred = sample.output.split("; ")
            sep_match = re.finditer(r" - (major|minor)/.+?; ", sample.output)
            begin_pos = [m.end(0) for m in sep_match]
            end_pos = [p - 2 for p in begin_pos]
            begin_pos.insert(0, 0)
            end_pos += [len(sample.output)]
            pred = [sample.output[b: e] for b, e in zip(begin_pos, end_pos)]
            pred = list(set(pred))
            for e in pred:
                if len(e.rsplit(" - ", 1)) != 2:
                    print("skip one predicted error: split '-' error")
                    # print(sample)
                    continue
                p_span, p_category = e.strip().rsplit(" - ", 1)
                pred_error_content = re.search(r"'(.*)'", p_span)
                if pred_error_content is None:
                    pred_error_content = re.search(r"'(.*)", p_span)
                if pred_error_content is None:
                    print("skip one predicted error: predicted span format error")
                    print(sample)
                    continue
                p_span = pred_error_content.group(1)
                if len(p_category.split("/")) != 2:
                    print("skip one predicted error: split '/' error")
                    # print(sample)
                    continue
                p_severity, p_category = p_category.split("/")

                s = sample.source + "\t" + sample.candidate
                pred_begin = s.find(p_span)
                if pred_begin < 0:
                    print("skip one predicted error: no predicted span")
                    # print(sample)
                    continue
                p_span_list.append(p_span)
                p_category_list.append(p_category)
                p_severity_list.append(p_severity)
        pred_span.append(p_span_list)
        pred_category.append(p_category_list)
        pred_severity.append(p_severity_list)

    processed_results = pd.DataFrame({
        "pred_span": pred_span,
        "pred_category": pred_category,
        "pred_severity": pred_severity,
    }, index=results.index)
    return processed_results


def process_type(output, type):
    if type == "polish":
        p = re.search(r"\nErrors:", output)
        if p is not None:
            r = output[p.end(0):].strip()
        else:
            print("No 'Errors:' in the output")
            r = "None"
        return r
    else:
        return output

def span_label(sample, major=False):
    label_spans = []
    length = []
    if not sample.category:
        return label_spans, length
    for span, category, severity in zip(sample.span, sample.category, sample.severity):
        if major and severity != "major":
            continue
        if category == "Accuracy/Omission" or category == "Source error":
            s = align_span_src(span, sample.source) + "\t" + sample.candidate
        else:
            s = sample.source + "\t" + span
        begin = s.find("<v>")
        end = s.find("</v>") - 3
        label_spans.append((begin, end))
        if category == "Accuracy/Omission" or category == "Source error":
            length.append(end - begin)
        else:
            length.append(len(s[begin:end].split(" ")))

    return label_spans, length


def no_error_prf1(data_path="../mqm-processed/wmt22_mqm_zh-en.pkl", result_path="results/gpt3.5_no_ref_sample.pkl"):
    test_data = pd.read_pickle(data_path)
    results = pd.read_pickle(result_path)
    df = pd.concat([test_data["category"], results], axis=1)
    tp = fp = fn = 0
    for sample in df.itertuples():
        if "-" in sample.output and ("major/" in sample.output or "minor/" in sample.output):
            if not sample.category:
                fn += 1
        elif not sample.category:
            tp += 1
        else:
            fp += 1
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)
    print(tp)
    print(tp + fp)
    print(tp + fn)
    print(f"P: {precision:.3f}")
    print(f"R: {recall:.3f}")
    print(f"F1: {f1:.3f}")


def save_results_txt(output_path, data_path="../mqm-processed/wmt22_mqm_zh-en_100.pkl", result_path="results/gpt3.5_no_ref3_100.pkl"):
    test_data = pd.read_pickle(data_path)
    results = pd.read_pickle(result_path)
    df = pd.concat([test_data[["source", "candidate", "reference", "span", "category", "severity"]], results], axis=1)
    with open(output_path, "w") as f:
        for sample in df.itertuples():
            f.write(f"Sample#{sample.Index}\n")
            f.write(sample.source + "\n")
            f.write(sample.reference + "\n")
            f.write(sample.candidate + "\n")
            if sample.span:
                for span, cate, seve in zip(sample.span, sample.category, sample.severity):
                    f.write(f"{span}\t{cate}\t{seve}\n")
            else:
                f.write("No error\n")
            f.write(sample.output + "\n\n")


def align_span_src(span, source):
    span_src = span.replace("<v>", "").replace("</v>", "")
    if span_src == source:
        return span
    else:
        i = j = 0
        s = ""
        while i < len(span) and j < len(source):
            if span[i] == source[j]:
                s += source[j]
                i += 1
                j += 1
            elif span[i] == "<" and i < len(span) - 3 and (span[i + 1: i + 3] == "v>" or span[i + 1: i + 3] == "/v"):
                if span[i: i + 3] == "<v>":
                    s += "<v>"
                    i += 3
                elif span[i: i + 4] == "</v>":
                    s += "</v>"
                    i += 4
                else:
                    print("should not be here! no <v> or </v>")
                    exit(0)
            elif source[j] == " ":
                s += source[j]
                j += 1
            elif (source[j] == "‘" or source[j] == "’") and span[i] == "'":
                s += span[i]
                i += 1
                j += 1
            # elif span[i] == " ":
            #     i += 1
            else:
                print(span)
                print(source)
                print("should not be here!")
                exit(0)
    return s

def intersection(pred, labels, s, src_len):
    n = 0
    if labels:
        for l in labels:
            if pred[0] < l[1] and pred[1] > l[0]:
                begin = max(pred[0], l[0])
                end = min(pred[1], l[1])
                if begin > src_len:
                    n += len(s[begin:end].split(" "))
                else:
                    n += begin - end
    return n


if __name__ == "__main__":
    fire.Fire({
        "sp_mr": sp_mr,
        "no_error_prf1": no_error_prf1,
        "save_results_txt": save_results_txt,
        "mcc": mcc
    })

def legacy():
    result_path = "results/gpt3.5_no_ref2_500.json"
    with open(result_path) as f:
        results = json.load(f)
    
    def lspan(errors, major=False):
        span = []
        length = []
        if errors[0] == "No-error":
            return span, length
        for e in errors:
            e_split = e.split("\t")
            if (major and e_split[4] == "Major") or not major:
                s = e_split[1] + "\t" + e_split[2]
                begin = s.find("<v>")
                end = s.find("</v>") - 3
                span.append((begin, end))
                length.append(len(s[begin:end].split(" ")))
        return span, length

    num_predict = num_intersection = 0
    num_major = num_major_intersection = 0
    pred_noerror = label_noerror = 0
    for result in results:
        pred = result["output"]
        label_spans, _ = lspan(result["error"])
        major_label_span, major_length = lspan(result["error"], True)
        num_major += sum(major_length)
        if " - " in pred and "/" in pred:
            pred_errors = pred.split(";")
            for e in pred_errors:
                span, category = e.strip().split(" - ")
                span = re.search(r"'(.*)'", span)
                assert span is not None
                span = span.group(1)
                severity, category = category.split("/")

                s = result["source"] + "\t" + result["translation"]
                span_begin = s.find(span)
                if span_begin < 0:
                    print("skip one predicted error")
                    print(result)
                    continue
                num_predict += len(span.split(" "))
                span_end = span_begin + len(span)
                num_intersection += intersection((span_begin, span_end), label_spans, s, len(result["source"]))
                num_major_intersection += intersection((span_begin, span_end), major_label_span, s, len(result["source"]))
        else:
            pred_noerror += 1
        if result["error"][0] == "No-error":
            label_noerror += 1

    span_precision = num_intersection / num_predict
    major_recall = num_major_intersection / num_major
    print(num_intersection, num_predict)
    print(num_major_intersection, num_major)
    print(f"SP: {span_precision:.3f}")
    print(f"MR: {major_recall:.3f}")
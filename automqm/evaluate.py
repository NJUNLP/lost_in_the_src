import json
import re
import fire
import pandas as pd
import os
from functools import partial
from sacremoses import MosesTokenizer

CATEGORIES = ["accuracy", "fluency", "terminology", "style", "locale convention", "source error", "non-translation"]

class Evaluator:
    def __init__(self, data_path="data/wmt22_mqm_zh-en_3200.pkl", result_path="results/llama3_8b_rt_stratified_seed42.pkl", type="") -> None:
        self.data_path = data_path
        self.result_path = result_path
        self.test_data = pd.read_pickle(data_path)
        self.results = preprocess_results(result_path, self.test_data, type)
        self.df = pd.concat([self.test_data[["source", "candidate", "span", "category", "severity"]], self.results], axis=1)

    def sf1_mf1(self, src="zh"):
        num_predict = num_label = num_intersection = 0
        num_major_predict = num_major = num_major_intersection = 0
        for sample in self.df.itertuples():
            span_labels, label_length = span_label(sample)
            major_label_spans, major_length = span_label(sample, major=True)
            num_major += sum(major_length)
            num_label += sum(label_length)

            for pred_pos, pred_span, pred_severity in zip(sample.pred_pos, sample.pred_span, sample.pred_severity):
                pred_begin, pred_end = pred_pos
                s = sample.source + "\t" + sample.candidate
                if pred_begin > len(sample.source) or src != "zh":
                    span_length = len(pred_span.split(" "))
                else:
                    span_length = len(pred_span)
                num_predict += span_length
                num_intersection += intersection((pred_begin, pred_end), span_labels, s, len(sample.source))
                if pred_severity == "major":
                    num_major_predict += span_length
                    num_major_intersection += intersection((pred_begin, pred_end), major_label_spans, s, len(sample.source))

        span_precision = num_intersection / num_predict
        span_recall = num_intersection / num_label
        sf1 = (2 * num_intersection) / (num_predict + num_label)
        major_precision = num_major_intersection / num_major_predict
        major_recall = num_major_intersection / num_major
        mf1 = (2 * num_major_intersection) / (num_major_predict + num_major)

        print(f"SP: {num_intersection:>5} / {num_predict:>5} = {span_precision:.3f}")
        print(f"SR: {num_intersection:>5} / {num_label:>5} = {span_recall:.3f}")
        print(f"S-F1: {sf1:.3f}")
        print(f"MP: {num_major_intersection:>5} / {num_major_predict:>5} = {major_precision:.3f}")
        print(f"MR: {num_major_intersection:>5} / {num_major:>5} = {major_recall:.3f}")
        print(f"M-F1: {mf1:.3f}")
        return num_intersection, num_predict, num_label, num_major_intersection, num_major_predict, num_major

    def mcc(self, tgt="en"):

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
        
        def parse_predicted(sample):
            translation_tok = tokenizer.tokenize(sample.candidate, escape=False)
            tag = [1] * len(translation_tok)
            if not sample.pred_category:
                return tag
            pos = []
            p_begin = p_end = 0
            for token in translation_tok:
                p_begin = sample.candidate.find(token, p_begin)
                assert p_begin >= 0
                p_end = p_begin + len(token)
                pos.append((p_begin, p_end))
                p_begin = p_end
            src_length = len(sample.source)
            for pred_pos in sample.pred_pos:
                pred_begin, pred_end = pred_pos
                if pred_begin > src_length:
                    pred_begin -= src_length + 1
                    pred_end -= src_length + 1
                    for i, p in enumerate(pos):
                        if p[0] < pred_end and p[1] > pred_begin:
                            tag[i] = 0
            return tag

        tokenizer = MosesTokenizer(lang=tgt)
        tag_gold, tag_pred = [], []
        for sample in self.df.itertuples():
            tag_gold.extend(parse_gold(sample))
            tag_pred.extend(parse_predicted(sample))
        assert len(tag_gold) == len(tag_pred)

        tp = tn = fp = fn = 0
        for pred_tag, gold_tag in zip(tag_pred, tag_gold):
            if pred_tag == 1:
                if pred_tag == gold_tag:
                    tp += 1
                else:
                    fp += 1
            else:
                if pred_tag == gold_tag:
                    tn += 1
                else:
                    fn += 1
        mcc_numerator = (tp * tn) - (fp * fn)
        mcc_denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        mcc = mcc_numerator / mcc_denominator
        print(f"MCC tp: {tp}")
        print(f"MCC tn: {tn}")
        print(f"MCC fp: {fp}")
        print(f"MCC fn: {fn}")
        print(f"MCC: {mcc:.3f}")
        return tp, tn, fp, fn

    def score_error(self):
        avg_error = 0
        for sample in self.df.itertuples():
            score_gold = score_label(sample.category, sample.severity)
            score_pred = score_label(sample.pred_category, sample.pred_severity)
            avg_error += abs(score_gold - score_pred)
        avg_error /= len(self.df)
        print(f"avg score error: {avg_error:.3f}")

    def major_count_error(self):
        avg_error = 0
        for sample in self.df.itertuples():
            num_gold = sum(s == "major" for s in sample.severity)
            num_pred = sum(s == "major" for s in sample.pred_severity)
            avg_error += abs(num_gold - num_pred)
        avg_error /= len(self.df)
        print(f"avg major count error: {avg_error:.2f}")

    def test_all(self, src="zh", tgt="en"):
        print()
        print(os.path.basename(self.result_path))
        self.sf1_mf1(src)
        self.mcc(tgt)
        self.score_error()
        self.major_count_error()

    def category_prf1(self):
        categories = ["accuracy", "fluency", "terminology", "style", "locale convention", "non-translation", "source error", "other", "no-error"]
        confusion_dict = {c: [0, 0, 0, 0, 0] for c in categories} #TP, FN, FP, num, major_num
        for sample in self.df.itertuples():
            labels = [l.split("/")[0].lower() for l in sample.category]
            for label, severity in zip(labels, sample.severity):
                if severity == "major":
                    confusion_dict[label][4] += 1
            labels_dup = labels[:]
            for p in sample.pred_category:
                if p in labels_dup:
                    confusion_dict[p][0] += 1
                    confusion_dict[p][3] += 1
                    labels_dup.remove(p)
                else:
                    confusion_dict[p][2] += 1
            for l in labels_dup:
                confusion_dict[l][1] += 1
                confusion_dict[l][3] += 1
            if not labels:
                if sample.pred_category:
                    confusion_dict["no-error"][1] += 1
                else:
                    confusion_dict["no-error"][0] += 1
                confusion_dict["no-error"][3] += 1
            elif not sample.pred_category:
                confusion_dict["no-error"][2] += 1

        for k, v in confusion_dict.items():
            precision = v[0] / (v[0] + v[2]) if v[0] > 0 else 0
            recall = v[0] / (v[0] + v[1]) if v[0] > 0 else 0
            f1 = 2 * v[0] / (2 * v[0] + v[1] + v[2]) if v[0] > 0 else 0
            print(f"{k}: P: {precision} R: {recall} F1: {f1} num: {v[3]}")
        print(confusion_dict)
        return confusion_dict


    def save_scores(self, prefix="automqm_no_src_ref_mistral-7b-src", output_dir="mt-metrics-eval-v2-3200/wmt22/metric-scores/en-de"):
        output_seg_file = f"{output_dir}/{prefix}.seg.score"
        output_sys_file = f"{output_dir}/{prefix}.sys.score"
        df = pd.concat([self.test_data[["source", "system"]], self.results], axis=1)
        df["pred_score"] = df.apply(lambda r: score_label(r["pred_category"], r["pred_severity"]), axis=1)
        df[["system", "pred_score"]].to_csv(output_seg_file, sep="\t", header=False, index=False)

        system_scores = df.groupby("system")["pred_score"].mean()
        system_scores.to_csv(output_sys_file, sep="\t", header=False)

    def save_results_txt(self, output_path):
        results = pd.read_pickle(self.result_path)
        df = pd.concat([self.test_data[["source", "candidate", "reference", "span", "category", "severity"]], results], axis=1)
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

def preprocess_results(result_path, test_data, type):
    if result_path.endswith("json"):
        with open(result_path) as f:
            results = json.load(f)
        results = pd.Series(results, name="output", dtype="string")
    else:
        results = pd.read_pickle(result_path)
    if type:
        process_type_partial = partial(process_type, type=type)
        results = results.apply(process_type_partial)
    df = pd.concat([test_data[["source", "candidate"]], results], axis=1)
    pred_span, pred_category, pred_severity, pred_pos = [], [], [], []
    for sample in df.itertuples():
        p_span_list, p_category_list, p_severity_list, p_pos_list = [], [], [], []
        if " - " in sample.output and ("major/" in sample.output or "minor/" in sample.output):
            # pred = sample.output.split("; ")
            sep_match = re.finditer(r" - (major|minor)/.+?; ", sample.output)
            begin_pos = [m.end(0) for m in sep_match]
            end_pos = [p - 2 for p in begin_pos]
            begin_pos.insert(0, 0)
            end_pos += [len(sample.output)]
            pred = [sample.output[b: e] for b, e in zip(begin_pos, end_pos)]
            # pred = list(set(pred))
            end_pos_dict = {}
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
                    # print(sample)
                    continue
                p_span = pred_error_content.group(1)
                if len(p_category.split("/")) != 2:
                    print("skip one predicted error: split '/' error")
                    # print(sample)
                    continue
                p_severity, p_category = p_category.split("/")
                p_category = p_category.rstrip(".;")
                if p_category not in CATEGORIES:
                    print(f"skip one predicted error: no predicted category {p_category}")
                    # print(sample)
                    continue

                s = sample.source + "\t" + sample.candidate
                prev_end_pos = end_pos_dict.get(p_span, 0)
                pred_begin = s.find(p_span, prev_end_pos)
                if pred_begin < 0:
                    print("skip one predicted error: no predicted span")
                    # print(sample)
                    continue
                pred_end = pred_begin + len(p_span)
                end_pos_dict[p_span] = pred_end
                p_span_list.append(p_span)
                p_category_list.append(p_category)
                p_severity_list.append(p_severity)
                p_pos_list.append((pred_begin, pred_end))
        pred_span.append(p_span_list)
        pred_category.append(p_category_list)
        pred_severity.append(p_severity_list)
        pred_pos.append(p_pos_list)

    processed_results = pd.DataFrame({
        "pred_span": pred_span,
        "pred_category": pred_category,
        "pred_severity": pred_severity,
        "pred_pos": pred_pos
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

def score_label(categories, severities):
    score = -0.0
    for cate, severity in zip(categories, severities):
        if severity == "major":
            score -= 25 if cate == "Non-translation" or cate == "non-translation" else 5
        elif severity == "minor":
            score -= 0.1 if cate == "Fluency/Punctuation" else 1
    return score

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
    fire.Fire(Evaluator)
import pandas as pd
import json

def main():
    testset_path = "data/wmt22_mqm_en-de.pkl"
    result_path = "results/ende/Llama2-7b-InsScore_nodup_rt.json"
    prefix = "Llama2-7b-InsScore_nodup_rt-refA"
    output_dir = "results/ende"
    add_None = True
    human_score_path = "results/ende/en-de.mqm.seg.score"
    

    output_seg_file = f"{output_dir}/{prefix}.seg.score"
    output_sys_file = f"{output_dir}/{prefix}.sys.score"
    testset = pd.read_pickle(testset_path)
    with open(result_path) as r:
        results_txt = json.load(r)
    testset["pred_score"] = [score_label(res) for res in results_txt]
    if add_None and human_score_path:
        with open(human_score_path) as f:
            first_line = f.readline().strip().split("\t")
            sys = first_line[0]
            none_idx = [0] if first_line[1] == "None" else []
            for i, l in enumerate(f, 1):
                l = l.strip().split("\t")
                if l[0] != sys:
                    break
                if l[1] == "None":
                    none_idx.append(i)
        num_sys = testset["system"].nunique()
        lines = i
        pred_score = [-999] * (num_sys * lines)
        j = 0

        for i in range(len(pred_score)):
            if i % lines in none_idx:
                continue
            else:
                pred_score[i] = testset["pred_score"].iloc[j]
                j += 1
        usystems = testset["system"].unique()
        systems = [item for item in usystems for _ in range(lines)]
        output = [f"{sys}\t{score}\n" for sys, score in zip(systems, pred_score)]
        with open(output_seg_file, "w") as o:
            o.writelines(output)
    else:
        testset[["system", "pred_score"]].to_csv(output_seg_file, sep="\t", header=False, index=False)

    system_scores = testset.groupby("system")["pred_score"].mean()
    system_scores.to_csv(output_sys_file, sep="\t", header=False)


def score_label(text: str):
    score = -0.0
    score -= text.count("Major/minor: Minor")
    score -= 5 * text.count("Major/minor: Major")
    return score

if __name__ == "__main__":
    main()
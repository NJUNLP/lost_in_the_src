import pandas as pd
import numpy as np
import sys

def main():
    times = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    data_path = "data/wmt21_mqm_zh-en.pkl"
    df = pd.read_pickle(data_path)
    # df["category"] = df["category"].apply(lambda c: [] if c[0] == "No-error" else c)
    df["category"] = df["category"].apply(lambda c: ["No-error"] if c == [] else c)
    rng = np.random.default_rng(42)
    data_size = len(df)
    time = 0
    while time < times:
        index = rng.choice(data_size, size=4, replace=False)
        while not check_icl_set(df.iloc[index], cat_diversity=4):
            index = rng.choice(data_size, size=4, replace=False)
        time += 1
    index = np.sort(index)
    print(index)
    output_examples(df.iloc[index])

def output_examples(examples):
    for row in examples.itertuples():
        print(row.source)
        print(row.candidate)
        print(row.reference)
        if len(row.category) > 0:
            for i in range(len(row.category)):
                print(f"{row.span[i]}\t{row.category[i]}\t{row.severity[i]}")
        else:
            print("No-error")
        print()


def check_icl_set(
    examples: pd.DataFrame,
    min_errors=3,
    majmin_threshold=2,
    cat_diversity=2,
    min_clen=20,
    max_clen=400,
):
    # Check if they have the same number of spans as severity/category
    if not examples.apply(
        lambda r:
        len(r["span"]) == len(r["severity"]) and len(
            r["span"]) == len(r["category"]),
        axis=1
    ).all():
        return False

    # Check if there are at least min_errors
    if examples["severity"].apply(lambda svs: len(svs)).sum() < min_errors:
        return False

    # Check that there"s a balance of major and minor errors.
    major_count = examples["severity"].apply(
        lambda svs: sum([s == "major" for s in svs])).sum()
    minor_count = examples["severity"].apply(
        lambda svs: sum([s == "minor" for s in svs])).sum()
    if abs(major_count - minor_count) > majmin_threshold:
        return False

    # Check that at least cat_diversity error types are represented.
    categories = examples["category"].apply(
        lambda cs: [c.split("/")[0] for c in cs])
    represented_error_types = set().union(*categories.tolist())
    if len(represented_error_types) < cat_diversity:
        return False

    top_clen = examples.apply(
        lambda row: max(len(row[s]) for s in ("source", "reference", "candidate")
                        ), axis=1).max()
    bot_clen = examples.apply(
        lambda row: min(len(row[s])
                        for s in ("source", "reference", "candidate")),
        axis=1).min()
    if top_clen > max_clen or bot_clen < min_clen:
        return False

    # All checks passed.
    return True

def check_icl_set_uniform(
    examples,
    min_errors=3,
    majmin_threshold=2,
    cat_diversity=5,
    min_clen=20,
    max_clen=400,
):
    num_shots = len(examples)
    # Check if there are at least min_errors
    num_errors = examples["severity"].apply(lambda svs: len(svs)).sum()
    if num_errors < min_errors:
        return False
    # if num_errors / num_shots < 4:
    #     return False

    # Check that there"s a balance of major and minor errors.
    major_count = examples["severity"].apply(
        lambda svs: sum([s == "major" for s in svs])).sum()
    minor_count = examples["severity"].apply(
        lambda svs: sum([s == "minor" for s in svs])).sum()
    if abs(major_count - minor_count) > majmin_threshold:
        return False

    # Check that at least cat_diversity error types are represented.
    categories = examples["category"].apply(
        lambda cs: [c.split("/")[0] for c in cs])
    represented_error_types = set().union(*categories.tolist())
    if len(represented_error_types) != cat_diversity:
        return False

    top_clen = examples.apply(
        lambda row: max(len(row[s]) for s in ("source", "reference", "candidate")
                        ), axis=1).max()
    bot_clen = examples.apply(
        lambda row: min(len(row[s])
                        for s in ("source", "reference", "candidate")),
        axis=1).min()
    if top_clen > max_clen or bot_clen < min_clen:
        return False

    # All checks passed.
    return True


if __name__ == "__main__":
    main()

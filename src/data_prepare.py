import numpy as np
import pandas as pd


def clean_csv(file_path):
    df = pd.read_json(file_path,  encoding='utf-8')
    df[
        "instruction"
    ] = "If you are a instructor, please answer the questions related to quiz based on the user's query"
    df = df.reset_index(drop=True)
    return df


def add_text_col(df):
    intro = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
    instruction = " ### Instruction: " + df["instruction"]
    input = " ### Input: " + df["input"]
    respones = " ### Response: " + df["output"]
    return intro + instruction + input + respones


if __name__ == "__main__":
    result_df = clean_csv("datasets/finetuned_data/final_merged.json")
    result_df["text"] = result_df.apply(add_text_col, axis=1)
    result_df = result_df[["instruction", "input", "output", "text"]]
    print("Shape of final pre-processed data:", result_df.shape)
    result_df.to_csv("datasets/finetuned_data/final_preprocessed.csv", index=False)
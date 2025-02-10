# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# A minimal instruction finetuning file based on the code in chapter 7

import json
import psutil
from tqdm import tqdm
import pandas as pd 
import urllib.request
from openai import OpenAI
# client = OpenAI(api_key="sk-e470915157d049a792a6049edccf14ed", base_url="https://api.deepseek.com/v1")
# def query_model(prompt, model="deepseek-chat", url="http://localhost:11434/api/chat"):
#     # Create the data payload as a dictionary
#     print(model,prompt)
#     response = client.chat.completions.create(
#         model=model,
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant"},
#             {"role": "user", "content": prompt},
#         ],
#         stream=False
#     )
#     return response.choices[0].message.content

def query_model(prompt, model="llama3", url="http://localhost:11434/api/chat"):
    # Create the data payload as a dictionary
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {     # Settings below are required for deterministic responses
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }
    print(data)
    # Convert the dictionary to a JSON formatted string and encode it to bytes
    payload = json.dumps(data).encode("utf-8")

    # Create a request object, setting the method to POST and adding necessary headers
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    # Send the request and capture the response
    response_data = ""
    with urllib.request.urlopen(request) as response:
        # Read and decode the response
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    return response_data

def format_input(entry):
    # instruction_text = (
    #     f"Below is an instruction that describes a task. "
    #     f"Write a response that appropriately completes the request."
    #     f"\n\n### Instruction:\n{entry['instruction']}"
    # )
    # instruction_text=(
    #     f"基于以下已知信息，简洁和专业的来回答用户的问题。用简体中文回答。"
    # )

    # input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    # return instruction_text + input_text
    input_text=entry['prompt']
    return input_text


def main(file_path):
    csv_path='test_result.csv'
    test_pd=pd.read_csv(csv_path,index_col=0)
    # model = "deepseek-chat"
    model = "qwen2.5"
    scores,rtn_pd = generate_model_scores(test_pd, "model_response", model)
    print(f"Number of scores: {len(scores)} of {len(test_pd)}")
    print(f"Average score: {sum(scores)/len(scores):.2f}\n")
    rtn_pd.to_csv(csv_path.split(".csv")[0]+"_scores"+".csv",encoding='utf-8-sig')


def generate_model_scores(pd_data:pd.DataFrame, json_key, model="deepseek-chat"):
    scores = []
    for i,entry in tqdm(pd_data.iterrows(), desc="Scoring entries"):
        if entry[json_key] == "":
            scores.append(0)
        else:
            prompt = (
                f"Given the input `{format_input(entry)}` "
                f"and correct output `{entry['答案']}`, "
                f"score the model response `{entry[json_key]}`"
                f" on a scale from 0 to 100, where 100 is the best score. "
                f"Respond with the integer number only."
            )
            score = query_model(prompt, model)
            try:
                scores.append(int(score))
            except ValueError:
                print(f"Could not convert score: {score}")
                continue
    pd_data['score']=scores
    return scores,pd_data


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate model responses with ollama"
    )
    parser.add_argument(
        "--file_path",
        required=True,
        help=(
            "The path to the test dataset `.json` file with the"
            " `'output'` and `'model_response'` keys"
        )
    )
    args = parser.parse_args()

    main(file_path=args.file_path)

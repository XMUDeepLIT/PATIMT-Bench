import openai
import requests
import json
import os
import base64
import numpy as np
import glob
import string
from PIL import Image
from io import BytesIO
import traceback
import fasttext
from tqdm import tqdm
"""
Script that use gpt-4o api to generate translation results for each ocr results.
"""

# Function to encode images in binary or base64 format
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def base64_to_image(base64_str):
    byte_data = base64.b64decode(base64_str)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def replace_single_quotes(text):
    # This pattern matches single quotes not followed by 's' or 't'
    for q in string.punctuation + " ":
        text = text.replace(f"'{q}", f'"{q}').replace(f"{q}'", f'{q}"')
    return text


def augment_period(text):
    # convert multiline string into single line
    text = "".join(line.strip() for line in text.splitlines())
    return (
        text.replace("```", "")
        .replace("json\n", "")
        .replace("”", '\\"')
        .replace("“", '\\"')
        .replace("‘", "\\'")
        .replace("’", "\\'")
        .strip()
    )


def replace_special_characters(text):
    special_chars = string.punctuation + " "
    for char in special_chars:
        text = text.replace(char, f"")
    return text


def gpt_generate(prompt, api_key):
    import os
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.openai.com/v1",
    )
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    output = json.loads(completion.model_dump_json())["choices"][0]["message"][
        "content"
    ]
    # print(output)
    return output

lang_map = {
    'en': ['English', "Chinese"],
    'zh': ['Chinese', 'English']
}



def Args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--n_gpus", type=int, default=8)
    parser.add_argument('--ocr_type', type=str, default='merge')
    return parser.parse_args()



if __name__ == "__main__":
    args = Args()
    local_rank = args.local_rank
    n_gpus = args.n_gpus
    api_keys = [
        'api_key',
    ]
    api_key = api_keys[0]
    print(f"use api key: {api_key}")
    # instruction json file
    SRC_FILE = f'refined_ocr_{local_rank}.jsonl'
    OUTPUT_FILE = f'annotation_{local_rank}.jsonl'

    data = [json.loads(line) for line in open(SRC_FILE).readlines()]
    total = len(data)
    data = data[local_rank::n_gpus]
    print(f"local rank: {local_rank}, total samples: {total},  process samples: {len(data)}")
    exist_len = len(open(OUTPUT_FILE, 'r').readlines()) if os.path.exists(OUTPUT_FILE) else 0
    action = "a" if exist_len > 0 else "w"
    lang_detector = fasttext.load_model("lid.176.ftz")
    print(f"*************Exist length: {exist_len}***************")
    ocr_key = 'merge_ocr' if args.ocr_type == 'merge' else 'ocr'
    print(f'process ocr type={ocr_key}')
    with open(OUTPUT_FILE, action) as fout:
        for i, line in enumerate(tqdm(data)):
            ocr_info = line[ocr_key]
            img_path = line['image']
            if i < exist_len:
                print(f"skip existing [{i}]: {img_path}")
                continue
            else:
                print(f"processing [{i}]: {img_path}")
                img_id = img_path.split(".")[0]
                labels = []
                for ocr in ocr_info:
                    src_lang = lang_detector.predict(ocr['text'], k=1)[0][0].replace("__label__", "")
                    prompt=f"Translate into {lang_map[src_lang][1]}. Only return the translation result: {ocr['text']}"
                    try:
                        tgt_text = gpt_generate(prompt,api_key=api_key)
                        labels.append(
                            {
                                "box": ocr["box"],
                                "src_text": ocr["text"],
                                "tgt_text": tgt_text,
                                "src_lang": lang_map[src_lang][0],
                                "tgt_lang": lang_map[src_lang][1],
                            }
                        )
                    except Exception as e:
                        print(i, ": ", e)
                        continue
                line[ocr_key] = labels
                fout.write(
                    json.dumps(
                        line,
                        default=default_dump,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                fout.flush()
                        

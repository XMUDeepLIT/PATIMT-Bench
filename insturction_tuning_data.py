import json
import PIL.Image as Image
import random
from tqdm import tqdm
import os
import numpy as np
import random
from PIL import Image
import copy
import torch
from typing import Dict
import re

random.seed(42)


DEFAULT_IMAGE_TOKEN = "<image>"
BOX_PLACEHOLDER = "<box_pad>"
SEP_TOKEN = "<|translation|>"

def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def regulate_boxes(image, label, box_format="llava", ocr_key='ocr'):
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    ret = []
    for val in label[ocr_key]:
        try:
            xmin, ymin, xmax, ymax = val["box"]
            if xmin > xmax and ymin > ymax:
                print("exchange min and max box")
                xmin, xmax, ymin, ymax = xmax, xmin, ymax, ymin
                assert xmin < xmax and ymin < ymax, f'box coord error: {[xmin, ymin, xmax, ymax]}'
                
        except Exception as e:
            print(f"val_box: {val['box']}")
            return None
        w, h = image.size
        margin = 0.0
        if "llava" in box_format:
            xmin, ymin, xmax, ymax = (
                round((1 - margin) * (xmin / w), 2),
                round((1 - margin) * (ymin / h), 2),
                round((1 + margin) * (xmax / w), 2),
                round((1 + margin) * (ymax / h), 2),
            )
        elif box_format == "qwen" or "internvl" in box_format:  # qwen
            xmin, ymin, xmax, ymax = (
                int((1 - margin) * xmin / w * 1000),
                int((1 - margin) * ymin / h * 1000),
                int((1 - margin) * xmax / w * 1000),
                int((1 - margin) * ymax / h * 1000),
            )
        elif 'deepseek' in box_format:  # qwen
            xmin, ymin, xmax, ymax = (
                int((1 - margin) * xmin / w * 999),
                int((1 - margin) * ymin / h * 999),
                int((1 - margin) * xmax / w * 999),
                int((1 - margin) * ymax / h * 999),
            )
        elif box_format == "qwen25":
            xmin, ymin, xmax, ymax = (
                int((1 - margin) * xmin),
                int((1 - margin) * ymin),
                int((1 - margin) * xmax),
                int((1 - margin) * ymax),
            )
        else:
            raise ValueError(f"Unknown box format: {box_format}")
        val["box"] = [xmin, ymin, xmax, ymax]
        ret.append(val)
    label[ocr_key] = ret
    return label

def crop_image(pil_image: Image, box, box_format="qwen"):
    base = 1000
    if box_format != "qwen" or 0 < box[0] < 1:
        base = 1
    box = [
        int(box[0] / base * pil_image.width),
        int(box[1] / base * pil_image.height),
        int(box[2] / base * pil_image.width),
        int(box[3] / base * pil_image.height),
    ]
    return pil_image.crop(box)


def format_box(box_list: list, box_format="qwen"):
    """
    input: [x1,y1,x2,y2]
    output: <box>(x1,y1),(x2,y2)</box>
    """
    if isinstance(box_list, str):
        return box_list
    if box_format == "qwen":
        if not (box_list[0] == 0 or 1 <= box_list[0] <= 1000):
            print(f"box should in range(0,1000), but got {box_list}")
            box_list = [max(min(1000, box_list[0]), 0),max(min(1000, box_list[1]), 0),max(min(1000, box_list[2]), 0),max(min(1000, box_list[3]), 0)]
        return f"<|box_start|>({box_list[0]},{box_list[1]}),({box_list[2]},{box_list[3]})<|box_end|>"
    if box_format == "internvl25":
        if not (box_list[0] == 0 or 1 <= box_list[0] <= 1000):
            print(f"box should in range(0,1000), but got {box_list}")
            box_list = [max(min(1000, box_list[0]), 0),max(min(1000, box_list[1]), 0),max(min(1000, box_list[2]), 0),max(min(1000, box_list[3]), 0)]
        return f"<box>[[{box_list[0]}, {box_list[1]}, {box_list[2]}, {box_list[3]}]]</box>"
    if "deepseek" in box_format:
        if not (box_list[0] == 0 or 1 <= box_list[0] < 1000):
            print(f"box should in range(0,999), but got {box_list}")
            box_list = [max(min(999, box_list[0]), 0),max(min(999, box_list[1]), 0),max(min(999, box_list[2]), 0),max(min(999, box_list[3]), 0)]
        return f"<|det|>[[{box_list[0]}, {box_list[1]}, {box_list[2]}, {box_list[3]}]]<|/det|>"
    if box_format == "qwen25":
        box_list = [max(0, box) for box in box_list]
        return f"[{box_list[0]}, {box_list[1]}, {box_list[2]}, {box_list[3]}]"
    elif "llava" in box_format:  # llava
        if not (0 <= box_list[0] < 1):
            print(f"box should in range(0,1), but got {box_list}")
            box_list = [max(min(1, box_list[0]), 0),max(min(1, box_list[1]), 0),max(min(1, box_list[2]), 0),max(min(1, box_list[3]), 0)]
        return f"{box_list}"
    else:
        raise ValueError(f"Unknown box format: {box_format}")


def add_period(text, period="."):
    if not text.endswith(period):
        text = text + period

    return text


def convert_to_json_string(data, data_args):
    json_str = json.dumps(data, indent="\t", ensure_ascii=False, default=default_dump)
    # add markdown format
    json_str = f"```json\n{json_str}\n```"
    return json_str


def format_response(prompt: str, label: dict, text_only=False, data_args=None):
    """
    labels: [
            {"from": "", "value": [
                {"src_lang": "", "tgt_lang": "", "box": ""}
            ]},
        ]
    """
    src_lang = label['value'][0]['src_lang']
    tgt_lang = label['value'][0]['tgt_lang']
    if "<box_pad>" in prompt["value"]:
        if data_args.translation_type in ["box", "text"]:
            while len(prompt["value"].split("<box_pad>")) > 2:
                prompt["value"] = prompt["value"][::-1].replace("<box_pad>"[::-1], "", 1)[
                    ::-1
                ]
            assert len(prompt["value"].split("<box_pad>")) == 2, "multiple <box_pad> token"

    all_box_str = ", ".join(
        [
            f'{format_box(v["box"], box_format=data_args.box_format)}'
            for v in label["value"]
        ]
    )
    
    if data_args.task_type == "rec_and_trans":
        if label is not None:
            if not text_only:  # full rec_and_trans
                if not args.random_question:
                    prompt["value"] = (
                        f'Read all the text in the image, and then translate into {label["value"][0]["tgt_lang"]}.'
                    )
                if data_args.output_json:  # json string
                    # add postfix
                    prompt["value"] = (
                        add_period(prompt["value"])
                        + ' Output result in the following JSON format (note xxx is placeholder for text, x1,y1,x2,y2 are placeholders for coordinate, ... means there may be more contents in the image).\n[{"bbox_2d": [x1,y1,x2,y2], "text_content": xxx, "translation": xxx}, {"bbox_2d": [x1,y1,x2,y2], "text_content": xxx, "translation": xxx}, ...]'
                    )
                    data = [
                        {
                            "bbox_2d": format_box(
                                s["box"], box_format=data_args.box_format
                            ),
                            "text_content": s["src_text"],
                            "translation": s["tgt_text"],
                        }
                        for s in label["value"]
                    ]
                    label["value"] = convert_to_json_string(data, data_args=data_args)
                else:  # plain text
                    # add postfix
                    if data_args.box_format == "qwen":
                        box_str = f"<|box_start|>(x1, y1), (x2, y2)<|box_end|>"
                    elif data_args.box_format == "llava":
                        box_str = f"[x1, y1, x2, y2]"
                    elif "internvl" in data_args.box_format:
                        box_str = f"<box>[x1, y1, x2, y2]</box>"
                    elif "deepseek" in data_args.box_format:
                        box_str = f"<|det|>[x1, y1, x2, y2]<|/det|>"
                    else:
                        raise ValueError(f"Unknown box format={data_args.box_format}")
                    prompt["value"] = (
                            add_period(prompt["value"])
                            + f" Return the recognized text content, translation result and boxes in format: text {SEP_TOKEN} translation {box_str}."
                        )
                    label["value"] = "\n".join(
                        [
                            f"{s['src_text']} {SEP_TOKEN} {s['tgt_text']} {format_box(s['box'], box_format=data_args.box_format)}"
                            for s in label["value"]
                        ]
                    )
            else:  # text rec_and_trans
                if not args.random_question:
                    prompt["value"] = (
                        f'Read the text in position {BOX_PLACEHOLDER}, and then translate into {label["value"][0]["tgt_lang"]}.'
                    )
                if data_args.output_json:
                    prompt["value"] = (
                        add_period(prompt["value"])
                        + ' Output result in the following JSON format (note xxx is placeholder for text, x1,y1,x2,y2 are placeholders for coordinate).\n{"bbox_2d": [x1,y1,x2,y2], "text_content": xxx, "translation": xxx}'
                    )
                    data = [
                        {
                            "bbox_2d": format_box(
                                s["box"], box_format=data_args.box_format
                            ),
                            "text_content": s["src_text"],
                            "translation": s["tgt_text"],
                        }
                        for s in label["value"]
                    ][0]
                    label["value"] = convert_to_json_string(data, data_args=data_args)
                else:
                    prompt["value"] = (
                        add_period(prompt["value"])
                        + " Output only the recognized text content and translation result without any additional descriptions or formatting."
                    )
                    label["value"] = "\n".join(
                        [
                            f"{s['src_text']} {SEP_TOKEN} {s['tgt_text']}"
                            for s in label["value"]
                        ]
                    )
    elif args.task_type == "trans":  # trans
        if label is not None:
            if not text_only:  # full trans
                if not args.random_question:
                    prompt["value"] = f"Translate all the texts in the image."
                if data_args.output_json:  # json string
                    prompt["value"] = (
                        add_period(prompt["value"])
                        + ' Output result in the following JSON format (note xxx is placeholder for text, x1,y1,x2,y2 are placeholders for coordinate, ... means there may be more contents in the image).\n[{"bbox_2d": [x1,y1,x2,y2], "translation": xxx}, {"bbox_2d": [x1,y1,x2,y2], "translation": xxx}, ...]'
                    )
                    data = [
                        {
                            "bbox_2d": format_box(
                                s["box"], box_format=data_args.box_format
                            ),
                            "translation": s["tgt_text"],
                        }
                        for s in label["value"]
                    ]
                    label["value"] = convert_to_json_string(data, data_args=data_args)
                else:  # plain text
                    # add postfix
                    if data_args.box_format == "qwen":
                        box_str = f"<|box_start|>(x1, y1), (x2, y2)<|box_end|>"
                    elif data_args.box_format == "llava":
                        box_str = f"[x1, y1, x2, y2]"
                    elif data_args.box_format == "internvl25":
                        box_str = f"<box>[x1, y1, x2, y2]</box>"
                    elif data_args.box_format == "deepseek":
                        box_str = f"<|det|>[x1, y1, x2, y2]<|/det|>"
                    else:
                        raise ValueError(f"Unknown box format={data_args.box_format}")
                    prompt["value"] = (
                        add_period(prompt["value"])
                        + f" Return the translation result and boxes in format: translation {box_str}."
                    )
                    label["value"] = "\n".join(
                        [
                            f"{s['tgt_text']} {format_box(s['box'], box_format=data_args.box_format)}"
                            for s in label["value"]
                        ]
                    )
            else:  # text trans
                if not args.random_question:
                    prompt["value"] = f"Translate the text located in position {BOX_PLACEHOLDER}."
                # add postfix
                if data_args.output_json:  # json string
                    prompt["value"] = (
                        add_period(prompt["value"])
                        + ' Output result in the following JSON format (note xxx is placeholder for text, x1,y1,x2,y2 are placeholders for coordinate, ... means there may be more contents in the image).\n{"bbox_2d": [x1,y1,x2,y2], "translation": xxx}'
                    )
                    data = [
                        {
                            "bbox_2d": format_box(
                                s["box"], box_format=data_args.box_format
                            ),
                            "translation": s["tgt_text"],
                        }
                        for s in label["value"]
                    ][0]
                    label["value"] = convert_to_json_string(data, data_args=data_args)
                else:  # plain text
                    prompt["value"] = (
                        add_period(prompt["value"])
                        + " Output only the translation result without any additional descriptions or formatting."
                    )
                    label["value"] = "\n".join(
                        [f"{s['tgt_text']}" for s in label["value"]]
                    )
    else:  # rec
        if label is not None:
            if not text_only:  # full trans
                if not args.random_question:
                    prompt["value"] = f"Identify all the text in the image."
                # add postfix
                if data_args.box_format == "qwen":
                    box_str = f"<|box_start|>(x1, y1), (x2, y2)<|box_end|>"
                elif data_args.box_format == "llava":
                    box_str = f"[x1, y1, x2, y2]"
                elif data_args.box_format == "internvl25":
                    box_str = f"<box>[x1, y1, x2, y2]</box>"
                elif data_args.box_format == "deepseek":
                    box_str = f"<|det|>[x1, y1, x2, y2]<|/det|>"
                else:
                    raise ValueError(f"Unknown box format={data_args.box_format}")
                prompt["value"] = (
                    add_period(prompt["value"])
                    + f" Return the recognized text content and boxes in format: text {box_str}."
                )
                if data_args.output_json:
                    data = [
                        {
                            "bbox_2d": format_box(
                                s["box"], box_format=data_args.box_format
                            ),
                            "text_content": s["src_text"],
                        }
                        for s in label["value"]
                    ]
                    label["value"] = convert_to_json_string(data, data_args=data_args)
                else:
                    label["value"] = "\n".join(
                        [
                            f"{s['src_text']} {format_box(s['box'], box_format=data_args.box_format)}"
                            for s in label["value"]
                        ]
                    )

            else:  # text trans
                if not args.random_question:
                    prompt["value"] = f"Identify the text in position {BOX_PLACEHOLDER}."
                prompt["value"] = (
                    add_period(prompt["value"])
                    + " Output only the recognized text content without any additional descriptions or formatting."
                )
                if data_args.output_json:
                    data = [
                        {
                            "bbox_2d": format_box(
                                s["box"], box_format=data_args.box_format
                            ),
                            "text_content": s["src_text"],
                        }
                        for s in label["value"]
                    ][0]
                    label["value"] = convert_to_json_string(data, data_args=data_args)
                else:
                    label["value"] = "\n".join(
                        [f"{s['src_text']}" for s in label["value"]]
                    )
    try:
        prompt["value"] = (
            prompt["value"]
            .replace(BOX_PLACEHOLDER, all_box_str)
            .replace(f"{DEFAULT_IMAGE_TOKEN}\n", "")
            .replace("<src_lang>", src_lang)
            .replace("<tgt_lang>", tgt_lang)
        )
    except:
        print('error')
    # print(prompt["value"])
    return prompt, label


def process_full_translation(prompts: dict, label: dict, data_args=None):
    """conversation: {from: '', value: ''}"""
    input_prompt = copy.deepcopy(random.choice(prompts).strip())
    prompt, label = format_response(
        prompt={'from': 'human', 'value': input_prompt}, label=label, text_only=False, data_args=data_args
    )
    return prompt, label


def cal_area(box):
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


def to_eight_boxes(boxes):
    if isinstance(boxes[0], list):
        return boxes
    x1, y1, x2, y2 = boxes[0], boxes[1], boxes[2], boxes[3]
    return [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]


def process_text_translation(
    prompts: dict, label: dict, max_conv_turn=5, data_args=None
):
    """
    Replace question with box translation
    conversation: {from: '', value: ''}
    return:"""
    conv = []
    ### sample ###
    sample_strategy = "weight"
    if sample_strategy == "weight":
        max_conv_turn = min(max_conv_turn, len(label["value"]))
        weights = [cal_area(val["box"]) for val in label["value"]]
        total = sum(weights)
        norm_weights = [w / total for w in weights]
        sampled_items = set()
        try_time = 0
        while len(sampled_items) < max_conv_turn:
            sampled_indices = random.choices(
                range(len(label["value"])), weights=norm_weights, k=max_conv_turn
            )
            try_time += 1
            if try_time > 50:
                # print("too many times to sample")
                sampled_items = random.sample(
                    range(len(label["value"])), k=max_conv_turn
                )
                break
            for idx in sampled_indices:
                if idx not in sampled_items:
                    sampled_items.add(idx)
                    if len(sampled_items) == max_conv_turn:
                        break
        label["value"] = [label["value"][i] for i in sampled_items]
    else:
        label["value"] = sorted(
            label["value"], key=lambda x: cal_area(x["box"]), reverse=True
        )
        max_ = min(len(label["value"]), 3)
        label["value"] = [random.choice(label["value"][:max_])]
    random.shuffle(label["value"])
    for lab in label["value"]:
        input_prompt = copy.deepcopy(random.choice(prompts)).strip() if args.random_question else ''
        prompt, lab = format_response(
            prompt={'from': 'human', 'value': input_prompt},
            label={"from": "gpt", "value": [lab]},
            text_only=True,
            data_args=data_args,
        )
        conv.append(prompt)
        conv.append(lab)
    return conv


class Args:
    def __init__(self):
        self.task_type = "rec_and_trans"  # rec / rec_and_trans / trans
        self.box_format = "qwen25" # qwen / qwen25 / llava / internvl25 / deepseekvl2
        self.translation_type = "mix"
        self.box_emb_type = "text"
        self.output_json = True
        self.data_num = 50
        self.dataset = "final"
        self.random_question=True
        self.use_merge_ocr=True


if __name__ == "__main__":
    args = Args()

    SRC_LABEL_FILE = '/path/to/gpt_output'
    postfix = 'jsonl' if 'deepseek' in args.box_format or 'internvl' in args.box_format else 'json'
    result_list = []
    OUTPUT_FILE = f"train-{args.dataset}-{args.data_num}k-{args.box_format}-{args.task_type}-json_{args.output_json}-merge_{args.use_merge_ocr}-random-question_{args.random_question}-mineru.{postfix}"
    full_prompts = open('instruction_data_question/full_trans').readlines()
    text_prompts = open('instruction_data_question/text_trans').readlines()
    max_conv = 1e9
    ocr_key = 'merge_ocr' if args.use_merge_ocr else 'ocr'
    with open(SRC_LABEL_FILE) as f:
        lines = f.readlines()
        json_list = [json.loads(line) for line in lines]
        box_error_num = 0
        process_error_num = 0
        with open(OUTPUT_FILE, "w") as outfile:
            for idx, item in enumerate(tqdm(json_list)):
                result_dict = {}
                c_id = item["id"]
                path = item["image"]
                pil_img = Image.open(path)
                try:
                    item = regulate_boxes(
                        image=pil_img, label=item, box_format=args.box_format, ocr_key=ocr_key
                    )
                except:
                    print("Box Process error")
                    import traceback

                    print(traceback.format_exc())
                    box_error_num += 1
                    continue
                try:
                    conversations = []
                    full_conv = None
                    
                    full_conv = process_full_translation(
                        prompts=copy.deepcopy(full_prompts),
                        label={'from': 'gpt', 'value': copy.deepcopy(item[ocr_key])},
                        data_args=args,
                    )
                   
                    text_conv = process_text_translation(
                        prompts=copy.deepcopy(text_prompts),
                        label={'from': 'gpt', 'value': copy.deepcopy(item[ocr_key])},
                        max_conv_turn=max_conv,
                        data_args=args,
                    )
                    if text_conv is None:
                        continue
                    conversations.extend(text_conv)
                    if full_conv is not None:
                        conversations.extend(full_conv)
                except Exception as e:
                    print("Process error")
                    import traceback

                    print(traceback.format_exc())
                    process_error_num += 1
                    continue
                conversations_new = []
                for conv_id, conv in enumerate(conversations):
                    if conv["from"] == "human":
                        if DEFAULT_IMAGE_TOKEN in conv["value"] and conv_id > 0:
                            conv["value"] = conv["value"].replace(
                                f"{DEFAULT_IMAGE_TOKEN}\n", ""
                            )
                    if 'qwen' in args.box_format or 'deepseek' in args.box_format:
                        if conv['from'] == 'human':
                            conversations_new.append({'role': 'user', 'content': conv['value']})
                        else:
                            conversations_new.append({'role': 'assistant', 'content': conv['value']})
                    elif 'llava' in args.box_format or 'internvl' in args.box_format:
                        if conv['from'] == 'human':
                            conversations_new.append({'from': 'human', 'value': conv['value']})
                        else:
                            conversations_new.append({'from': 'gpt', 'value': conv['value']})
                conversations = conversations_new
                if 'qwen' in args.box_format or 'deepseek' in args.box_format:
                    if DEFAULT_IMAGE_TOKEN not in conversations[0]["content"]:
                        conversations[0]["content"] = "<image>\n" + conversations[0]["content"]
                    if DEFAULT_IMAGE_TOKEN not in conversations[-2]["content"]:
                        conversations[-2]["content"] = "<image>\n" + conversations[-2]["content"]
                    # text trans
                    result_list.append(
                        {
                            "messages": conversations[:-2],
                            "images": [path],
                        }
                    )
                    # full trans
                    result_list.append(
                        {
                            "messages": conversations[-2:],
                            "images": [path],
                        }
                    )
                elif 'llava' in args.box_format:
                    if DEFAULT_IMAGE_TOKEN not in conversations[0]["value"]:
                        conversations[0]["value"] = "<image>\n" + conversations[0]["value"]
                    if DEFAULT_IMAGE_TOKEN not in conversations[-2]["value"]:
                        conversations[-2]["value"] = "<image>\n" + conversations[-2]["value"]
                    result_list.append(
                        {
                            "conversations": conversations[:-2],
                            "image": path,
                        }
                    )
                    result_list.append(
                        {
                            "conversations": conversations[-2:],
                            "image": path,
                        }
                    )
                elif 'internvl' in args.box_format:
                    if DEFAULT_IMAGE_TOKEN not in conversations[0]["value"]:
                        conversations[0]["value"] = "<image>\n" + conversations[0]["value"]
                    if DEFAULT_IMAGE_TOKEN not in conversations[-2]["value"]:
                        conversations[-2]["value"] = "<image>\n" + conversations[-2]["value"]
                    result_list.append(
                        {
                            "id": idx*2,
                            "image": path,
                            "width": pil_img.width,
                            "height": pil_img.height,
                            "conversations": conversations[:-2],
                        }
                    )
                    result_list.append(
                        {
                            "id": idx*2 + 1,
                            "image": path,
                            "width": pil_img.width,
                            "height": pil_img.height,
                            "conversations": conversations[-2:],
                        }
                    )
                
            random.shuffle(result_list)
            print(f"Box Error Num: {box_error_num}")
            print(f"Process Error Num: {process_error_num}")
            if "internvl" in args.box_format or "deepseek" in args.box_format:
                for i in result_list:
                    outfile.write(json.dumps(i,ensure_ascii=False)+'\n')
            else:
                json.dump(
                    result_list, outfile, indent=4, default=default_dump, ensure_ascii=False
                )
            outfile.close()
    print(f"Write {len(result_list)}")
    print("GG")

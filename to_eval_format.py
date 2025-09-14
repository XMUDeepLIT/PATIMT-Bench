import json
import os
import numpy as np
import random
from PIL import Image
import copy

random.seed(42)

SRC_PATH = './test'
full_prompts = open('instruction_data_question/full_trans').readlines()
text_prompts = open('instruction_data_question/text_trans').readlines()

DEFAULT_IMAGE_TOKEN = "<image>"
BOX_PLACEHOLDER = "<box_pad>"
SEP_TOKEN="<|translation|>"


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
        elif box_format == "qwen" or "internvl" in box_format or 'deepseek' in box_format:  # qwen
            xmin, ymin, xmax, ymax = (
                int((1 - margin) * xmin / w * 1000),
                int((1 - margin) * ymin / h * 1000),
                int((1 - margin) * xmax / w * 1000),
                int((1 - margin) * ymax / h * 1000),
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
    elif box_format == "pad":
        return f"{BOX_PLACEHOLDER}"
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
    # data = [str(line) for line in data]
    json_str = json.dumps(data, indent='\t', ensure_ascii=False, default=default_dump)
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
                    elif data_args.box_format == "llava" or data_args.box_format == 'qwen25':
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


def process_full_translation(prompt: dict, label: dict, data_args=None):
    """conversation: {from: '', value: ''}"""
    prompt['value'] = random.choice(full_prompts).strip()
    prompt, label = format_response(
        prompt=prompt, label=label, text_only=False, data_args=data_args
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
    prompt: dict, label: dict, data_args=None
):
    """
    Replace question with box translation
    conversation: {from: '', value: ''}
    return:"""
    conv = []
    ### sample ###
    label["value"] = sorted(
        label["value"], key=lambda x: cal_area(x["box"]), reverse=True
    )
    max_ = min(len(label["value"]), 3)
    label["value"] = [random.choice(label["value"][:max_])]
    prompt['value'] = random.choice(text_prompts).strip()
    prompt, label = format_response(
        prompt=prompt,
        label=label,
        text_only=True,
        data_args=data_args,
    )
    return prompt, label



def Args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type", type=str, default="rec_and_trans")
    parser.add_argument("--box_format", type=str, default="qwen25")
    parser.add_argument("--translation_type", type=str, default="mix")
    parser.add_argument("--output_json", action="store_true")
    parser.add_argument("--random_question", action="store_false")
    parser.add_argument("--lang", type=str, default="EN-ZH")
    parser.add_argument("--scene", type=str, default="ads_book_posters")
    parser.add_argument("--task", type=str, default="text")
    return parser.parse_args()


if __name__ == "__main__":
    args = Args()
    LANG = args.lang
    SCENE = args.scene
    TASK = args.task
    GPT_LABEL = os.path.join(SRC_PATH, LANG, SCENE, "gpt_label_new.jsonl")
    OUTPUT_DIR = os.path.join(SRC_PATH, LANG, SCENE, TASK)
    gpt_labels = open(GPT_LABEL, "r").readlines()

    prompt_path = os.path.join(OUTPUT_DIR, f"prompt-{args.box_format}-{args.task_type}-json_{args.output_json}.jsonl")
    answer_path = os.path.join(OUTPUT_DIR, f"answer-{args.box_format}-{args.task_type}-json_{args.output_json}.jsonl")
    prompt_file = open(prompt_path, "w")
    answer_file = open(answer_path, "w")
    for idx, line in enumerate(gpt_labels):
        """
        {id, image, conversations: [
            {from: human, value: question1},
            {from: gpt, value: answer1},  --[{box, src_text, tgt_text, src_lang, tgt_lang}]
            {from: human, value: question2},
            {from: gpt, value: answer2}   --[{box, src_text=<None>, tgt_text=<None>, src_lang=<None>, tgt_lang=<None>}]
        ]}
        """
        data = json.loads(line)
        question_id = idx
        try:
            prompt, label = data["conversations"][0], data["conversations"][1]
            source = regulate_boxes(
                image=data["image"],
                label=label,
                box_format=args.box_format,
                ocr_key='value'
            )
            for val in data["conversations"][1]['value']:
                val['src_lang'] = val['src_lang'].replace('zh', 'Chinese').replace('en', "English")
                val['tgt_lang'] = val['tgt_lang'].replace('zh', 'Chinese').replace('en', "English")
            
            if TASK == 'text':
                prompt, label = process_text_translation(
                    prompt=prompt, label=label, data_args=args
                )
            elif TASK == 'full':
                prompt, label = process_full_translation(
                prompt=prompt, label=copy.deepcopy(label), data_args=args
            )
            question = prompt['value']
            answer = label['value']
        except Exception as e:
            print(e)
            import traceback

            print(traceback.format_exc())
            continue
        prompt_file.write(
            json.dumps(
                {
                    "question_id": question_id,
                    "image_id": data["id"],
                    "image": data["image"],
                    "text": question,
                    "answers": answer,
                },
                default=default_dump,
                ensure_ascii=False,
            )
            + "\n"
        )

        answer_file.write(
            json.dumps(
                {
                    "question_id": question_id,
                    "image_id": data["id"],
                    "image": data["image"],
                    "question": question,
                    "answers": answer,
                },
                default=default_dump,
                ensure_ascii=False,
            )
            + "\n"
        )

    prompt_file.close()
    answer_file.close()
    print(f"prompt file saved to {prompt_path.split('/')[-1]}")
    print(f"answer file saved to {answer_path.split('/')[-1]}")
    print(f"Total {len(gpt_labels)}")
    print("GG")

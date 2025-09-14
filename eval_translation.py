import os
import argparse
import json
import re

import re
import numpy as np
from PIL import Image
from comet import download_model, load_from_checkpoint

SEP_TOKEN = "<|translation|>"

def decode_bbox_from_caption(text, box_format="qwen"):
    DEFAULT_IMAGE_TOKEN = "<image>"

    import re

    def decode_qwen_box(text):
        pattern = r"(.*?)<\|box_start\|>\((\d+),\s*(\d+)\),\((\d+),\s*(\d+)\)<\|box_end\|>(.*)"
        matches = re.findall(pattern, text)
        references = []
        boxes = []
        for match in matches:
            ref = match[0].strip()
            x1, y1, x2, y2 = int(match[1]), int(match[2]), int(match[3]), int(match[4])
            references.append(ref)
            boxes.append([x1, y1, x2, y2])
        if references == []:
            references = [text]
        return references, boxes

    def decode_qwen2_box(text):
        pattern = r"(.*?)\((\d+),(\d+)\),\((\d+),(\d+)\)(.*)"
        matches = re.findall(pattern, text)
        references = []
        boxes = []
        for match in matches:
            ref = match[0].strip()

            x1, y1, x2, y2 = int(match[1]), int(match[2]), int(match[3]), int(match[4])
            references.append(ref)
            boxes.append([x1, y1, x2, y2])
        if references == []:
            references = [text]
        print()
        return references, boxes

    def decode_qwen25_box(input_text):
        pattern = r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"

        texts = []
        coordinates = []

        if input_text.startswith(DEFAULT_IMAGE_TOKEN):
            lines = [input_text]
        else:
            lines = input_text.split("\n")
        for line in lines:
            match = re.search(pattern, line)
            if match:
                coords = [float(match.group(i)) for i in range(1, 5)]
                text = re.sub(pattern, "", line).strip()
                texts.append(text)
                coordinates.append(coords)
            else:
                texts.append(line.strip())

        return texts, coordinates

    def decode_llava_box(input_text):
        pattern = r"\[(\d{1}\.\d{1,2}),\s*(\d{1}\.\d{1,2}),\s*(\d{1}\.\d{1,2}),\s*(\d{1}\.\d{1,2})\]"

        texts = []
        coordinates = []
        if input_text.startswith(DEFAULT_IMAGE_TOKEN):
            lines = [input_text]
        else:
            lines = input_text.split("\n")
        for line in lines:
            match = re.search(pattern, line)
            if match:
                coords = [float(match.group(i)) for i in range(1, 5)]
                text = re.sub(pattern, "", line).strip()
                texts.append(text)
                coordinates.append(coords)
            else:
                texts.append(line.strip())

        return texts, coordinates

    def decode_box_padding(text, box_pad="<|box_pad|>"):
        if box_pad not in text:
            return text, []
        entities = text.split(box_pad)
        boxes = [[box_pad]] * (len(entities) - 1)
        return entities, boxes

    def decode_internvl_box(input_text):
        pattern = r"(.*?)<box>\s*\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]\s*</box>"
        texts = []
        coordinates = []
        matches = re.findall(pattern, input_text)
        for match in matches:
            text = match[0].strip()
            x1, y1, x2, y2 = float(match[1]), float(match[2]), float(match[3]), float(match[4])
            texts.append(text)
            coordinates.append([x1,y1,x2,y2])
        if len(coordinates) == 0:
            texts = [input_text]
        return texts, coordinates

    def decode_deepseek_box(input_text):
        pattern = r"(.*?)<\|det\|>\s*\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]\s*<\|/det\|>"
        texts = []
        coordinates = []
        texts = []
        coordinates = []
        matches = re.findall(pattern, input_text)
        for match in matches:
            text = match[0].strip()
            x1, y1, x2, y2 = float(match[1]), float(match[2]), float(match[3]), float(match[4])
            texts.append(text)
            coordinates.append([x1,y1,x2,y2])
        if len(coordinates) == 0:
            texts = [input_text]
        return texts, coordinates

    if box_format == "qwen":
        return decode_qwen_box(text)
    elif box_format == "qwen2":
        return decode_qwen2_box(text)
    elif box_format == "llava":
        return decode_llava_box(text)
    elif box_format == "pad":
        return decode_box_padding(text)
    elif "internvl" in box_format:
        return decode_internvl_box(text)
    elif box_format == 'qwen25':
        return decode_qwen25_box(text)
    elif 'deepseek' in box_format:
        return decode_deepseek_box(text)
    else:
        raise NotImplementedError(f"unrecognized box_format: {box_format}")


def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

class TranslationAccuracyEvaluator:
    def __init__(self):
        self.comet_model = load_from_checkpoint('Unbabel/wmt22-comet-da/checkpoints/model.ckpt', reload_hparams=True)
       

    def parse_json(self, json_output):
        # Parsing out the markdown fencing
        lines = json_output.splitlines()
        for i, line in enumerate(lines):
            if line == "```json":
                json_output = "\n".join(
                    lines[i + 1 :]
                )  # Remove everything before "```json"
                json_output = json_output.split("```")[
                    0
                ]  # Remove everything after the closing "```"
                break  # Exit the loop once "```json" is found
        return json_output

    def json_to_plain_text(self, json_input, args):
        json_output = self.parse_json(json_input)
        json_output = json.loads(json_output)
        if not isinstance(json_output, list):
            json_output = [json_output]
        key = (
            "translation" if "translation" in json_output[0].keys() else "text_content"
        )
        if "iou" in args.metrics.lower():  # return box
            text_output = "\n".join(
                [f'{item["text_content"]} {SEP_TOKEN} {item[key]} {item["bbox_2d"]}' for item in json_output]
            )
        else:
            text_output = "\n".join([f"{item['text_content']} {SEP_TOKEN} {item[key]}" for item in json_output])

        return text_output

    def split_source(self, text):
        text_list = text.split("\n")
        source = []
        trans = []
        for text in text_list:
            source.append(text.split(SEP_TOKEN)[0])
            trans.append(text.split(SEP_TOKEN)[-1])
        translation_result = "\n".join(trans)
        return source, translation_result
    
    def split_rec(self, text):
        text_list = text.split("\n")
        ret = []
        for text in text_list:
            ret.append(text.split(SEP_TOKEN)[-1])
        translation_result = "\n".join(ret)
        return translation_result

    def resize_boxes(self, bboxes, height, width, box_format='llava'):
        if box_format != 'qwen25':
            return bboxes
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=2 * 14,
            min_pixels=256 * 28 * 28,
            max_pixels=2048 * 28 * 28,
        )
        ret = []
        for bbox in bboxes:
            ret.append(
                [
                    int(bbox[0] / width * resized_width),
                    int(bbox[1] / height * resized_height),
                    int(bbox[2] / width * resized_width),
                    int(bbox[3] / height * resized_height),
                ]
            )
        return ret

    def eval_pred_list(self, pred_list, trg_lang, args):
        logs = []
        for idx, entry in enumerate(pred_list):
            if args.json is True:
                try:
                    entry["pred_response"] = self.json_to_plain_text(
                        entry["pred_response"], args
                    )
                except:
                    print("parse json output error, run string replacement...")
                    entry["pred_response"] = (
                        entry["pred_response"]
                        .replace("```json", "")
                        .replace("```", "")
                        .replace('"bbox_2d"', "")
                        .replace('"translation"', "")
                        .replace("\n", "")
                        .replace("\t", "")
                        # .replace("[", "")
                        # .replace("]", "")
                        .replace("{", "")
                        .replace("}", "")
                    )
                entry["gt_response"] = self.json_to_plain_text(
                    entry["gt_response"], args
                )
            sources, gt_response = self.split_source(entry["gt_response"])
            gt_response = (
                self.split_rec(gt_response)
                .lower()
                .strip()
                .replace("assistant", "")
                .replace("<|im_start|>", "")
            )
            pred_response = (
                self.split_rec(entry["pred_response"])
                .lower()
                .strip()
                .replace("assistant", "")
                .replace("<|im_start|>", "")
            )
            if "iou" in args.metrics.lower():
                gt_texts, gt_boxes = decode_bbox_from_caption(
                    gt_response, box_format=args.box_format
                )
                print(f"gt_texts: {gt_texts}\ngt_boxes: {gt_boxes}")
                pred_texts, pred_boxes = decode_bbox_from_caption(
                    pred_response, box_format=args.box_format
                )
                print(f"pred_texts: {pred_texts}\npred_boxes: {pred_boxes}")
                pil_img = Image.open(entry["image"]).convert("RGB")
                height, width = pil_img.height, pil_img.width
                gt_boxes = self.resize_boxes(
                    bboxes=gt_boxes, height=height, width=width, box_format=args.box_format
                )
                ground_truths = [
                    (
                        {
                            "translation": gt_text,
                            "source": source,
                            "bbox_2d": gt_box,
                        }
                    )
                    for (gt_text, source, gt_box) in zip(gt_texts, sources, gt_boxes)
                ]
                predictions = [
                    (
                        {
                            "translation": pred_text,
                            "bbox_2d": pred_box,
                        }
                    )
                    for (pred_text, pred_box) in zip(pred_texts, pred_boxes)
                ]
            else:  # bleu
                ground_truths = [
                        {
                            "translation": gt_text,
                            "source": source,
                        } for (source, gt_text) in zip(sources, [gt_response])]
                predictions = [
                        {
                            "translation": pred_text
                        } for pred_text in [pred_response]]
            try:
                matched_results, iou, corpus_bleu_score, comet_score = self.eval_single(
                    ground_truths=ground_truths,
                    predictions=predictions,
                    trg_lang=trg_lang,
                    args=args,
                )
                logs.append(
                    {
                        "bleu": corpus_bleu_score,
                        "comet": comet_score,
                        "iou": iou,
                        "source": [item['source'] for item in matched_results],
                        "gt": [item["reference"] for item in matched_results],
                        "pred": [item["translation"] for item in matched_results],
                    }
                )
            except Exception as e:
                print(e)
                import traceback
                print(traceback.format_exc())
                logs.append(
                    {"bleu": 0, "comet": 0, "iou": 0, "gt": gt_response, "pred": pred_response}
                )

        avg_bleu, avg_comet, avg_iou = (
            round(np.mean([item["bleu"] for item in logs]), 3),
            round(np.mean([item["comet"] for item in logs]), 3),  
            round(np.mean([item["iou"] for item in logs]), 3),
        )
        return avg_bleu, avg_comet, avg_iou, logs

    def eval_single(self, ground_truths, predictions, trg_lang="zh", args=None):
        matched_results = []
        from sacrebleu.metrics import BLEU

        bleu = BLEU(trg_lang=trg_lang, effective_order=True)
         
        if "iou" in args.metrics.lower():
            for gt in ground_truths:
                best_iou = 0
                for pred in predictions:
                    try:
                        iou = self.compute_iou(pred["bbox_2d"], gt["bbox_2d"])
                    except Exception as e:
                        print(e)
                        print("prediction:")
                        print(pred)
                        print("ground truth:")
                        print(gt)
                    if iou >= best_iou:
                        best_iou = iou
                        matched_pred = pred
                if best_iou >= 0.5:
                    bleu_score = round(
                        bleu.sentence_score(
                            matched_pred["translation"], [gt["translation"]]
                        ).score,
                        3,
                    )
                    matched_results.append(
                        {
                            "bbox_2d": matched_pred["bbox_2d"],
                            "translation": matched_pred["translation"],
                            "reference": gt["translation"],
                            "iou": best_iou,
                            "sentence_bleu": bleu_score,
                            "source": gt['source'],
                        }
                    )
                else:
                    matched_results.append(
                        {
                            "bbox_2d": matched_pred["bbox_2d"],
                            "translation": matched_pred["translation"],
                            "reference": gt["translation"],
                            "iou": best_iou,
                            "sentence_bleu": 0,
                            "source": gt['source'],
                        }
                    )
        else:
            for translation, reference in zip(predictions, ground_truths):
                bleu_score = round(
                    bleu.sentence_score(translation['translation'], [reference['translation']]).score, 3
                )
                matched_results.append(
                    {
                        "translation": translation['translation'],
                        "reference": reference['translation'],
                        "sentence_bleu": bleu_score,
                        "source": reference['source'],
                    }
                )
        ## Calculate the corpus BLEU score
        all_predictions = [x["translation"] for x in matched_results]
        all_references = [x["reference"] for x in matched_results]
        all_sources = [x["source"] for x in matched_results]
        corpus_bleu_score = round(
            bleu.corpus_score(all_predictions, [all_references]).score, 3
        )
        print(f"Corpus BLEU score: {corpus_bleu_score}")
        comet_score = np.mean(self.comet_model.predict([{'src': src, 'mt': mt, 'ref': ref} for (src, mt, ref) in zip(all_sources, all_predictions, all_references)])['scores'])
        avg_iou = (
            round(np.mean([result["iou"] for result in matched_results]), 3)
            if "iou" in args.metrics.lower()
            else 0
        )
        return matched_results, avg_iou, corpus_bleu_score, comet_score

    def compute_iou(self, box1, box2):
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area if union_area != 0 else 0
        return iou


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str)
    parser.add_argument("--result-file", type=str)
    parser.add_argument("--result-dir", type=str)
    parser.add_argument("--output_file", type=str, required=False, default=None)
    parser.add_argument("--metrics", type=str, default="bleu, iou")
    parser.add_argument("--box_format", type=str, default="llava")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def prompt_processor(prompt):
    if prompt.startswith("OCR tokens: "):
        pattern = r"Question: (.*?) Short answer:"
        match = re.search(pattern, prompt, re.DOTALL)
        question = match.group(1)
    elif "Reference OCR token: " in prompt and len(prompt.split("\n")) == 3:
        if prompt.startswith("Reference OCR token:"):
            question = prompt.split("\n")[1]
        else:
            question = prompt.split("\n")[0]
    elif len(prompt.split("\n")) == 2:
        question = prompt.split("\n")[0]
    else:
        question = prompt

    return question.lower()


def eval_file(annotation_file, result_file):
    experiment_name = os.path.splitext(os.path.basename(result_file))[0]
    print(experiment_name)
    annotations = [
        json.loads(line) for line in open(annotation_file).readlines()
    ]
    results = [json.loads(line) for line in open(result_file)]
    annotations = {
        (annotation["question_id"]): annotation for annotation in annotations
    }

    pred_list = []
    for result in results:
        annotation = annotations[(result["question_id"])]
        if isinstance(annotation["answers"], dict):
            annotation["answers"] = annotation["answers"]["value"]
        pred_list.append(
            {
                "image": annotation["image"],
                "pred_response": result["pred_response"],
                "gt_response": annotation["answers"],
            }
        )
    trg_lang = "en" if "ZH-EN" in annotation_file else "zh"

    evaluator = TranslationAccuracyEvaluator()
    avg_bleu, avg_comet, avg_iou, logs = evaluator.eval_pred_list(pred_list, trg_lang, args)
    print("Samples: {}\tBleu: {}\tComet: {}\tIoU: {}\n".format(len(pred_list), avg_bleu, avg_comet, avg_iou))
    return len(pred_list), avg_bleu, avg_comet, avg_iou, logs


if __name__ == "__main__":
    args = get_args()
    if args.result_dir is not None:
        for result_file in sorted(os.listdir(args.result_dir)):
            if not result_file.endswith(".jsonl"):
                print(f"Skipping {result_file}")
                continue
        args.result_file = os.path.join(args.result_dir, result_file)

    total, avg_bleu, avg_comet, avg_iou, logs = eval_file(args.annotation_file, args.result_file)
    if args.output_file is not None:
        if not (
            (args.output_file.endswith("json")) or (args.output_file.endswith("jsonl"))
        ):
            os.makedirs(args.output_file, exist_ok=True)
            args.output_file = os.path.join(
                args.output_file, args.result_file.split("/")[-1]
            )

        with open(args.output_file, "w") as fout:
            for item in logs:
                fout.write(
                    json.dumps(
                        item,
                        default=default_dump,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            fout.write(
                json.dumps(
                    {
                        "annotation-file": args.annotation_file,
                        "result-file": args.result_file,
                    },
                    default=default_dump,
                    ensure_ascii=False,
                )
                + "\n"
            )
            fout.write(
                json.dumps(
                    {
                        "metrics": args.metrics,
                        "total samples": total,
                        "bleu": avg_bleu,
                        "comet": avg_comet,
                        "IoU": avg_iou,
                    },
                    default=default_dump,
                    ensure_ascii=False,
                )
                + "\n"
            )
    print("GG")

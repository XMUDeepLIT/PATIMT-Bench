import os
import json
import subprocess
from typing import List, Dict, Tuple
import os
from PIL import Image, ImageDraw, ImageFont
import json
import cv2
import numpy as np
import torch
from tqdm import tqdm
import traceback
from magic_pdf.data.data_reader_writer import FileBasedDataWriter
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.data.read_api import read_local_images
import easyocr
from transformers import CLIPProcessor, CLIPModel


class MinerUHandler:
    def __init__(self, output_tmp_img_dir, output_json_dir):
        os.makedirs(output_tmp_img_dir, exist_ok=True)
        os.makedirs(output_json_dir, exist_ok=True)
        self.image_writer = FileBasedDataWriter(output_tmp_img_dir)
        self.md_writer = FileBasedDataWriter(output_json_dir)

    def run(self, img_path, save_name):
        try:
            img_name = img_path.split(".")[0].split("/")[-1]
            ds = read_local_images(img_path)[0]

            pipe_result = ds.apply(doc_analyze, ocr=True).pipe_ocr_mode(
                self.image_writer
            )
            # pipe_result = ds.apply(doc_analyze, ocr=False).pipe_txt_mode(self.image_writer)
            pipe_result.draw_layout(
                os.path.join(self.md_writer._parent_dir, f"{save_name}_layout.pdf")
            )
            middle_json = json.loads(pipe_result.get_middle_json())
            return middle_json
        except Exception as e:
            raise RuntimeError(f"MinerU Error!: {e}")

    def format_result(self, para, page_size, img_size):
        w, h = page_size
        width, height = img_size
        [xmin, ymin, xmax, ymax] = para["bbox"]
        xmin = xmin / w
        ymin = ymin / h
        xmax = xmax / w
        ymax = ymax / h
        if "lines" in para.keys():
            spans = [item["spans"] for item in para["lines"]]
            text = []
            temp_text = []
            for item in spans:
                for i in item:
                    if i["type"] == "inline_equation":
                        import re

                        pattern = r"([a-zA-Z0-9])\\\\?%"
                        i["content"] = re.sub(pattern, r"\1%", i["content"])
                        # cnt_equa +=1
                        temp_text.append(i["content"])

                    else:
                        if len(i["content"]) >= 2 and i["type"] == "text":
                            temp_text.append(i["content"])
                            text.append(i["content"])

            text = "".join(text)
            temp_text = "".join(temp_text)
            return [
                {
                    "box": [
                        int(xmin * width),
                        int(ymin * height),
                        int(xmax * width),
                        int(ymax * height),
                    ],
                    "text": text,
                }
            ]



class SpatialMerger:
    def easyocr_spatial_merge(self, ocr_results, x_ths=1, y_ths=0.5, mode="ltr"):
        # create basic attributes
        box_group = []
        for box in ocr_results:
            all_x = [int(box["box"][0]), int(box["box"][2])]
            all_y = [int(box["box"][1]), int(box["box"][3])]
            min_x = min(all_x)
            max_x = max(all_x)
            min_y = min(all_y)
            max_y = max(all_y)
            height = max_y - min_y
            box_group.append(
                [
                    box["text"],
                    min_x,
                    max_x,
                    min_y,
                    max_y,
                    height,
                    0.5 * (min_y + max_y),
                    0,
                ]
            )  # last element indicates group
        # cluster boxes into paragraph
        current_group = 1
        while len([box for box in box_group if box[7] == 0]) > 0:
            box_group0 = [box for box in box_group if box[7] == 0]  # group0 = non-group
            # new group
            if len([box for box in box_group if box[7] == current_group]) == 0:
                box_group0[0][7] = current_group  # assign first box to form new group
            # try to add group
            else:
                current_box_group = [
                    box for box in box_group if box[7] == current_group
                ]
                mean_height = np.mean([box[5] for box in current_box_group])
                min_gx = (
                    min([box[1] for box in current_box_group]) - x_ths * mean_height
                )
                max_gx = (
                    max([box[2] for box in current_box_group]) + x_ths * mean_height
                )
                min_gy = (
                    min([box[3] for box in current_box_group]) - y_ths * mean_height
                )
                max_gy = (
                    max([box[4] for box in current_box_group]) + y_ths * mean_height
                )
                add_box = False
                for box in box_group0:
                    same_horizontal_level = (min_gx <= box[1] <= max_gx) or (
                        min_gx <= box[2] <= max_gx
                    )
                    same_vertical_level = (min_gy <= box[3] <= max_gy) or (
                        min_gy <= box[4] <= max_gy
                    )
                    if same_horizontal_level and same_vertical_level:
                        box[7] = current_group
                        add_box = True
                        break
                # cannot add more box, go to next group
                if add_box == False:
                    current_group += 1
        # arrage order in paragraph
        result = []
        for i in set(box[7] for box in box_group):
            current_box_group = [box for box in box_group if box[7] == i]
            mean_height = np.mean([box[5] for box in current_box_group])
            min_gx = min([box[1] for box in current_box_group])
            max_gx = max([box[2] for box in current_box_group])
            min_gy = min([box[3] for box in current_box_group])
            max_gy = max([box[4] for box in current_box_group])

            text = ""
            while len(current_box_group) > 0:
                highest = min([box[6] for box in current_box_group])
                candidates = [
                    box
                    for box in current_box_group
                    if (box[6] < highest + 0.4 * mean_height)
                ]
                # get the far left
                if mode == "ltr":
                    most_left = min([box[1] for box in candidates])
                    for box in candidates:
                        if box[1] == most_left:
                            best_box = box
                elif mode == "rtl":
                    most_right = max([box[2] for box in candidates])
                    for box in candidates:
                        if box[2] == most_right:
                            best_box = box

                text += " " + best_box[0]
                current_box_group.remove(best_box)
            result.append(
                {"box": [min_gx, min_gy, max_gx, max_gy], "text": text.strip()}
            )

        return result


class OCRProcessor:

    def __init__(self, output_json_dir, output_tmp_img_dir):
        self.mineru_handler = MinerUHandler(output_tmp_img_dir=output_tmp_img_dir, output_json_dir=output_json_dir)
        self.spatial_merger = SpatialMerger()
        self.ignore_list = [
            'http://',
            'https://'
            'www.',
            '.com',
            '.cn'
        ]

    def process(
        self, img_path: str, img_type: str, img_size, ocr_results: List[Dict], save_name
    ) -> List[Dict]:
        ocr_results = [ret for ret in ocr_results if ret["text"] != ""]
        if img_type == "pdf":
            exist_boxes = []
            ocr_blocks = []
            try:
                mineru_results = self.mineru_handler.run(img_path, save_name)
                final_results = []
                page_size = mineru_results['pdf_info'][0]['page_size']
                for para in mineru_results['pdf_info'][0]['para_blocks']:
                    # page_size = para['page_size']
                    if para["type"] not in ["text", "title"]:
                        normed_bbox = self._norm_bbox(para["bbox"], page_size, img_size)
                        region_ocr = self._extract_region_ocr(ocr_results, normed_bbox, img_size, page_size)
                        ocr_blocks.extend(region_ocr)
                    else:
                        normed_bbox = self._norm_bbox(para["bbox"], page_size, img_size)
                        exist_boxes.append(normed_bbox)
                        final_results.extend(self.mineru_handler.format_result(para, page_size=page_size, img_size=img_size))
                for para in mineru_results['pdf_info'][0]['discarded_blocks']: # type=discarded
                    # page_size = para['page_size']
                    if not any(ignore in ''.join([''.join([span['content'] for span in line['spans']]) for line in para['lines']]) for ignore in self.ignore_list):
                        normed_bbox = self._norm_bbox(para["bbox"], page_size, img_size)
                        exist_boxes.append(normed_bbox)
                        region_ocr = self._extract_region_ocr(ocr_results, normed_bbox, img_size, page_size)
                        ocr_blocks.extend(region_ocr)
                # process remain_results
                remain_ocr = [d for d in ocr_results if d not in ocr_blocks]
                remain_ocr = self.find_remain_ocr(remain_ocr, exist_boxes)
                ocr_blocks.extend(remain_ocr)
                merged = self.spatial_merger.easyocr_spatial_merge(ocr_blocks)
                final_results.extend(merged)
                return final_results
            
            except Exception as e:
                print("MinerU error!")
                print(traceback.format_exc())
                print(e)
                return self.spatial_merger.easyocr_spatial_merge(ocr_results)
        else:
            return self.spatial_merger.easyocr_spatial_merge(ocr_results)

    def find_remain_ocr(self, ocr_result, exist_bboxes):
        result = []
        for element in ocr_result:
            box1 = element["box"]
            no_overlap = True
            for box2 in exist_bboxes:
                if self.is_box_inside_another(box1, box2, ratio=0.1):
                    no_overlap = False
                    break
            if no_overlap:
                result.append(element)
        return result

    def _extract_region_ocr(self, ocr_data: List[Dict], bbox: List[int], img_size, page_size) -> List[Dict]:
        region_ocr = []
        x1_min, y1_min, x2_max, y2_max = bbox
        for item in ocr_data:
            x1, y1, x2, y2 = item['box']
            if self.is_box_inside_another([x1, y1, x2, y2], [x1_min, y1_min, x2_max, y2_max]):
                region_ocr.append(item)
        return region_ocr

    def is_box_inside_another(self, box1, box2, ratio = 0.8):
        intersection_x_min = max(box1[0], box2[0])
        intersection_y_min = max(box1[1], box2[1])
        intersection_x_max = min(box1[2], box2[2])
        intersection_y_max = min(box1[3], box2[3])

        intersection_width = max(0, intersection_x_max - intersection_x_min)
        intersection_height = max(0, intersection_y_max - intersection_y_min)
        intersection_area = intersection_width * intersection_height

        box1_width = box1[2] - box1[0]
        box1_height = box1[3] - box1[1]
        box1_area = box1_width * box1_height

        box2_width = box2[2] - box2[0]
        box2_height = box2[3] - box2[1]
        box2_area = box2_width * box2_height

        if box1_area == 0:
            return False

        overlap_ratio = max(intersection_area / box1_area, intersection_area / box2_area)

        return overlap_ratio > ratio

    def _norm_bbox(self, box, page_size, img_size):
        w, h = page_size
        width, height = img_size
        [xmin, ymin, xmax, ymax] = box
        xmin = int(xmin / w * width)
        ymin = int(ymin / h * height)
        xmax = int(xmax / w * width)
        ymax = int(ymax / h * height)
        return [xmin, ymin, xmax, ymax]



def parse_json(json_output):
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


def decode_qwen25_box(input_text):
    import re

    pattern = r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
    texts, coordinates = [], []
    if input_text.startswith("<image>"):
        lines = [input_text]
    else:
        lines = input_text.split("\n")
    for line in lines:
        match = re.search(pattern, line)
        if match:
            coords = [int(match.group(i)) for i in range(1, 5)]
            text = re.sub(pattern, "", line).strip()
            texts.append(text)
            coordinates.append(coords)
        else:
            texts.append(line.strip())
    return texts, coordinates


def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def gen_OCR(reader, img_path, output_path):
    ocr_ret = reader.readtext(img_path, paragraph=False)
    # print("ocr ret: ", ocr_ret)
    merged_list = [
        {'box':
            [val[0][0][0], val[0][0][1], val[0][2][0], val[0][2][1]],
           'text': val[1],
        }
        for val in ocr_ret
    ]
    with open(output_path, 'w') as fout:
        fout.write(json.dumps({'ocr': merged_list}, default=default_dump, ensure_ascii=False))
        fout.flush()
        fout.close()
    return merged_list

merge_dict = {
    'ads': 'ocr',
    'book': 'ocr',
    'posters': 'ocr',
    'natural': 'ocr',
    'street': 'ocr',
    'hand-written': 'ocr',
    'informatics': 'pdf',
    'document': 'pdf',
    'chart': 'ocr',
    'table': 'ocr',
}

def classify_image(image_path, model, processor, text_features, categories_for_prompts):
    """
    Classify an image into one of the predefined categories using CLIP
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Category name or "Other" if similarity is below threshold
    """
    # Load and preprocess image
    image = Image.open(image_path)
    image_inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarity scores
        similarity = (image_features @ text_features.T).squeeze()
        max_sim = similarity.max().item()
        idx = similarity.argmax().item()
        return categories_for_prompts[idx]



def Args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--n_gpus", type=int, default=8)
    parser.add_argument("--label_file", type=str, required=True)
    
    return parser.parse_args()



if __name__ == "__main__":
    args = Args()
    local_rank = args.local_rank
    n_gpus = args.n_gpus
    categories = {
        "ads": ["advertisement"],
        "book": ["book cover", "magazine cover", "comic book cover"],
        "posters": ["movie poster", "podcast poster", "TV show poster", "event poster", "poster", "concert poster", "conference poster", "travel poster", "art poster"],
        "natural": ["natural scene", "landscape", "nature background", "wildlife scene", "Trail sign", "Park map", "Info board", "Gate sign", "Stone plaque", "Wood post","Kiosk sign", "Exhibit panel"],
        "street": ["street view", "urban scene", "city street", "suburban neighborhood", "rural road", "traffic scene", "billboard", "shop front"],
        "hand-written": ["hand-written", "handwriting letter"],
        "infographic": ["infographics", "diagram", "mind map", "statistical graph"],
        "document": ["document", "contract"],
        "chart": ["chart", "bar chart", "pie chart", "scatter plot", "line chart", "Histogram", "area chart", "bubble chart"],
        "table": ["table", "spreadsheet", "matrix", "grid"]
    }

    templates = [
        "a photo of a {}.",
        "a blurry photo of a {}.",
        "a black and white photo of a {}.",
        "a low contrast photo of a {}.",
        "a high contrast photo of a {}.",
        "a bad photo of a {}.",
        "a good photo of a {}.",
        "a photo of a small {}.",
        "a photo of a big {}."
    ]

    # Initialize CLIP model and processor
    model_name = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    prompts = []
    categories_for_prompts = []
    for category, labels in categories.items():
        for label in labels:
            for template in templates:
                prompts.append(template.format(label))
                categories_for_prompts.append(category)

    # Precompute text features
    with torch.no_grad():
        text_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    LABEL_FILE = args.label_file # containing img_path and ocr results
    OUTPUT_FILE = f"refined_ocr_{local_rank}.jsonl" # output file that contains processed ocr annotation and gpt-labeled translation
    VIS_OUTPUT_PATH = 'mineru_tmp' # tmp path for mineru outputs
    reader = easyocr.Reader(
        ["ch_sim", "en"],
        gpu=True,
    )

    data = []
    for idx, line in enumerate(open(LABEL_FILE, 'r').readlines()):
        try:
            d = json.loads(line)
        except Exception as e:
            print(idx, ": ", e)
            print('')
        data.append(d)

    total = len(data)
    data = data[local_rank::n_gpus]
    print(f"local rank: {local_rank}, total samples: {total},  process samples: {len(data)}")
    merger = OCRProcessor(output_json_dir=os.path.join(VIS_OUTPUT_PATH, 'output_data'), output_tmp_img_dir=os.path.join(VIS_OUTPUT_PATH, 'tmp_imgs'))
    exist_len = len(open(OUTPUT_FILE, 'r').readlines()) if os.path.exists(OUTPUT_FILE) else 0
    minerU_queue = []
    batch_size = 50
    skip = 0
    action = 'w' if exist_len == 0 else 'a'
    with open(OUTPUT_FILE, action) as fout:
        for idx, (line) in enumerate(tqdm(data)):
            if idx < exist_len:
                print(f'skip {idx}')
                continue
            else:
                print(f'process {idx}')
            img_path = line['image']
            img_name = img_path.split("/")[-1]
            pil_img = Image.open(img_path).convert("RGB")
            ocr_info = line['ocr']
            pred_cls = classify_image(model=model, processor=processor, image_path=img_path, text_features=text_features, categories_for_prompts=categories_for_prompts)
            merged = merger.process(img_path=img_path, img_type=pred_cls, img_size=pil_img.size, ocr_results=ocr_info)
            ret = []

            print(f"raw: {len(ocr_info)} <=> refine: {len(merged)}")
            if len(merged) == 0:
                print(f"merge return no result, use original result")
                merged = ocr_info
            line['merge_ocr'] = merged
            fout.write(json.dumps(line, default=default_dump, ensure_ascii=False) + '\n')
            fout.flush()
        fout.close()
print('total: ', len(data))
print('write: ', len(open(OUTPUT_FILE, 'r').readlines()))
print("skip: ", skip)
print("GG")

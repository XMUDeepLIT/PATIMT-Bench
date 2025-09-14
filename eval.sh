PYTHON=''

GPUS=0
export CUDA_VISIBLE_DEVICES=$GPUS
export LOCAL_RANK=0
N_GPUS=$(echo "$GPUS" | awk -F, '{print NF}')

MODEL_NAME=''

SCENE='chart'
LANG='EN-ZH'
TASK='trans'
TRANSLATION_TYPE='text'
BOX_FORMAT=qwen # qwen2/qwen25/llava/internvl/deepseek
METRICS=bleu # or blue,iou for full image translation with grounding
ANN_FILE='your/model/output/answer.jsonl'


${PYTHON} .eval_translation.py \
--annotation-file ${ANN_FILE} \
--result-file ./output/${TASK}_${TRANSLATION_TYPE}-${BOX_FORMAT}-${SCENE}.jsonl \
--output_file ./eval_results \
--metrics ${METRICS} > eval.log 2>&1 &
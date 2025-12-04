#!/bin/sh
export CUDA_VISIBLE_DEVICES=0

cd Steering_Vectors_SAE_lens
export HF_TOKEN=""
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"



python3 create_data.py


#python3 list_saes.py

MODEL_ID="google/gemma-2-9b" # "google/gemma-2-2b"
SAE_RELEASE="gemma-scope-9b-pt-res-canonical" # "gemma-scope-2b-pt-res-canonical"
SAE_WIDTH="16k" # "32k"



DIM="bad"
python3 collect_sae_activations.py \
  --model_name $MODEL_ID \
  --sae_release $SAE_RELEASE \
  --sae_width $SAE_WIDTH \
  --batch_size 2 \
  --dataset_path "data/good_bad_data.json" \
  --dim $DIM \
  --output_dir activations



DIM="good"
python3 collect_sae_activations.py \
  --model_name $MODEL_ID \
  --sae_release $SAE_RELEASE \
  --sae_width $SAE_WIDTH \
  --batch_size 2 \
  --dataset_path "data/good_bad_data.json" \
  --dim $DIM \
  --output_dir activations




python3 create_steer_vector.py \
  --model_name $MODEL_ID \
  --sae_release $SAE_RELEASE \
  --sae_width $SAE_WIDTH \
  --dataset_path "activations" \
  --dims "good" "bad" \
  --output_dir vectors




### you can change the code take prompts.txt and change the code to read the data and save output to txt
DIM="good"

## base model (alpha=0) no sae
python3 steering_inference.py \
  --model_name $MODEL_ID \
  --sae_release $SAE_RELEASE \
  --sae_width $SAE_WIDTH \
  --dataset_path "vectors" \
  --dim $DIM \
  --layer 14 \
  --alpha 0.0 \
  --max_new_tokens 20 \
  --prompt "Generate a simple german sentence with poitive sentiment about baseball. \nExample about Auto: Ich liebe das Auto. \nSentence about base basll:" #\
#   --use_sae \

## steer residual stream
python3 steering_inference.py \
  --model_name $MODEL_ID \
  --sae_release $SAE_RELEASE \
  --sae_width $SAE_WIDTH \
  --dataset_path "vectors" \
  --dim $DIM \
  --layer 14 \
  --alpha 100.0 \
  --max_new_tokens 20 \
  --prompt "Generate a simple german sentence with poitive sentiment about baseball. \nExample about Auto: Ich liebe das Auto. \nSentence about base basll:" #\
#   --use_sae \



## steer SAE sparse space
python3 steering_inference.py \
  --model_name $MODEL_ID \
  --sae_release $SAE_RELEASE \
  --sae_width $SAE_WIDTH \
  --dataset_path "vectors" \
  --dim $DIM \
  --layer 14 \
  --alpha 100.0 \
  --max_new_tokens 20 \
  --prompt "Generate a simple german sentence with poitive sentiment about baseball. \nExample about Auto: Ich liebe das Auto. \nSentence about base basll:" \
  --use_sae


### it causes repeat?

DIM="bad"

## steer residual stream
python3 steering_inference.py \
  --model_name $MODEL_ID \
  --sae_release $SAE_RELEASE \
  --sae_width $SAE_WIDTH \
  --dataset_path "vectors" \
  --dim $DIM \
  --layer 14 \
  --alpha 100.0 \
  --max_new_tokens 20 \
  --prompt "Generate a simple german sentence with poitive sentiment about baseball. \nExample about Auto: Ich liebe das Auto. \nSentence about base basll:" #\
#   --use_sae \



## steer SAE sparse space
python3 steering_inference.py \
  --model_name $MODEL_ID \
  --sae_release $SAE_RELEASE \
  --sae_width $SAE_WIDTH \
  --dataset_path "vectors" \
  --dim $DIM \
  --layer 14 \
  --alpha 100.0 \
  --max_new_tokens 20 \
  --prompt "Generate a simple german sentence with poitive sentiment about baseball. \nExample about Auto: Ich liebe das Auto. \nSentence about base basll:" \
  --use_sae
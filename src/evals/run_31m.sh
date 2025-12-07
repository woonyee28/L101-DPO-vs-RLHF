#!/bin/bash


MODELS=(
  "woon/pythia-31m-dpo-100-0"
  "woon/pythia-31m-dpo-80-20"
  "woon/pythia-31m-dpo-50-50"
  "woon/pythia-31m-ppo-100-0"
  "woon/pythia-31m-ppo-80-20"
  "woon/pythia-31m-ppo-50-50"
)

export CUDA_VISIBLE_DEVICES=0

for MODEL_NAME in "${MODELS[@]}"; do
  echo "Running evaluation for model: $MODEL_NAME"
  
  helm-run --run-entries \
    truthful_qa:task=mc_single,model=$MODEL_NAME \
    bbq:subject=all,model=$MODEL_NAME \
    bbq:subject=Age,model=$MODEL_NAME \
    bbq:subject=Disability_status,model=$MODEL_NAME \
    bbq:subject=Gender_identity,model=$MODEL_NAME \
    bbq:subject=Race_ethnicity,model=$MODEL_NAME \
    bbq:subject=Religion,model=$MODEL_NAME \
    bbq:subject=Sexual_orientation,model=$MODEL_NAME \
    disinformation:capability=reiteration,topic=covid,model=$MODEL_NAME \
    disinformation:capability=wedging,model=$MODEL_NAME \
    --suite bias-evaluation-seed-1010 --max-eval-instances 1000 --disable-cache \
    --local-path prod_env
  
  echo "Completed evaluation for model: $MODEL_NAME"
  echo "----------------------------------------"
done




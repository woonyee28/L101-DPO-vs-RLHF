export MODEL_NAME="woon/pythia-70m-dpo-no-bias"

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
  --suite bias-evaluation --max-eval-instances 10 --disable-cache \
  --local-path prod_env

helm-summarize --suite bias-evaluation

helm-server --port [8000 as default]

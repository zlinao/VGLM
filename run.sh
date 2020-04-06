## GPT2:
#qa
python ./train.py --gradient_accumulation_steps=8 --max_history=1 --train_batch_size=2 --valid_batch_size=2 --n_epochs 3 --task qa --dataset_path data/CoQA/data.json
python ./evaluate.py --task qa --no_sample --max_history=1 --dataset_path data/CoQA/data.json --model_checkpoint runs/$model_checkpoint

#mt
python ./train.py --gradient_accumulation_steps=4 --max_history=2 --train_batch_size=8 --valid_batch_size=8 --n_epochs 8 --task mt --dataset_path data/NMT/data_en_ge.json
python ./evaluate.py --task mt --no_sample --max_history=2 --model_checkpoint runs/$model_checkpoint

#nlg
python ./train.py --gradient_accumulation_steps=4 --max_history=2 --train_batch_size=8 --valid_batch_size=8 --n_epochs 10 --task nlg --dataset_path data/NLG/data.json --lr 0.00005
python ./evaluate.py --task nlg --no_sample --model_checkpoint runs/$model_checkpoint

#persona
python ./train.py --gradient_accumulation_steps=2 --max_history=2 --train_batch_size=8 --valid_batch_size=8 --n_epochs 3 --task dialogue --dataset_path data/persona/data.json
python ./evaluate.py --task dialogue --no_sample --max_history=2 --model_checkpoint runs/$model_checkpoint

#summarization
python ./train.py --gradient_accumulation_steps=8 --max_history=2 --train_batch_size=4 --valid_batch_size=4 --n_epochs 10 --task summarization --dataset_path data/CNNDAILY/data.json --lr 0.00008
python ./evaluate.py --task summarization --no_sample --model_checkpoint runs/$model_checkpoint


## VLM finetune Adapter and Task embedding:
#qa
python ./train.py --gradient_accumulation_steps=8 --max_history=1 --train_batch_size=2 --valid_batch_size=2 --n_epochs 5 --task qa --dataset_path data/CoQA/data.json --adapter_bottleneck 300 --lr 0.0005 
python ./evaluate.py --task qa --no_sample --max_history=1 --dataset_path data/CoQA/data.json --adapter_bottleneck 300 --model_checkpoint runs/$model_checkpoint

#mt
# without distillation
python ./train.py --gradient_accumulation_steps=4 --max_history=2 --train_batch_size=8 --valid_batch_size=8 --n_epochs 8 --task mt --dataset_path data/NMT/data_en_ge.json --adapter_bottleneck 300 --lr 0.0005
# use sentence level knowledge distillation:
python ./sentence_distiller.py --task mt --max_history=2 --model_checkpoint runs/$fully_finetuned_gpt2_checkpoint --no_sample
python ./train.py --gradient_accumulation_steps=4 --max_history=2 --train_batch_size=8 --valid_batch_size=8 --n_epochs 8 --task mt --dataset_path data/NMT/data_en_ge.json --adapter_bottleneck 300 --lr 0.0005 --distillation
# evaluate
python ./evaluate.py --task mt --no_sample --adapter_bottleneck 300 --model_checkpoint runs/$model_checkpoint

#nlg
python ./train.py --gradient_accumulation_steps=4 --max_history=2 --train_batch_size=8 --valid_batch_size=8 --n_epochs 10 --task nlg --dataset_path data/NLG/data.json --lr 0.005 --adapter_bottleneck 10
python ./evaluate.py --task nlg --no_sample --model_checkpoint runs/$model_checkpoint --adapter_bottleneck 10

#persona
python ./train.py --gradient_accumulation_steps=2 --max_history=2 --train_batch_size=8 --valid_batch_size=8 --n_epochs 3 --task dialogue --dataset_path data/persona/data.json --lr 0.001 --adapter_bottleneck 100
python ./evaluate.py --task dialogue --no_sample --max_history=2 --model_checkpoint runs/$model_checkpoint --adapter_bottleneck 100

#summarization
python ./train.py --gradient_accumulation_steps=8 --max_history=2 --train_batch_size=4 --valid_batch_size=4 --n_epochs 10 --task summarization --dataset_path data/CNNDAILY/data.json --lr 0.0005 --adapter_bottleneck 100
python ./sentence_distiller.py --task summarization --no_sample --max_history=2 --model_checkpoint runs/$fully_finetuned_gpt2_checkpoint
python ./train.py --gradient_accumulation_steps=8 --max_history=2 --train_batch_size=4 --valid_batch_size=4 --n_epochs 5 --task summarization --dataset_path data/CNNDAILY/data.json --lr 0.0005 --adapter_bottleneck 100 --distillation
python ./evaluate.py --task summarization --no_sample --model_checkpoint runs/$model_checkpoint  --adapter_bottleneck 100

# combine all the adapters and task embedding into single model
# line 68 of combine_all.py to pull the list of checkpoint
python combine_all.py
python ./evaluate_vlm.py --task mt --no_sample --model_checkpoint runs/VLM

#multitask training
python ./train_vlm.py --gradient_accumulation_steps=16 --train_batch_size=1 --valid_batch_size=1 --n_epochs 3

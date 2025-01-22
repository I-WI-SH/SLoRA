CUDA_VISIBLE_DEVICES=2 python examples/run_glue.py \
--model_name_or_path facebook/roberta-base \
--task_name qqp \
--apply_slora --apply_lora --lora_type svd \
--target_rank 1  --lora_r 2  \
--reg_orth_coef 0.1 \
--init_warmup 8000 --final_warmup 25000 --mask_interval 100 \
--beta1 0.85 --beta2 0.85 \
--lora_module query,key,value,intermediate,layer.output,attention.output \
--lora_alpha 32 \
--do_train --do_eval \
--max_seq_length 320 \
--per_device_train_batch_size 32  --per_device_eval_batch_size 32 --learning_rate 8e-4 \
--num_train_epochs 5 --warmup_steps 2000 \
--cls_dropout 0.15 --weight_decay 0.01 \
--evaluation_strategy steps --eval_steps 3000 \
--save_strategy steps --save_steps 10000 \
--logging_steps 500 \
--tb_writter_loginterval 100 \
--report_to tensorboard \
--seed 6 \
--root_output_dir ./output/qqp \
--overwrite_output_dir \
--overwrite_cache


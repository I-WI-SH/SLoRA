CUDA_VISIBLE_DEVICES=5 python examples/run_glue.py \
--model_name_or_path microsoft/deberta-v3-base \
--task_name cola \
--apply_lora --apply_slora --lora_type svd \
--target_rank 4   --lora_r 8   \
--reg_orth_coef 0.1 \
--init_warmup 800 --final_warmup 3500 --mask_interval 10 \
--beta1 0.85 --beta2 0.85 \
--lora_module query,key,value,intermediate,layer.output,attention.output \
--lora_alpha 32 \
--do_train --do_eval --max_seq_length 64 \
--per_device_train_batch_size 32 --per_device_eval_batch_size 32 --learning_rate 8e-4 \
--num_train_epochs 25 --warmup_steps 100 \
--cls_dropout 0.10 --weight_decay 0.00 \
--evaluation_strategy steps --eval_steps 100 \
--save_strategy steps --save_steps 10000 \
--logging_steps 10 \
--tb_writter_loginterval 10 \
--report_to tensorboard \
--seed 6 \
--root_output_dir ./output/cola \
--overwrite_output_dir \
--overwrite_cache




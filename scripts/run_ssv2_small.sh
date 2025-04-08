### ViT
# 1-shot
python3 run_train.py --method TEAM --backbone ViT --test_later --learning_rate 0.0005 --shot 1 --agg_num 50 --num_workers 4 --tasks_per_batch 4 --dataset dataset_path/ssv2_small_FSAR
seq 500 500 10000 | parallel -j 5 'python3 run_eval.py --method TEAM --backbone ViT --shot 1 --agg_num 50 --num_workers 4 --dataset dataset_path/ssv2_small_FSAR -pc work/ssv2_small/TEAM/ViT/1-shot/an50/checkpoint_{}.pt'
python3 run_remain_best.py --dir work/ssv2_small/TEAM/ViT/1-shot/an50

# 5-shot
python3 run_train.py --method TEAM --backbone ViT --test_later --learning_rate 0.0005 --shot 5 --agg_num 80 --num_workers 4 --tasks_per_batch 4 --dataset dataset_path/ssv2_small_FSAR
seq 500 500 10000 | parallel -j 5 'python3 run_eval.py --method TEAM --backbone ViT --shot 5 --agg_num 80 --num_workers 4 --dataset dataset_path/ssv2_small_FSAR -pc work/ssv2_small/TEAM/ViT/1-shot/an80/checkpoint_{}.pt'
python3 run_remain_best.py --dir work/ssv2_small/TEAM/ViT/5-shot/an80
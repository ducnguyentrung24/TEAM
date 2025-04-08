### ResNet
# 1-shot
python3 run_train.py --method TEAM --backbone ResNet --test_later --learning_rate 0.001 --shot 1 --agg_num 60 --num_workers 4 --tasks_per_batch 16 --dataset dataset_path/hmdb51_FSAR
seq 500 500 10000 | parallel -j 5 'python3 run_eval.py --method TEAM --backbone ResNet --shot 1 --agg_num 60 --num_workers 4 --dataset dataset_path/hmdb51_FSAR -pc work/hmdb/TEAM/ResNet/1-shot/an60/checkpoint_{}.pt'
python3 run_remain_best.py --dir work/hmdb/TEAM/ResNet/1-shot/an60

# 5-shot
python3 run_train.py --method TEAM --backbone ResNet --test_later --learning_rate 0.001 --shot 5 --agg_num 70 --num_workers 4 --tasks_per_batch 16 --dataset dataset_path/hmdb51_FSAR
seq 500 500 10000 | parallel -j 5 'python3 run_eval.py --method TEAM --backbone ResNet --shot 5 --agg_num 70 --num_workers 4 --dataset dataset_path/hmdb51_FSAR -pc work/hmdb/TEAM/ResNet/5-shot/an70/checkpoint_{}.pt'
python3 run_remain_best.py --dir work/hmdb/TEAM/ResNet/5-shot/an70

### ViT
# 1-shot
python3 run_train.py --method TEAM --backbone ViT --test_later --learning_rate 0.0001 --shot 1 --agg_num 50 --num_workers 4 --tasks_per_batch 4 --dataset dataset_path/hmdb51_FSAR
seq 500 500 10000 | parallel -j 5 'python3 run_eval.py --method TEAM --backbone ViT --shot 1 --agg_num 50 --num_workers 4 --dataset dataset_path/hmdb51_FSAR -pc work/hmdb/TEAM/ViT/1-shot/an50/checkpoint_{}.pt'
python3 run_remain_best.py --dir work/hmdb/TEAM/ViT/1-shot/an50

# 5-shot
python3 run_train.py --method TEAM --backbone ViT --test_later --learning_rate 0.0001 --shot 5 --agg_num 60 --num_workers 4 --tasks_per_batch 4 --dataset dataset_path/hmdb51_FSAR
seq 500 500 10000 | parallel -j 5 'python3 run_eval.py --method TEAM --backbone ViT --shot 5 --agg_num 60 --num_workers 4 --dataset dataset_path/hmdb51_FSAR -pc work/hmdb/TEAM/ViT/1-shot/an60/checkpoint_{}.pt'
python3 run_remain_best.py --dir work/hmdb/TEAM/ViT/5-shot/an60
### ResNet
# 1-shot
python3 run_train.py --method TEAM --backbone ResNet --test_later --learning_rate 0.001 --shot 1 --agg_num 60 --num_workers 4 --tasks_per_batch 16 --dataset dataset_path/kinetics_FSAR
seq 500 500 10000 | parallel -j 5 'python3 run_eval.py --method TEAM --backbone ResNet --shot 1 --agg_num 60 --num_workers 4 --dataset dataset_path/kinetics_FSAR -pc work/kinetics100/TEAM/ResNet/1-shot/an60/checkpoint_{}.pt'
python3 run_remain_best.py --dir work/kinetics100/TEAM/ResNet/1-shot/an60

# 5-shot
python3 run_train.py --method TEAM --backbone ResNet --test_later --learning_rate 0.001 --shot 5 --agg_num 80 --num_workers 4 --tasks_per_batch 16 --dataset dataset_path/kinetics_FSAR
seq 500 500 10000 | parallel -j 5 'python3 run_eval.py --method TEAM --backbone ResNet --shot 5 --agg_num 80 --num_workers 4 --dataset dataset_path/kinetics_FSAR -pc work/kinetics100/TEAM/ResNet/5-shot/an80/checkpoint_{}.pt'
python3 run_remain_best.py --dir work/kinetics100/TEAM/ResNet/5-shot/an80

### ViT
# 1-shot
python3 run_train.py --method TEAM --backbone ViT --test_later --learning_rate 0.0001 --shot 1 --agg_num 80 --num_workers 4 --tasks_per_batch 4 --dataset dataset_path/kinetics_FSAR
seq 500 500 10000 | parallel -j 5 'python3 run_eval.py --method TEAM --backbone ViT --shot 1 --agg_num 80 --num_workers 4 --dataset dataset_path/kinetics_FSAR -pc work/kinetics100/TEAM/ViT/1-shot/an80/checkpoint_{}.pt'
python3 run_remain_best.py --dir work/kinetics100/TEAM/ViT/1-shot/an80

# 5-shot
python3 run_train.py --method TEAM --backbone ViT --test_later --learning_rate 0.0001 --shot 5 --agg_num 80 --num_workers 4 --tasks_per_batch 4 --dataset dataset_path/kinetics_FSAR
seq 500 500 10000 | parallel -j 5 'python3 run_eval.py --method TEAM --backbone ViT --shot 5 --agg_num 80 --num_workers 4 --dataset dataset_path/kinetics_FSAR -pc work/kinetics100/TEAM/ViT/1-shot/an80/checkpoint_{}.pt'
python3 run_remain_best.py --dir work/kinetics100/TEAM/ViT/5-shot/an80
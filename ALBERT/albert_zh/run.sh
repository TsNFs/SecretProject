export BERT_BASE_DIR=./albert_xlarge_zh
export TEXT_DIR=./lcqmc
nohup python3 run_classifier.py   --task_name=lcqmc_pair   --do_train=true   --do_eval=true   --data_dir=$TEXT_DIR   --vocab_file=./albert_config/vocab.txt  \
    --bert_config_file=./albert_config/albert_config_tiny.json --max_seq_length=128 --train_batch_size=64   --learning_rate=1e-4  --num_train_epochs=20 \
    --output_dir=albert_lcqmc_checkpoints --init_checkpoint=$BERT_BASE_DIR/albert_model.ckpt > res.txt & 2>&1

import paddle.fluid as fluid
import paddlehub as hub

# 拿到ERNIE预训练模型 置为可调参
module = hub.Module(name="ernie")
inputs, outputs, program = module.context(trainable="True", max_seq_len=128)
pooled_output = outputs["pooled_output"]

# 准备数据
ds = hub.dataset.ChnSentiCorp()
# for e in ds.get_train_examples():
#     print(e.text_a, e.label)
reader = hub.reader.ClassifyReader(dataset=ds, vocab_path=module.get_vocab_path(), max_seq_len=128)

# 优化策略 适用ERNIE和BERT
strategy=hub.AdamWeightDecayStrategy(
    learning_rate=1e-4,
    lr_scheduler="linear_decay",
    warmup_proportion=0.0,
    weight_decay=0.01
)

# 运行配置
config = hub.RunConfig(
    use_cuda=True,
    num_epoch=5,
    batch_size=64,
    strategy=strategy)

feed_list = [
    inputs["input_ids"].name,
    inputs["position_ids"].name,
    inputs["segment_ids"].name,
    inputs["input_mask"].name
]

# 创建下游学习任务
cls_task = hub.TextClassifierTask(
    data_reader=reader,
    feature=pooled_output,
    feed_list=feed_list,
    num_classes=ds.num_labels,
    config=config)

# 微调训练
cls_task.finetune_and_eval()
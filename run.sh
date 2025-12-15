#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
. ./path.sh || exit 1;

stage=5
stop_stage=5

# data_url=www.openslr.org/resources/60
# data_dir=/mnt/lyuxiang.lx/data/tts/openslr/libritts
# pretrained_model_dir=../../../pretrained_models/CosyVoice2-0.5B

data_dir=/data/tts-data/paimon_4k
pretrained_model_dir=/data/tts-models/cosyvoice2-0.5b
output_model_dir=/data/tts-models/cosyvoice2-0.5b-paimon

# if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
#   echo "Data Download"
#   for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
#     local/download_and_untar.sh ${data_dir} ${data_url} ${part}
#   done
# fi

#######################################
# 数据预处理
#######################################

# 把“一个目录里成对的 .wav 和 .normalized.txt” 整理成 Kaldi 训练/解码时最常用的那 4 个“表文件”
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Data preparation, prepare wav.scp/text/utt2spk/spk2utt"
  # 下面的文件夹要换成我们自己的数据集名称
  for x in test train; do
    mkdir -p data/$x
    python local/prepare_data.py --src_dir $data_dir/$x --des_dir data/$x
  done
fi

# 用cosyvoice中自带的cam++声纹模型onnx提取说话人embedding
# 存成 utt2embedding.pt 和 spk2embedding.pt，后面做说话人相关任务（区分、聚类、多说话人 TTS 等）直接拿来用。
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Extract campplus speaker embedding, you will get spk2embedding.pt and utt2embedding.pt in data/$x dir"
  for x in test train; do
    tools/extract_embedding.py --dir data/$x \
      --onnx_path $pretrained_model_dir/campplus.onnx
  done
fi

# 把每条语音（≤30 s）喂进一个 ONNX 版的 「 Whisper-风格语音离散化 tokenizer 」，输出一串整数编号（speech token），
# 相当于给声音做了“文字化”压缩，结果存成 utt2speech_token.pt 供后续模型使用。
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Extract discrete speech token, you will get utt2speech_token.pt in data/$x dir"
  for x in test train; do
    tools/extract_speech_token.py --dir data/$x \
      --onnx_path $pretrained_model_dir/speech_tokenizer_v2.onnx
  done
fi

# 把准备好的数据整理成训练时需要的 parquet 格式，方便后续大规模分布式训练时高效读取
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare required parquet format data, you should have prepared wav.scp/text/utt2spk/spk2utt/utt2embedding.pt/spk2embedding.pt/utt2speech_token.pt"
  for x in test train; do
    mkdir -p data/$x/parquet
    tools/make_parquet_list.py --num_utts_per_parquet 1000 \
      --num_processes 10 \
      --src_dir data/$x \
      --des_dir data/$x/parquet
  done
fi

# 保存一个spk2info.pt文件到指定的文件夹
spk_id=paimon
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Save your speaker embedding as spk2info.pt"
  for model in llm flow llm_flow; do
    python tools/save_spk2info.py \
      --spk_id $spk_id \
      --origin_model_dir $pretrained_model_dir \
      --target_model_dir $output_model_dir/$model \
      --emb_path data/train/spk2embedding.pt
  done
fi
#######################################
# 微调
#######################################

# train llm
export CUDA_VISIBLE_DEVICES="6,7"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
# train parameters
job_id=1986
dist_backend="nccl"
num_workers=10
prefetch=200
train_engine=torch_ddp
# swanlab audio display parameters
# train
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Run train. We will train llm and flow model sequentially"
  if [ $train_engine == 'deepspeed' ]; then
    echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
  fi
  cat data/train/parquet/data.list > data/train.data.list
  cat data/test/parquet/data.list > data/dev.data.list
  # NOTE will update llm/hift training later
  for model in llm flow; do
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
      cosyvoice/bin/train.py \
      --train_engine $train_engine \
      --config conf/cosyvoice2-paimon.yaml \
      --train_data data/train.data.list \
      --cv_data data/dev.data.list \
      --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
      --model $model \
      --checkpoint $pretrained_model_dir/$model.pt \
      --model_dir `pwd`/exp/cosyvoice2/$model/$train_engine \
      --tensorboard_dir `pwd`/tensorboard/cosyvoice2/$model/$train_engine \
      --target_model_path $output_model_dir \
      --inference_target_text `pwd`/examples/my_tts_text.json \
      --emb_path `pwd`/data/train/spk2embedding.pt \
      --result_inference_dir `pwd`/output/${spk_id}_inference  \
      --spk_id $spk_id \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --use_amp \
      --deepspeed_config ./conf/ds_stage2.json \
      --deepspeed.save_states model+optimizer
  done
fi

# average model
average_num=1
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  for model in llm flow; do
    decode_checkpoint=`pwd`/exp/cosyvoice2/$model/$train_engine/${model}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python cosyvoice/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path `pwd`/exp/cosyvoice2/$model/$train_engine \
      --num ${average_num} \
      --val_best
  done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Export your model for inference speedup. Remember copy your llm or flow model to model_dir"
  for model in llm flow; do
    python cosyvoice/bin/export_jit.py --model_dir $output_model_dir/$model
    python cosyvoice/bin/export_onnx.py --model_dir $output_model_dir/$model
  done
fi





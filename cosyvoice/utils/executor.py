# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from contextlib import nullcontext
import os
import glob
import shutil

import torch
import torch.distributed as dist

from cosyvoice.utils.train_utils import update_parameter_and_lr, log_per_step, log_per_save, batch_forward, batch_backward, save_model, cosyvoice_join
from cosyvoice.inference.inference import cosyvoice_zero_shot_inference,CosyVoiceSpeakerInference
from cosyvoice.inference.audio import swanlab_audio
import random
import json
import re

class Executor:

    def __init__(self, gan: bool = False, ref_model: torch.nn.Module = None, dpo_loss: torch.nn.Module = None):
        self.gan = gan
        self.ref_model = ref_model
        self.dpo_loss = dpo_loss
        self.step = 0
        self.epoch = 0
        self.rank = int(os.environ.get('RANK', 0))
        self.device = torch.device('cuda:{}'.format(self.rank))

    def train_one_epoc(self, model, optimizer, scheduler, train_data_loader, cv_data_loader, writer, info_dict, scaler, group_join, ref_model=None):
        ''' Train one epoch
        '''

        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {} rank {}'.format(self.epoch, lr, self.rank))
        logging.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(info_dict['accum_grad']))
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        model.train()
        if self.ref_model is not None:
            self.ref_model.eval()
        model_context = model.join if info_dict['train_engine'] == 'torch_ddp' else nullcontext
        with model_context():
            for batch_idx, batch_dict in enumerate(train_data_loader):
                info_dict["tag"] = "TRAIN"
                ### 训练开始计数
                info_dict["step"] = self.step
                info_dict["epoch"] = self.epoch
                info_dict["batch_idx"] = batch_idx
                ### 监控所有 GPU 进程的同步状态
                if cosyvoice_join(group_join, info_dict):
                    break

                # Disable gradient synchronizations across DDP processes.
                # Within this context, gradients will be accumulated on module
                # variables, which will later be synchronized.
                if info_dict['train_engine'] == 'torch_ddp' and (batch_idx + 1) % info_dict["accum_grad"] != 0:
                    ### 本地GPU累计梯度，不同步其他GPU
                    context = model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    ### GPU之间同步通信
                    context = nullcontext

                with context():
                    ### 前向传播+反向传播
                    ### 微调阶段没有dpo相关参数
                    info_dict = batch_forward(model, batch_dict, scaler, info_dict, ref_model=self.ref_model, dpo_loss=self.dpo_loss)
                    info_dict = batch_backward(model, scaler, info_dict)

                info_dict = update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict)
                log_per_step(writer, info_dict)
                # NOTE specify save_per_step in cosyvoice.yaml if you want to enable step save
                if info_dict['save_per_step'] > 0 and (self.step + 1) % info_dict['save_per_step'] == 0 and \
                   (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    dist.barrier()
                    self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=False)
                    model.train()
                if (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    self.step += 1
        dist.barrier()
        ### save model here
        self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=True)

    def train_one_epoc_gan(self, model, optimizer, scheduler, optimizer_d, scheduler_d, train_data_loader, cv_data_loader,
                           writer, info_dict, scaler, group_join):
        ''' Train one epoch
        '''

        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {} rank {}'.format(self.epoch, lr, self.rank))
        logging.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(info_dict['accum_grad']))
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        model.train()
        model_context = model.join if info_dict['train_engine'] == 'torch_ddp' else nullcontext
        with model_context():
            for batch_idx, batch_dict in enumerate(train_data_loader):
                info_dict["tag"] = "TRAIN"
                info_dict["step"] = self.step
                info_dict["epoch"] = self.epoch
                info_dict["batch_idx"] = batch_idx
                if cosyvoice_join(group_join, info_dict):
                    break

                # Disable gradient synchronizations across DDP processes.
                # Within this context, gradients will be accumulated on module
                # variables, which will later be synchronized.
                if info_dict['train_engine'] == 'torch_ddp' and (batch_idx + 1) % info_dict["accum_grad"] != 0:
                    context = model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    context = nullcontext

                with context():
                    batch_dict['turn'] = 'discriminator'
                    info_dict = batch_forward(model, batch_dict, scaler, info_dict)
                    info_dict = batch_backward(model, scaler, info_dict)
                info_dict = update_parameter_and_lr(model, optimizer_d, scheduler_d, scaler, info_dict)
                optimizer.zero_grad()
                log_per_step(writer, info_dict)
                with context():
                    batch_dict['turn'] = 'generator'
                    info_dict = batch_forward(model, batch_dict, scaler, info_dict)
                    info_dict = batch_backward(model, scaler, info_dict)
                info_dict = update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict)
                optimizer_d.zero_grad()
                log_per_step(writer, info_dict)
                # NOTE specify save_per_step in cosyvoice.yaml if you want to enable step save
                if info_dict['save_per_step'] > 0 and (self.step + 1) % info_dict['save_per_step'] == 0 and \
                   (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    dist.barrier()
                    self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=False)
                    model.train()
                if (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    self.step += 1
        dist.barrier()
        self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=True)

    @torch.inference_mode()
    def cv(self, model, cv_data_loader, writer, info_dict, on_batch_end=True):
        ''' Cross validation on
        '''
        logging.info('Epoch {} Step {} on_batch_end {} CV rank {}'.format(self.epoch, self.step + 1, on_batch_end, self.rank))
        model.eval()
        total_num_utts, total_loss_dict = 0, {}  # avoid division by 0
        for batch_idx, batch_dict in enumerate(cv_data_loader):
            info_dict["tag"] = "CV"
            info_dict["step"] = self.step
            info_dict["epoch"] = self.epoch
            info_dict["batch_idx"] = batch_idx

            num_utts = len(batch_dict["utts"])
            total_num_utts += num_utts

            if self.gan is True:
                batch_dict['turn'] = 'generator'
            info_dict = batch_forward(model, batch_dict, None, info_dict)

            for k, v in info_dict['loss_dict'].items():
                if k not in total_loss_dict:
                    total_loss_dict[k] = []
                total_loss_dict[k].append(v.mean().item() * num_utts)
            log_per_step(None, info_dict)
        for k, v in total_loss_dict.items():
            total_loss_dict[k] = sum(v) / total_num_utts
        info_dict['loss_dict'] = total_loss_dict
        log_per_save(writer, info_dict)
        model_name = 'epoch_{}_whole'.format(self.epoch) if on_batch_end else 'epoch_{}_step_{}'.format(self.epoch, self.step + 1)
        ### save model at the end of epoch
        # save checkpoint, we want to save to the original model path llm.pt/flow.pt
        # and inference new model and display in swanlab audio
        save_model(model, model_name, info_dict)

        # ----------------- 新增代码：在原始保存后添加固定名称的保存 -----------------
        """
        目标：
        1. 在每一个epoch结束后保存的epoch_*_whole.pt模型，通过cosyvoice/bin/average_model.py得到多个checkpoint文件中的平均模型，并输出比如llm.pt或者flow.pt
        2. 使用得到的平均权重文件，替换原始(pretrained model path)的模型权重中对应的文件，每次训练得到新的模型文件
        3. 在每个epoch结束后，使用新的模型文件进行推理，得到测试数据的音频和文本
        4. 将推理得到的音频和文本上传到swanlab中进行展示
        """
        if dist.get_rank() == 0:
            logging.info("Rank 0: Starting automated model averaging, replacement, and inference.")

            ### 改名
            try:
                # 保存的模型地址：exp/cosyvoice2/llm...
                save_model_dir = info_dict['model_dir']
                # 将当前的保存好的模型名称转换成llm.pt或者flow.pt并保存，原始模型不能覆盖
                # 要先读取，然后保存
                current_checkpoint_name = f"{model_name}.pt"
                # 具体的保存的新的模型的地址：exp/cosyvoice2/llm/torch_ddp/epoch_{}_whole.pt
                current_checkpoint_path = os.path.join(save_model_dir, current_checkpoint_name)
                state_dict = torch.load(current_checkpoint_path, map_location='cpu')
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                state_dict = {k: v for k, v in state_dict.items() if k not in {'epoch', 'step'}}
                # 新名字
                # 模型名字，要根据模型类型来确定，llm、flow
                model_type = info_dict['model']
                target_model_name = f"{model_type}.pt"
                target_model_path = os.path.join(save_model_dir, target_model_name)
                # 改名字，但是原始模型不能覆盖,新的地址：exp/cosyvoice2/llm/torch_ddp/llm.pt
                torch.save(state_dict, target_model_path)
                logging.info(f"Successfully modify model name: {target_model_path}")
            except Exception as e:
                logging.error(f"Failed to modify model name: {e}")


            ### 替换
            try:
                # 最终的目标模型地址：/data/tts-models/cosyvoice2-0.5b-mydeimos/llm
                final_target_model_path = os.path.join(info_dict['target_model_path'],model_type)
                # 具体旧的模型的地址，但是最终是新模型的保存地址:/data/tts-models/cosyvoice2-0.5b-mydeimos/llm/llm.pt
                final_model_path = os.path.join(final_target_model_path, target_model_name)
                # 替换
                shutil.copyfile(target_model_path, final_model_path)

                ### 如果是flow模型，需要将llm.pt和flow.pt一起复制给llm_flow文件夹里的模型中
                # 即使没有训练llm.pt也没关系
                # 由于之前已经将所有的模型权重转移过去了，因此直接copy过去就行
                mixed_model_type=None
                if model_type == "flow":
                    mixed_model_type='llm_flow'
                    mixed_dir = os.path.join(info_dict['target_model_path'], mixed_model_type)

                    target_llm_path=os.path.join(info_dict['target_model_path'],'llm/llm.pt')
                    target_flow_path=os.path.join(info_dict['target_model_path'],'flow/flow.pt')

                    # copy到新的模型中
                    shutil.copyfile(target_llm_path,os.path.join(mixed_dir,'llm.pt'))
                    shutil.copyfile(target_flow_path,os.path.join(mixed_dir,'flow.pt'))

                    logging.info(f"If flow model is saved, copy llm.pt and flow.pt to {os.path.join(info_dict['target_model_path'],mixed_model_type)}")
                logging.info(f"Successfully replace original model llm.pt or flow.pt, final_model_path={final_model_path}")
            except Exception as e:
                logging.error(f"Fail replace original model:{e}")

            ### 反正上面乱七八糟的跑完，我只要target_model_path作为model_path


            ### 推理生成audio文件
            result_inference_audio_path = []
            try:
                # 总共的文本内容
                with open(info_dict['inference_target_text'], 'r', encoding='utf-8') as f:
                    texts = json.load(f)
                texts = texts[info_dict['spk_id']]['sft_inference']
                # 根据文本推理得到所有的保存文件地址
                inference=CosyVoiceSpeakerInference(info_dict['spk_id'],os.path.join(info_dict['target_model_path'],model_type))
                output_path = os.path.join(info_dict['result_inference_dir'], model_type)
                for idx, text in enumerate(texts):
                    result_inference_audio_path.append(
                        {
                            "wav_path":inference.speaker_inference(text, idx, output_path,sample_rate=24000),
                            "text_content":text
                        }
                        )
                # 如果有llm_flow文件夹，则用这个模型再跑一次
                if mixed_model_type:
                    inference=CosyVoiceSpeakerInference(info_dict['spk_id'],os.path.join(info_dict['target_model_path'],mixed_model_type))
                    output_path = os.path.join(info_dict['result_inference_dir'], mixed_model_type)
                    for idx, text in enumerate(texts):
                        result_inference_audio_path.append(
                        {
                            "wav_path":inference.speaker_inference(text, idx, output_path,sample_rate=24000),
                            "text_content":text
                        }
                        )
                if len(result_inference_audio_path)==0:
                    logging.info("请输入正确的task_type！")
                else:
                    logging.info(f"Successfully inference and get audio result in {result_inference_audio_path}")
            except Exception as e:
                logging.error(f"Failed to inference at the end of each epoch:{e}")
            ### audio文件上传到swanlab展示
            # 将推理生成的audio文件上传到swanlab
            try:
                swan_config=info_dict['swan_config']
                if len(result_inference_audio_path)==0:
                    logging.info("在输入SwanLab之前，请输入正确的task_type！")
                else:
                    swanlab_audio(swan_config=swan_config, 
                                  results_path=result_inference_audio_path,
                                  text_examples_num=len(texts))
                    logging.info(f"Successfully display output audio in swanlab")
            except Exception as e:
                logging.error(f"Failed to display audio in swanlab: {e}")

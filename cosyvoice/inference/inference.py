import sys
sys.path.append('/home/lxy/tts_project/CosyVoice/third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav

import torchaudio
import argparse
import os
import torch
import json

from pathlib import Path
import logging
import random


# zero_shot inference
def cosyvoice_zero_shot_inference(model_dir, tts_text, spk_id, test_data_dir, result_dir, example_id, task_type, target_sr:int=16000):
    # use cosyvoice2
    cosyvoice = CosyVoice2(model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=False)
    # text file path and wav file path
    text_file_path = f"1_{example_id}.normalized.txt"
    wav_file_path = f"1_{example_id}.wav"

    # whole path
    test_data_path = Path(test_data_dir)
    text_file_path = test_data_path / text_file_path
    wav_file_path = test_data_path / wav_file_path

    # wav prompt text
    prompt_text = Path(text_file_path).read_text(encoding='utf-8').strip()
    print("参考的语音的文本内容：", prompt_text)

    # download prompt speech
    prompt_speech_16k = load_wav(wav_file_path,int(target_sr))

    if task_type == "zero-shot":
        # download your targer text
        text = json.load(open(tts_text, 'r', encoding='utf-8'))[spk_id]['zero-shot'][0]
        print("需要合成的文本内容：", text)
    elif task_type == "cross-lingual":
        # download your targer text
        text = json.load(open(tts_text, 'r', encoding='utf-8'))[spk_id]['cross-lingual'][0]
    elif task_type == "instruction":
        # download your targer text
        text = json.load(open(tts_text, 'r', encoding='utf-8'))[spk_id]['instruction-zero-shot']['text']
        instruction = json.load(open(tts_text, 'r', encoding='utf-8'))[spk_id]['instruction-zero-shot']['instruction']
    else:
        return "请输入正确的task_type！"

    # result save
    os.makedirs(result_dir, exist_ok=True)

    ## usage:zero-shot, cross-lingual, instruction
    if task_type == "zero-shot":
        for _, outputs in enumerate(cosyvoice.inference_zero_shot(tts_text=text, prompt_text=prompt_text, prompt_speech_16k=prompt_speech_16k, stream=False)):
            tts_fn = os.path.join(result_dir, f'zero_shot.wav')
            torchaudio.save(tts_fn, outputs['tts_speech'], cosyvoice.sample_rate)
            return tts_fn
    elif task_type == "cross-lingual":
        for _, outputs in enumerate(cosyvoice.inference_cross_lingual(tts_text=text, prompt_speech_16k=prompt_speech_16k, stream=False)):
            tts_fn = os.path.join(result_dir, f'cross_lingual_zero_shot.wav')
            torchaudio.save(tts_fn, outputs['tts_speech'], cosyvoice.sample_rate)
            return tts_fn
    elif task_type == "instruction":
        for _, outputs in enumerate(cosyvoice.inference_instruct2(tts_text=text, instruct_text=instruction, prompt_speech_16k=prompt_speech_16k, stream=False)):
            tts_fn = os.path.join(result_dir, f'instruction_zero_shot.wav')
            torchaudio.save(tts_fn, outputs['tts_speech'], cosyvoice.sample_rate)
            return tts_fn
    else:
        return "请输入正确的task_type！"
    

# sft inference
class CosyVoiceSpeakerInference:
    def __init__(self, spk_id:str, model_dir:str):
        """
        该部分做SFT之后的推理，使用CosyVoice中的inference_sft函数，这个函数只需要两个参数：
        tts_text: target audio content
        spk_id: speaker id equal to speaker embedding(an example audio speaker emb OR averge data speaker emb)

        只要微调后，让模型学会某个人说话的语音、音色等，直接用该角色对应的embedding作为v就可以生成target content
        spk_id应该对应spk2info.pt中的id序号，因此需要先save这个文件

        这个文件其实在模型sft的时候经过0-3步骤已经生成，其实具体在run.sh的步骤1中实现
        生成的spk2embedding.pt and utt2embedding.pt就是我们要的

        utt2embedding.pt是所有的数据集的embedding
        spk2embedding.pt是所有数据集的embedding取平均

        因此我们可以直接使用这个保存文件，因为有多个人，可以保存到一个文件里，然后读取的时候直接读对应的spk_id就行
        
        因此我们有如下函数：
        1. save_spk2info: 保存对应speaker的embedding，并将spk2info.pt文件保存到output_model_path中
        2. load_spk2emb: 读取原始的spk2embedding.pt文件
        3. speaker_inference: 对应speaker的推理结果 
        
        :param self: Description
        """
        # 你需要设置的spk_id
        self.spk_id = spk_id
        # 你的模型地址
        self.model_dir = model_dir
    
    def save_spk2info(self, emb_path:str):
        """
        load spk2embedding.pt and transfer list to tensor, and then save to spk2info.pt
        """
        try:
            emb = torch.load(emb_path)
            if "spk2embedding.pt" in emb_path:
                # 是取平均之后的embedding文件
                new_spk2emb={
                            self.spk_id:{
                                "embedding":torch.tensor([emb['1']],device='cuda')
                            }
                        }
            else:
                # 所有数据集中的embedding文件
                len_data=len(emb)
                example_num=f"1_{str(random.randint(1, len_data))}"

                new_spk2emb={
                            self.spk_id:{
                                "embedding":torch.tensor([emb[example_num]],device='cuda')
                            }
                        }
            new_spk2emb_path = os.path.join(self.model_dir, "spk2info.pt")
            torch.save(new_spk2emb, new_spk2emb_path)
            logging.info(f"Successfully save spk2info.pt, your path is {new_spk2emb_path}")
            return new_spk2emb_path
        except Exception as e:
            logging.error(f"Fail to save spk2info.pt , your error: {e}")

    def speaker_inference(self, tts_text:str, text_id:str, output_dir:str, sample_rate:int=24000):
        """
        use tts_test and look for spk_id equal to embedding from spk2info.pt as inputs 
        use inputs to inference_sft to inference and save .wav
        """
        model =CosyVoice2(self.model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=False)
        os.makedirs(output_dir, exist_ok=True)
        try:
            for _,outputs in enumerate(model.inference_sft(tts_text=tts_text,spk_id=self.spk_id)):
                tts_fn = os.path.join(output_dir, f'{self.spk_id}_sft_inference_{text_id}.wav')
                torchaudio.save(tts_fn, outputs['tts_speech'], sample_rate=sample_rate)
            logging.info(f"Successfully save your sft inference output in {tts_fn}")
            return tts_fn
        except Exception as e:
            logging.error(f"Failed to sft inference, your error is: {e}")
        
        
    
    

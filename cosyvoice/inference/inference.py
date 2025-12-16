import sys
sys.path.append('/home/lxy/tts_project/cosyvoice-paimon-sft/third_party/Matcha-TTS')
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
        texts = json.load(open(tts_text, 'r', encoding='utf-8'))[spk_id]['zero-shot']
    elif task_type == "cross-lingual":
        # download your targer text
        texts = json.load(open(tts_text, 'r', encoding='utf-8'))[spk_id]['cross-lingual']
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
        for idx, text in enumerate(texts):
            for _, outputs in enumerate(cosyvoice.inference_zero_shot(tts_text=text, prompt_text=prompt_text, prompt_speech_16k=prompt_speech_16k, stream=False)):
                tts_fn = os.path.join(result_dir, f'zero_shot_{idx}.wav')
                torchaudio.save(tts_fn, outputs['tts_speech'], cosyvoice.sample_rate)
    elif task_type == "cross-lingual":
        for idx, text in enumerate(texts):
            for _, outputs in enumerate(cosyvoice.inference_cross_lingual(tts_text=text, prompt_speech_16k=prompt_speech_16k, stream=False)):
                tts_fn = os.path.join(result_dir, f'cross_lingual_zero_shot_{idx}.wav')
                torchaudio.save(tts_fn, outputs['tts_speech'], cosyvoice.sample_rate)
    elif task_type == "instruction":
        for _, outputs in enumerate(cosyvoice.inference_instruct2(tts_text=text, instruct_text=instruction, prompt_speech_16k=prompt_speech_16k, stream=False)):
            tts_fn = os.path.join(result_dir, f'instruction_zero_shot.wav')
            torchaudio.save(tts_fn, outputs['tts_speech'], cosyvoice.sample_rate)
    else:
        return "请输入正确的task_type！"
    

# sft inference
class CosyVoiceSpeakerInference:
    def __init__(self, spk_id:str, model_dir:str):
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
        
        
    
    

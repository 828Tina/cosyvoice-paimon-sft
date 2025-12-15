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

from cosyvoice.inference.inference import CosyVoiceSpeakerInference

def get_args():
    parser = argparse.ArgumentParser(description='inference with your model')
    parser.add_argument('--spk_id', required=True, help='your speaker name or id')
    parser.add_argument('--model_dir', required=True, help='your model path')
    parser.add_argument('--model', required=True, help='your model type')
    parser.add_argument('--emb_path', required=True, help='spk2embedding.pt or utt2embedding.pt in ./data')
    parser.add_argument('--tts_text_path', required=True, help='your target text path(json)')
    parser.add_argument('--output_wav_path', required=True, help='target content audio path')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    spkinference=CosyVoiceSpeakerInference(args.spk_id, args.model_dir)

    # save spk2info.pt
    spk2info_path = spkinference.save_spk2info(args.emb_path)
    print(f"Your spk2info.pt already save in {spk2info_path}")

    # inference
    texts = json.load(open(args.tts_text_path, 'r', encoding='utf-8'))
    texts = texts[args.spk_id]['sft_inference']

    # save dir
    output_path = os.path.join(args.output_wav_path, args.model)
    results_path=[]
    for idx, text in enumerate(texts):
        results_path.append(spkinference.speaker_inference(text, idx, output_path,sample_rate=24000))
    return results_path

if __name__ =="__main__":
    main()
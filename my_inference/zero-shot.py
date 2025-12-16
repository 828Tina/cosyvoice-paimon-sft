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

from cosyvoice.inference.inference import cosyvoice_zero_shot_inference

def get_args():
    parser = argparse.ArgumentParser(description='inference with your model')
    parser.add_argument('--model_dir', required=True, help='your model path')
    parser.add_argument('--spk_id', required=True, help='spk id')
    parser.add_argument('--tts_text', required=True, help='tts text file(json) path')
    parser.add_argument('--result_dir', required=True, help='asr result file path')
    parser.add_argument('--test_data_dir', required=True, help='test data file path')
    parser.add_argument('--target_sr', required=True, help='target sampling rate')
    parser.add_argument('--example_id', required=True, help='spk wav(your train file) id')
    parser.add_argument('--task_type', required=True, help='zero-shot、cross-lingual、instruction')
    args = parser.parse_args()
    return args

def main():
    # your parameters
    args = get_args()
    # inference
    cosyvoice_zero_shot_inference(args.model_dir, args.tts_text, args.spk_id,
                        args.test_data_dir, args.result_dir, args.example_id, 
                        args.task_type, args.target_sr)

if __name__ == "__main__":
    main()


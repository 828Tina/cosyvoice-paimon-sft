import argparse
import swanlab
import os
from pathlib import Path
import json

from cosyvoice.inference.audio import swanlab_audio

def args_parse():
    parser = argparse.ArgumentParser(description='Upload audio to SwanLab')
    parser.add_argument('--swan_config', required=True, help='SwanLab Config')
    parser.add_argument('--texts_path', type=str, required=True, help='Your output wav paths')
    parser.add_argument('--text_examples_num', type=int, required=True, help='example texts num')
    return parser.parse_args()

def main():
    args = args_parse()
    # swanlab init
    swanlab_audio(args.swan_config, args.texts_path, args.text_examples_num)

if __name__ == "__main__":
    main()
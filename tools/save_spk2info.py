import os
import json
import argparse
import sys
sys.path.append('/home/lxy/tts_project/CosyVoice/third_party/Matcha-TTS')
from cosyvoice.inference.inference import CosyVoiceSpeakerInference

import shutil
from pathlib import Path
import logging


def get_args():
    parser = argparse.ArgumentParser(description='save spk2info.pt')
    parser.add_argument('--spk_id', required=True, help='your speaker name or id')
    parser.add_argument('--origin_model_dir', required=True, help='your model path')
    parser.add_argument('--target_model_dir', required=True, help='your model path')
    parser.add_argument('--emb_path', required=True, help='spk2embedding.pt or utt2embedding.pt in ./data')
    return parser.parse_args()

def main():
    args=get_args()
    # 1. 准备目标目录
    target_dir=Path(args.target_model_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 2. 按需复制原始模型参数（仅当目标目录为空时才复制，避免重复）
    origin_dir=Path(args.origin_model_dir)
    if not any(target_dir.iterdir()):          # 空目录才复制
        logging.info("目标目录为空，开始复制原始模型参数...")
        for item in origin_dir.iterdir():
            dest = target_dir / item.name
            if item.is_file():
                shutil.copy2(item, dest)
            else:
                shutil.copytree(item, dest, dirs_exist_ok=True)
        logging.info("原始模型参数复制完成。")
    else:
        logging.info("目标目录已存在文件，跳过复制步骤。")

    # 3. 保存说话人嵌入
    spkinference=CosyVoiceSpeakerInference(args.spk_id, args.target_model_dir)

    # save spk2info.pt
    spk2info_path = spkinference.save_spk2info(args.emb_path)
    print(f"Your spk2info.pt already save in {spk2info_path}")

if __name__=="__main__":
    main()
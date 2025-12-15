import os
import shutil
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='inference with your model')
    parser.add_argument('--spk_id', required=True, help='your speaker name or id')
    parser.add_argument('--orig_data_path', required=True, help='original data save path')
    parser.add_argument('--target_data_path', required=True, help='target data save path')
    args = parser.parse_args()
    return args

def get_wav_lab_pairs(folder_path):
    wav_lab_pairs = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.wav'):
            wav_path = os.path.join(folder_path, filename)
            lab_filename = filename[:-4] + '.lab'
            lab_path = os.path.join(folder_path, lab_filename)
            if os.path.exists(lab_path):
                wav_lab_pairs.append((wav_path, lab_path))
    return wav_lab_pairs

def process_files(folder_path, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    wav_lab_pairs = get_wav_lab_pairs(folder_path)

    for idx, (wav_path, lab_path) in enumerate(wav_lab_pairs, start=1):
        new_wav_name = f"{1}_{idx}.wav"
        new_txt_name = f"{1}_{idx}.normalized.txt"

        new_wav_path = os.path.join(save_dir, new_wav_name)
        new_txt_path = os.path.join(save_dir, new_txt_name)

        shutil.copy2(wav_path, new_wav_path) 
        with open(lab_path, 'r', encoding='utf-8') as lab_file:
            lab_content = lab_file.read()
        with open(new_txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(lab_content)
    print(f"Processed {len(wav_lab_pairs)} file pairs from {folder_path} to {save_dir}.")

def split_into_subfolders(base_dir, subfolder_prefix="paimon", max_files_per_folder=5000):
    all_files = sorted(os.listdir(base_dir))
    total_files = len(all_files) // 2  # 每对文件算作一个单位

    for i in range((total_files + max_files_per_folder - 1) // max_files_per_folder):
        # 子文件夹命名
        subfolder_name = f"{subfolder_prefix}_{i + 1}"
        subfolder_path = os.path.join(base_dir, subfolder_name)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        # 起始保存和结束索引
        start_idx = i * max_files_per_folder
        end_idx = min(start_idx + max_files_per_folder, total_files)

        for j in range(start_idx, end_idx):
            wav_filename = f"1_{j + 1}.wav"
            txt_filename = f"1_{j + 1}.normalized.txt"

            shutil.move(os.path.join(base_dir, wav_filename), os.path.join(subfolder_path, wav_filename))
            shutil.move(os.path.join(base_dir, txt_filename), os.path.join(subfolder_path, txt_filename))

    print(f"Split files into subfolders in {base_dir}.")

def main():
    args=get_args()

    os.makedirs(args.target_data_path,exist_ok=True)

    process_files(args.orig_data_path, args.target_data_path)
    split_into_subfolders(args.target_data_path, subfolder_prefix=args.spk_id, max_files_per_folder=5000)

if __name__ == "__main__":
    main()
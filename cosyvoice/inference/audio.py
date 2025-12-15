import swanlab
import json
import logging
import os

EXAMPLE_DIR="/data/tts-data/paimon_all/paimon_1"

def swanlab_audio(swan_config, results_path:list, text_examples_num:int):
    # Upload text
    # llm:text_examples_num flow:2*text_examples_num
    if len(results_path)==text_examples_num:
        llm_audio_list=[]
        for _, path in enumerate(results_path):
            # {"wav_path","text_content"}
            text_content=path['text_content']
            wav_path=path['wav_path']
            audio = swanlab.Audio(wav_path,caption=text_content)
            # multi steps save
            llm_audio_list.append(audio)
            # swanlab log
        swanlab.log({"llm_audio":llm_audio_list})
    else:
        if len(results_path)==text_examples_num*2:
            flow_audio_list=[]
            llm_flow_audio_list=[]
            for _,path in enumerate(results_path):
                text_content=path['text_content']
                wav_path=path['wav_path']
                audio = swanlab.Audio(wav_path,caption=text_content)
                if "llm_flow" not in wav_path:
                    # swanlab log
                    flow_audio_list.append(audio)
                else:
                    llm_flow_audio_list.append(audio)
            swanlab.log({"flow_audio":flow_audio_list})
            swanlab.log({"llm_flow_audio":llm_flow_audio_list})
        else:
            logging.info("Failed to save correct num of wav")
    # example data
    text_path=os.path.join(EXAMPLE_DIR,"1_99.normalized.txt")
    wav_path=os.path.join(EXAMPLE_DIR,"1_99.wav")

    with open(text_path,"r",encoding="utf-8") as f:
        content=f.read()
    audio=swanlab.Audio(wav_path,caption=content)
    swanlab.log({"example: same as the last one":audio})
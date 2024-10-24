import os
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

import subprocess
import numpy as np
import soundfile as sf
import torch
import gc
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from glob import glob
from pathlib import Path
from re import split
from io import BytesIO
from pydub import AudioSegment
from random import choice, randint
from hashlib import md5

#===============推理预备================
def pre_infer(config_path, output_path):
    global tts_config, tts_pipeline
    if config_path in [None, ""]:
        config_path = "GPT-SoVITS/configs/tts_infer.yaml"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    tts_config = TTS_Config(config_path)
    print(tts_config)
    tts_pipeline = TTS(tts_config)
    
def load_weights(gpt, sovits):
    if gpt != "":
        tts_pipeline.init_t2s_weights(gpt)
    if sovits != "":
        tts_pipeline.init_vits_weights(sovits)
    
#===============推理函数================
def pack_ogg(io_buffer:BytesIO, data:np.ndarray, rate:int):
    with sf.SoundFile(io_buffer, mode='w', samplerate=rate, channels=1, format='ogg') as audio_file:
        audio_file.write(data)
    return io_buffer


def pack_raw(io_buffer:BytesIO, data:np.ndarray, rate:int):
    io_buffer.write(data.tobytes())
    return io_buffer


def pack_wav(io_buffer:BytesIO, data:np.ndarray, rate:int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format='wav')
    return io_buffer

def pack_aac(io_buffer:BytesIO, data:np.ndarray, rate:int):
    process = subprocess.Popen([
        'ffmpeg',
        '-f', 's16le',  # 输入16位有符号小端整数PCM
        '-ar', str(rate),  # 设置采样率
        '-ac', '1',  # 单声道
        '-i', 'pipe:0',  # 从管道读取输入
        '-c:a', 'aac',  # 音频编码器为AAC
        '-b:a', '192k',  # 比特率
        '-vn',  # 不包含视频
        '-f', 'adts',  # 输出AAC数据流格式
        'pipe:1'  # 将输出写入管道
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer

def pack_audio(io_buffer:BytesIO, data:np.ndarray, rate:int, media_type:str):
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer

def tts_infer(text, text_lang, ref_audio_path, prompt_text, prompt_lang, top_k, top_p, temperature, text_split_method, batch_size, batch_threshold, split_bucket, speed_facter, fragment_interval, seed, media_type, parallel_infer, repetition_penalty):
    t_lang = ["all_zh","en","all_ja","all_yue","all_ko","zh","ja","yue","ko","auto","auto_yue"][["中文","英语","日语","粤语","韩语","中英混合","日英混合","粤英混合","韩英混合","多语种混合","多语种混合(粤语)"].index(text_lang)]
    p_lang = ["all_zh","en","all_ja","all_yue","all_ko","zh","ja","yue","ko","auto","auto_yue"][["中文","英语","日语","粤语","韩语","中英混合","日英混合","粤英混合","韩英混合","多语种混合","多语种混合(粤语)"].index(prompt_lang)]
    cut_method = ["cut0","cut1","cut2","cut3","cut4","cut5"][["不切","凑四句一切","凑50字一切","按中文句号。切","按英文句号.切","按标点符号切"].index(text_split_method)]
    infer_dict = {
        "text": text,
        "text_lang": t_lang,
        "ref_audio_path": ref_audio_path,
        "prompt_text": prompt_text,
        "prompt_lang": p_lang,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": cut_method,
        "batch_size": batch_size,
        "batch_threshold": batch_threshold,
        "split_bucket": split_bucket,
        "speed_factor": speed_facter,
        "fragment_interval": fragment_interval,
        "seed": seed,
        "media_type": media_type,
        "streaming_mode": False,
        "parallel_infer": parallel_infer,
        "repetition_penalty": repetition_penalty
    }
    with torch.no_grad():
        tts_gen = tts_pipeline.run(infer_dict)
        sr, audio = next(tts_gen)
        torch.cuda.empty_cache()
        gc.collect()
    
    audio = pack_audio(BytesIO(), audio, sr, media_type).getvalue()
    
    return audio

def audio_md5(audio):
    audio_md5 = md5(audio).hexdigest()
    return audio_md5

#===============通用函数================
# 获取模型列表
def get_models(model_path):
    models = glob(f"{model_path}/*")
    model_list = []
    for mod in models:
        model_name = Path(mod).name
        if model_name != "default":
            model_list.append(model_name)
    return model_list

# 根据模型名称获取角色列表
def get_speakers(model_path):
    static_spk = glob(f"{model_path}/reference_audios/emotions/*")
    random_spk = glob(f"{model_path}/reference_audios/randoms/*")
    s_speakers = []
    r_speakers = []
    if len(static_spk) == 0:
        static_spk = ["不支持的合成方式"]
    if len(random_spk) == 0:
        random_spk = ["不支持的合成方式"]
    for spk in static_spk:
        speaker_name = Path(spk).name
        s_speakers.append(speaker_name)
    for spk in random_spk:
        speaker_name = Path(spk).name
        r_speakers.append(speaker_name)
    return s_speakers, r_speakers

# 根据情感参考音频文件名分离情感名称和参考文本
def get_emotion_text(file_name):
    emotion = split("【|】", file_name)[1]
    emo_text = split("【|】", file_name)[2]
    return emotion, emo_text

# 随机种子码
def random_seed():
    seed = randint(0, 4294967295)
    return seed

def reset_seed():
    return -1

#判断参考音频长度是否符合要求
def check_audio_length(audio_path):
    audio = AudioSegment.from_file(audio_path)
    audio_length = len(audio) / 1000
    if audio_length < 3 or audio_length > 10:
        return False
    else:
        return True
    
#获取模型路径
def get_model_path(model_path):
    gpt_models = glob(f"{model_path}/*.ckpt")
    sovits_models = glob(f"{model_path}/*.pth")
    gpt_model = gpt_models[0] if len(gpt_models) > 0 else ""
    sovits_model = sovits_models[0] if len(sovits_models) > 0 else ""
    return gpt_model, sovits_model
    
#=========================================
#=================接口函数=================
#---------------固定角色情感---------------
# 获取固定角色情感参考音频池支持的语言
def get_s_lang(model_path, speaker_name):
    s_ref_langs = glob(f"{model_path}/reference_audios/emotions/{speaker_name}/*")
    s_langs = []
    if len(s_ref_langs) == 0:
        s_langs.append("None")
    else:
        for lang in s_ref_langs:
            lang_name = Path(lang).name
            s_langs.append(lang_name)
    return s_langs

# 获取固定角色情感参考音频池支持的情感
def get_s_emotions(model_path, speaker_name, lang_name):
    s_ref_emotions = glob(f"{model_path}/reference_audios/emotions/{speaker_name}/{lang_name}/*")
    s_emotions = []
    if len(s_ref_emotions) == 0:
        s_emotions.append("None")
    else:
        for emo in s_ref_emotions:
            emotion_file = Path(emo).name
            emotion, emo_text = get_emotion_text(emotion_file)
            s_emotions.append(emotion)
    return s_emotions

# 获取固定角色情感参考音频池的情绪参考文本
def get_s_emo_text(model_path, speaker_name, lang_name, emotion):
    s_ref_emotions = glob(f"{model_path}/reference_audios/emotions/{speaker_name}/{lang_name}/*")
    if len(s_ref_emotions) != 0:
        for emo in s_ref_emotions:
            emotion_file = Path(emo).name.replace(".wav", "")
            emo_emotion, emo_text = get_emotion_text(emotion_file)
            if emotion == emo_emotion:
                return emo_text
    else:
        return "None"
        
# 根据角色名更新固定角色情感参考音频池的语言和情感
def update_s_lang_emo(model_path, speaker_name):
    s_langs = get_s_lang(model_path, speaker_name)
    s_emos = get_s_emotions(model_path, speaker_name, s_langs[0])
    s_emo_text = get_s_emo_text(model_path, speaker_name, s_langs[0], s_emos[0])
    return s_langs, s_emos, s_emo_text

# 根据语言更新固定角色情感参考音频池的情感
def update_s_emo(model_path, speaker_name, lang_name):
    s_emos = get_s_emotions(model_path, speaker_name, lang_name)
    s_emo_text = get_s_emo_text(model_path, speaker_name, lang_name, s_emos[0])
    return s_emos, s_emo_text

# 根据情感更新固定角色情感参考音频池的情绪文本
def update_s_emo_text(model_path, speaker_name, lang_name, emotion):
    s_emo_text = get_s_emo_text(model_path, speaker_name, lang_name, emotion)
    return s_emo_text

#---------------随机参考音频---------------
# 获取随机参考音频池支持的语言
def get_r_lang(model_path, speaker_name):
    r_ref_langs = glob(f"{model_path}/reference_audios/randoms/{speaker_name}/*")
    r_langs = []
    if len(r_ref_langs) == 0:
        r_langs.append("None")
    else:
        for lang in r_ref_langs:
            lang_name = Path(lang).name
            r_langs.append(lang_name)
    return r_langs

# 获取随机参考音频池的随机参考音频文件名以及参考文本
def get_ref_audio_random(model_path, speaker_name, lang_name):
    r_ref_audios = glob(f"{model_path}/reference_audios/randoms/{speaker_name}/{lang_name}/*.wav")
    if len(r_ref_audios) == 0:
        ref_audio = "None"
        ref_text = "None"
    else:
        ref_audio = choice(r_ref_audios)
        ref_text_name = ref_audio.replace(".wav", ".lab")
        ref_text = Path(ref_text_name).read_text(encoding="utf-8")
    return ref_audio, ref_text

# 根据角色名更新随机参考音频池的语言
def update_r_lang(model_path, speaker_name):
    r_langs = get_r_lang(model_path, speaker_name)
    ref_audio, ref_text = get_ref_audio_random(model_path, speaker_name, r_langs[0])
    return r_langs, ref_audio, ref_text

# 根据语言更新随机参考音频池的参考音频和参考文本
def update_r_audio_text(model_path, speaker_name, lang_name):
    if model_path == "请选择模型":
        return "请选择模型", "请选择模型"
    else:
        ref_audio, ref_text = get_ref_audio_random(model_path, speaker_name, lang_name)
        return ref_audio, ref_text
#=========================================
#=================加载模型================
def load_model(model_path, model_name):
    model_full_path = f"{model_path}/{model_name}"
    gpt, sovits = get_model_path(model_full_path)
    load_weights(gpt, sovits)
    s_spks, r_spks = get_speakers(model_full_path)
    s_langs = get_s_lang(model_full_path, s_spks[0])
    s_emos = get_s_emotions(model_full_path, s_spks[0], s_langs[0])
    s_emo_text = get_s_emo_text(model_full_path, s_spks[0], s_langs[0], s_emos[0])
    r_langs = get_r_lang(model_full_path, r_spks[0])
    ref_audio, ref_text = get_ref_audio_random(model_full_path, r_spks[0], r_langs[0])
    return model_full_path, s_spks, r_spks, s_langs, s_emos, s_emo_text, r_langs, ref_audio, ref_text
#=========================================
#=================推理模式=================
# 角色与情感
def infer_by_spk_emo(text, language, speaker, emotion, emo_text, emo_lang, top_k, top_p, temperature, cut_method, batch_size, batch_threshold, split_bucket, speed, fragment_interval, seed, media_type, parallel_infer, repetition_penalty, output_path, model_path):
    audio_path = None
    if model_path == "None":
        msg = "请先选择模型！"
    elif emo_lang == "None":
        msg = "当前模型不支持角色与情感合成！"
    elif text == "":
        msg = "请输入要合成的文本！"
    else:
        ref_audio = f"{model_path}/reference_audios/emotions/{speaker}/{emo_lang}/【{emotion}】{emo_text}.wav"
        audio = tts_infer(text, language, ref_audio, emo_text, language, top_k, top_p, temperature, cut_method, batch_size, batch_threshold, split_bucket, speed, fragment_interval, seed, media_type, parallel_infer, repetition_penalty)
        audio_name = audio_md5(audio)
        Path(f"{output_path}/{audio_name}.{media_type}").write_bytes(audio)
        audio_path = f"{output_path}/{audio_name}.{media_type}"
        msg = "合成成功！当前方式：角色与情感"
    return audio_path, msg

# 自定义参考音频
def infer_by_ref_audio(text, language, ref_audio, ref_text, p_lang, top_k, top_p, temperature, cut_method, batch_size, batch_threshold, split_bucket, speed, fragment_interval, seed, media_type, parallel_infer, repetition_penalty, output_path, model_path):
    audio_path = None
    if model_path == "None":
        msg = "请先选择模型！"
    elif text == "":
        msg = "请输入要合成的文本！"
    elif ref_audio is None:
        msg = "请上传参考音频！"
    else:
        ref_audio_length = check_audio_length(ref_audio)
        if not ref_audio_length:
            msg = "参考音频长度应在3-10秒之间！"
        else:
            audio = tts_infer(text, language, ref_audio, ref_text, p_lang, top_k, top_p, temperature, cut_method, batch_size, batch_threshold, split_bucket, speed, fragment_interval, seed, media_type, parallel_infer, repetition_penalty)
            audio_name = audio_md5(audio)
            Path(f"{output_path}/{audio_name}.{media_type}").write_bytes(audio)
            audio_path = f"{output_path}/{audio_name}.{media_type}"
            if ref_text == "":
                msg = "合成成功！当前方式：自定义参考音频（无参考文本）"
            else:
                msg = "合成成功！当前方式：自定义参考音频"
    return audio_path, msg

# 随机参考音频
def infer_by_random_ref(text, language, ref_audio, ref_text, p_lang, top_k, top_p, temperature, cut_method, batch_size, batch_threshold, split_bucket, speed, fragment_interval, seed, media_type, parallel_infer, repetition_penalty, output_path, model_path):
    audio_path = None
    if model_path == "None":
        msg = "请先选择模型！"
    elif p_lang == "None":
        msg = "当前模型不支持随机参考音频合成！"
    elif text == "":
        msg = "请输入要合成的文本！"
    else:
        audio = tts_infer(text, language, ref_audio, ref_text, p_lang, top_k, top_p, temperature, cut_method, batch_size, batch_threshold, split_bucket, speed, fragment_interval, seed, media_type, parallel_infer, repetition_penalty)
        audio_name = audio_md5(audio)
        Path(f"{output_path}/{audio_name}.{media_type}").write_bytes(audio)
        audio_path = f"{output_path}/{audio_name}.{media_type}"
        msg = "合成成功！当前方式：随机参考音频"
    return audio_path, msg
#=========================================
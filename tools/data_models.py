""" 响应模型模块 """

from pydantic import BaseModel
from typing import Literal

#OpenAI推理接口其它参数
class otherParams(BaseModel):
    app_key: str = ""
    text_lang: str = "多语种混合"
    prompt_lang: str = "中文"
    emotion: str = "默认"
    top_k: int = 10
    top_p: float = 1.0
    temperature: float = 1.0
    text_split_method: str = "按标点符号切"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    fragment_interval: float = 0.3
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    sample_steps: int = 16
    if_sr: bool = False
    seed: int = -1

#OpenAI风格的推理接口
class openaiLikeInfer(BaseModel):
    model: Literal["tts-v2", "tts-v3", "tts-v4", "tts-v2Pro", "tts-v2ProPlus"]
    input: str = ""
    voice: str = ""
    response_format: str = "wav"
    speed: float = 1.0
    other_params: otherParams = otherParams()

# 定义请求参数模型
class requestVersion(BaseModel):
    version: str
    
class Shutdown(BaseModel):
    password: str
    
class inferWithEmotions(BaseModel):
    app_key: str = ""
    dl_url: str = ""
    version: str = "v4"
    model_name: str = ""
    prompt_text_lang: str = ""
    emotion: str = ""
    text: str = ""
    text_lang: str = ""
    top_k: int = 10
    top_p: float = 1.0
    temperature: float = 1.0
    text_split_method: str = "按标点符号切"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_facter: float = 1.0
    fragment_interval: float = 0.3
    media_type: str = "wav"
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    seed: int = -1
    sample_steps: int = 16
    if_sr : bool = False
    
class inferWithMulti(BaseModel):
    app_key: str = ""
    dl_url: str = ""
    content : str = ""
    top_k: int = 10
    top_p: float = 1.0
    temperature: float = 1.0
    text_split_method: str = "按标点符号切"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    fragment_interval: float = 0.3
    media_type: str = "wav"
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    seed: int = -1
    sample_steps: int = 16
    if_sr : bool = False
    
    
class inferWithClassic(BaseModel):
    app_key: str = ""
    dl_url: str = ""
    version: str = "v4"
    gpt_model_name: str = ""
    sovits_model_name: str = ""
    ref_audio_path: str = ""
    prompt_text: str = ""
    prompt_text_lang: str = ""
    text: str = ""
    text_lang: str = ""
    top_k: int = 10
    top_p: float = 1.0
    temperature: float = 1.0
    text_split_method: str = "按标点符号切"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_facter: float = 1.0
    fragment_interval: float = 0.3
    media_type: str = "wav"
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    seed: int = -1
    sample_steps: int = 16
    if_sr : bool = False
    
class checkModelInstalled(BaseModel):
    version: str = "v4"
    category: str = ""
    language: str = ""
    model_name: str = ""
    
class installModel(BaseModel):
    version: str = "v4"
    category: str = ""
    language: str = ""
    model_name: str = ""
    dl_url: str = ""
import argparse
import gradio as gr
from tools import my_infer

#==============初始化参数===============
parser = argparse.ArgumentParser(description="GPT-SoVITS api")
parser.add_argument("-c", "--tts_config", type=str, default="GPT_SoVITS/configs/tts_infer.yaml", help="tts_infer路径")
parser.add_argument("-a", "--bind_addr", type=str, default="127.0.0.1", help="default: 127.0.0.1")
parser.add_argument("-p", "--port", type=int, default="9880", help="default: 9880")
parser.add_argument("-m", "--model_path", type=str, default="./models", help="模型路径")
parser.add_argument("-o", "--output", type=str, help="合成音频输出路径", default="./tmp")
args = parser.parse_args()
    
config_path = args.tts_config
bind_addr = args.bind_addr
port = args.port
m_path = args.model_path
o_path = args.output
    
my_infer.pre_infer(config_path, o_path)

#===============通用函数================
# 获取模型列表
def get_models(model_path):
    global model_list
    model_list = my_infer.get_models(model_path)
    
get_models(m_path)

# 根据模型名称获取角色列表
def get_speakers(model_path):
    static_spk, random_spk = my_infer.get_speakers(model_path)
    return static_spk, random_spk

# 根据情感参考音频文件名分离情感名称和参考文本
def get_emotion_text(file_name):
    emotion, emo_text = my_infer.get_emotion_text(file_name)
    return emotion, emo_text

# 随机种子码
def random_seed():
    seed = my_infer.random_seed()
    return seed

def reset_seed():
    return my_infer.reset_seed()

#===============固定角色情感=============
# 获取固定角色情感参考音频池支持的语言
def get_s_lang(model_path, speaker_name):
    s_langs = my_infer.get_s_lang(model_path, speaker_name)
    return s_langs

# 获取固定角色情感参考音频池支持的情感
def get_s_emotions(model_path, speaker_name, lang_name):
    s_emotions = my_infer.get_s_emotions(model_path, speaker_name, lang_name)
    return s_emotions

# 获取固定角色情感参考音频池支持的情感
def get_s_emo_text(model_path, speaker_name, lang_name, emotion):
    emo_text = my_infer.get_s_emo_text(model_path, speaker_name, lang_name, emotion)
    return emo_text

# 根据角色名更新固定角色情感参考音频池的语言和情感
def update_s_lang_emo(model_path, speaker_name):
    s_langs, s_emos, s_emo_text = my_infer.update_s_lang_emo(model_path, speaker_name)
    return gr.update(choices=s_langs, value=s_langs[0]), gr.update(choices=s_emos, value=s_emos[0]), s_emo_text

# 根据语言更新固定角色情感参考音频池的情感
def update_s_emo(model_path, speaker_name, lang_name):
    s_emos, s_emo_text = my_infer.update_s_emo(model_path, speaker_name, lang_name)
    return gr.update(choices=s_emos, value=s_emos[0]), s_emo_text

# 根据情感更新固定角色情感参考音频池的情绪文本
def update_s_emo_text(model_path, speaker_name, lang_name, emotion):
    s_emo_text = my_infer.update_s_emo_text(model_path, speaker_name, lang_name, emotion)
    return s_emo_text

#===============随机参考音频=============
# 获取随机参考音频池支持的语言
def get_r_lang(model_path, speaker_name):
    r_langs = my_infer.get_r_lang(model_path, speaker_name)
    return r_langs

# 获取随机参考音频池的随机参考音频文件名以及参考文本
def get_ref_audio_random(model_path, speaker_name, lang_name):
    ref_file_random, ref_text_random = my_infer.get_ref_audio_random(model_path, speaker_name, lang_name)
    return ref_file_random, ref_text_random

# 根据角色名更新随机参考音频池的语言
def update_r_lang(model_path, speaker_name):
    r_langs, ref_file_random, ref_text_random = my_infer.update_r_lang(model_path, speaker_name)
    return gr.update(choices=r_langs, value=r_langs[0]), ref_file_random, ref_text_random

# 根据语言更新随机参考音频池的参考文本
def update_r_audio_text(model_path, speaker_name, lang_name):
    ref_file_random, ref_text_random = my_infer.update_r_audio_text(model_path, speaker_name, lang_name)
    return ref_file_random, ref_text_random

#===============加载模型===============
def load_model(model_name):
    model_full_path, s_spks, r_spks, s_langs, s_emos, s_emo_text, r_langs, ref_audio, ref_text = my_infer.load_model(m_path, model_name)
    return model_full_path, gr.update(choices=s_spks,value=s_spks[0]), gr.update(choices=s_langs,value=s_langs[0]), gr.update(choices=s_emos,value=s_emos[0]), s_emo_text, gr.update(choices=r_spks,value=r_spks[0]), gr.update(choices=r_langs,value=r_langs[0]), ref_audio, ref_text

#===============开始推理===============
# 角色与情感
def infer_by_spk_emo(text, language, speaker, emotion, emo_text, emo_lang, top_k, top_p, temperature, cut_method, batch_size, batch_threshold, split_bucket, speed, fragment_interval, seed, media_type, parallel_infer, repetition_penalty,  model_path, output_path = o_path):
    audio_path, msg = my_infer.infer_by_spk_emo(text, language, speaker, emotion, emo_text, emo_lang, top_k, top_p, temperature, cut_method, batch_size, batch_threshold, split_bucket, speed, fragment_interval, seed, media_type, parallel_infer, repetition_penalty, output_path, model_path)
    gr.Info(msg)
    return audio_path

# 自定义参考音频
def infer_by_ref_audio(text, language, ref_audio, ref_text, p_lang, top_k, top_p, temperature, cut_method, batch_size, batch_threshold, split_bucket, speed, fragment_interval, seed, media_type, parallel_infer, repetition_penalty, model_path, output_path = o_path):
    audio_path, msg = my_infer.infer_by_ref_audio(text, language, ref_audio, ref_text, p_lang, top_k, top_p, temperature, cut_method, batch_size, batch_threshold, split_bucket, speed, fragment_interval, seed, media_type, parallel_infer, repetition_penalty, output_path, model_path)
    gr.Info(msg)
    return audio_path

# 随机参考音频
def infer_by_random_ref(text, language, ref_audio, ref_text, p_lang, top_k, top_p, temperature, cut_method, batch_size, batch_threshold, split_bucket, speed, fragment_interval, seed, media_type, parallel_infer, repetition_penalty, model_path, output_path = o_path):
    audio_path, msg = my_infer.infer_by_random_ref(text, language, ref_audio, ref_text, p_lang, top_k, top_p, temperature, cut_method, batch_size, batch_threshold, split_bucket, speed, fragment_interval, seed, media_type, parallel_infer, repetition_penalty, output_path, model_path)
    gr.Info(msg)
    return audio_path

#===============推理界面================
style="""
.reflush-button {
    width: 50px;
    margin-top: 12px;
}
.random-button {
    height: 88px;
}
"""

with gr.Blocks(title="GPT-Sovits推理整合包",css=style) as app:
    gr.Markdown("## <center>二次元游戏角色 [GPT-Sovits](https://github.com/RVC-Boss/GPT-SoVITS) 语音合成整合包</center>")
    gr.Markdown("**注：** <br>1. 本整合包由 [AI Hobbyist](https://www.ai-hobbyist.com) 提供，如有问题可点击加群 [AI Hobbyist 交流群](https://qm.qq.com/q/g6iGNZkstW) 或 [Discord](https://discord.gg/eGzeMgYSPD) 或 [发送邮件](mailto:support@acgnai.top) 提问。<br>2. 使用前请阅读注意事项：[点我阅读](https://www.bilibili.com/read/cv36652528)") 
    gr.Markdown("如果需要自行训练，可以点击如下链接获取数据集：[原神数据集](https://www.bilibili.com/read/cv24180458) | [星穹铁道数据集](https://www.bilibili.com/read/cv36649798) | [鸣潮数据集](https://www.bilibili.com/read/cv36092527)")
    default_text = "请选择模型"
    with gr.Tabs():
        with gr.TabItem("推理"):
            with gr.Row():
                with gr.Column(scale=6):
                    with gr.Tab(label="要合成的文本"):
                        text = gr.TextArea(lines=29, label="要合成的文本", placeholder="请输入要合成的文本", show_label=False)
                    with gr.Tab(label="参数设置"):
                        with gr.Column():
                            with gr.TabItem("基本设置"):
                                with gr.Row():
                                    with gr.Column(scale=2):
                                        media_type = gr.Radio(label="音频格式", choices=["wav","ogg", "aac"], value="wav", interactive=True)
                                    with gr.Column(scale=2):
                                        fragment_interval  = gr.Slider(label="分段间隔(秒)",minimum=0.01,maximum=1.0,step=0.01,value=0.3,interactive=True)
                        with gr.Column():
                            with gr.TabItem("并行推理"):
                                with gr.Row():
                                    with gr.Column(scale=2):
                                        parallel_infer = gr.Checkbox(label="启用并行推理", value=True, interactive=True, show_label=True)
                                    with gr.Column(scale=2):
                                        split_bucket = gr.Checkbox(label="启用数据分桶(并行推理时会降低一点计算量)", value=True, interactive=True, show_label=True)
                                with gr.Row():
                                    with gr.Column(scale=2):
                                        batch_size = gr.Slider(minimum=1,maximum=200,step=1,label="批量大小",value=10,interactive=True)
                                    with gr.Column(scale=2):
                                        batch_threshold = gr.Slider(minimum=0,maximum=1,step=0.01,label="批处理阈值",value=0.75,interactive=True)
                        with gr.Column():
                            with gr.TabItem("推理参数"):
                                with gr.Row():
                                    with gr.Column(scale=2):
                                        top_k = gr.Slider(label="前k个采样（Top-k）",minimum=1,maximum=100,step=1,value=10,interactive=True)
                                    with gr.Column(scale=2):
                                        top_p = gr.Slider(label="累计概率采样 (Top-p)",minimum=0.01,maximum=1.0,step=0.01,value=1.0,interactive=True)
                                with gr.Row():
                                    with gr.Column(scale=2):
                                        temperature = gr.Slider(label="温度系数 (Temperature)",minimum=0.01,maximum=1,step=0.01,value=1.0,interactive=True)
                                    with gr.Column(scale=2):
                                        repetition_penalty = gr.Slider(minimum=0,maximum=2,step=0.05,label="重复惩罚",value=1.35,interactive=True)
                with gr.Column(scale=2):
                    with gr.Tab(label="合成选项"):  
                        #with gr.Column():
                            with gr.Column():
                                model = gr.Dropdown(label="模型名称", choices=model_list, value=default_text, interactive=True,allow_custom_value=True)
                                model_path = gr.Textbox(label="已加载模型",interactive=False,value=default_text)
                            with gr.Column():
                                language = gr.Dropdown(label="语言",choices=["中文","英语","日语","粤语","韩语","中英混合","日英混合","粤英混合","韩英混合","多语种混合","多语种混合(粤语)"],value="中文",interactive=True)
                                cut_method = gr.Dropdown(label="切分方式",choices=["不切","凑四句一切","凑50字一切","按中文句号。切","按英文句号.切","按标点符号切"],value="按标点符号切",interactive=True)
                                speed = gr.Slider(label="语速",minimum=0.01,maximum=2.0,step=0.01,value=1.0,interactive=True)
                            with gr.Column():
                                with gr.TabItem("随机种子"):
                                    with gr.Row():
                                        btn6 = gr.Button("🔄", variant="primary", min_width=0, elem_classes=["reflush-button"])
                                        seed_number = gr.Number(label="随机种子",minimum=-1,maximum=4294967295,step=1,value=-1,interactive=True,min_width=200,show_label=False)
                                        btn5 = gr.Button("🎲", variant="primary", min_width=0, elem_classes=["reflush-button"])
                                        btn5.click(random_seed, outputs=[seed_number])
                                        btn6.click(reset_seed, outputs=[seed_number])
            audio = gr.Audio(label="合成结果", interactive=False, type="filepath")
            with gr.Row():
                    with gr.Tabs():
                        with gr.Row():
                            with gr.Column(scale=6):
                                with gr.TabItem(label = "角色与情感"):
                                    with gr.Row():
                                        with gr.Column(scale=2):
                                            speaker = gr.Dropdown(label="角色",choices=[],interactive=True,value=default_text,allow_custom_value=True)
                                        with gr.Column(scale=2):
                                            emo_lang = gr.Dropdown(label="语言",choices=[],interactive=True,value=default_text,allow_custom_value=True)
                                        with gr.Column(scale=2):
                                            emotion = gr.Dropdown(label="情感",choices=[],interactive=True,value=default_text,allow_custom_value=True)
                                    with gr.Row():
                                        emo_text = gr.Textbox(label="情感文本",interactive=False,value=default_text)
                                    with gr.Row():
                                        btn1 = gr.Button("一键合成", variant="primary")

                                with gr.TabItem(label="自定义参考音频"):
                                    ref_audio = gr.Audio(label="参考音频", type="filepath", interactive=True)
                                    with gr.Row():
                                        with gr.Column(scale=5):
                                            ref_text = gr.Textbox(lines=1, label="参考文本", placeholder="请输入参考文本")
                                        with gr.Column(scale=1):
                                            p_lang = gr.Dropdown(label="参考文本语言",choices=["中文","英语","日语","粤语","韩语","中英混合","日英混合","粤英混合","韩英混合","多语种混合","多语种混合(粤语)"],value="中文",interactive=True)
                                    with gr.Row():
                                        btn2 = gr.Button("一键合成", variant="primary")
                                        
                                with gr.TabItem(label="随机参考音频"):
                                    with gr.Row():
                                        with gr.Column(scale=1):
                                            ref_speaker = gr.Dropdown(label="参考角色",choices=[],interactive=True,value=default_text,allow_custom_value=True)
                                        with gr.Column(scale=1):
                                            ref_text_language = gr.Dropdown(label="参考文本语言",choices=[],value=default_text,interactive=True,allow_custom_value=True)
                                        with gr.Column(scale=4):
                                            ref_file_random = gr.Text(label="参考音频路径（自动获取）",interactive=False,value=default_text)
                                    with gr.Row():
                                        with gr.Column(scale=5):
                                            ref_text_random = gr.Textbox(lines=1, label="参考文本（自动获取）", interactive=False,value=default_text)
                                        with gr.Column(scale=1):
                                            btn4 = gr.Button("🎲试试手气", variant="primary", elem_classes=["random-button"])
                                    with gr.Row():
                                        btn3 = gr.Button("一键合成", variant="primary")
            model.change(load_model,inputs=[model],outputs=[model_path,speaker,emo_lang,emotion,emo_text,ref_speaker,ref_text_language,ref_file_random,ref_text_random])
            speaker.change(update_s_lang_emo,inputs=[model_path,speaker],outputs=[emo_lang,emotion,emo_text])
            emo_lang.change(update_s_emo,inputs=[model_path,speaker,emo_lang],outputs=[emotion,emo_text])
            emotion.change(update_s_emo_text,inputs=[model_path,speaker,emo_lang,emotion],outputs=[emo_text])
            ref_speaker.change(update_r_lang,inputs=[model_path,ref_speaker],outputs=[ref_text_language,ref_file_random,ref_text_random])
            ref_text_language.change(update_r_audio_text,inputs=[model_path,ref_speaker,ref_text_language],outputs=[ref_file_random,ref_text_random])
            btn4.click(update_r_audio_text,inputs=[model_path,ref_speaker,ref_text_language],outputs=[ref_file_random,ref_text_random])
            btn1.click(infer_by_spk_emo,inputs=[text,language,speaker,emotion,emo_text,emo_lang,top_k,top_p,temperature,cut_method,batch_size,batch_threshold,split_bucket,speed,fragment_interval,seed_number,media_type,parallel_infer,repetition_penalty,model_path],outputs=[audio])
            btn2.click(infer_by_ref_audio,inputs=[text,language,ref_audio,ref_text,p_lang,top_k,top_p,temperature,cut_method,batch_size,batch_threshold,split_bucket,speed,fragment_interval,seed_number,media_type,parallel_infer,repetition_penalty,model_path],outputs=[audio])
            btn3.click(infer_by_random_ref,inputs=[text,language,ref_file_random,ref_text_random,ref_text_language,top_k,top_p,temperature,cut_method,batch_size,batch_threshold,split_bucket,speed,fragment_interval,seed_number,media_type,parallel_infer,repetition_penalty,model_path],outputs=[audio])
            
            
def main():
    app.launch(server_name=bind_addr, server_port=port, inbrowser=True)
    
if __name__ == "__main__":
    main()
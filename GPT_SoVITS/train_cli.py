from GPT_SoVITS.train_util import open1abc, open1Ba, open1Bb
from config import pretrained_sovits_name, pretrained_gpt_name, cnhubert_path, bert_path
import argparse
import os

def set_version(version):
    os.environ["version"] = version

# D模型
pretrained_sovits_d_models = {
    "v1": "GPT_SoVITS/pretrained_models/s2D488k.pth",
    "v2": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth",
    "v2Pro": "GPT_SoVITS/pretrained_models/v2Pro/s2Dv2Pro.pth",
    "v2ProPlus": "GPT_SoVITS/pretrained_models/v2Pro/s2Dv2ProPlus.pth"
}

# 获取对应版本预训练权重路径
def get_pretrained_paths(version):
    sovits_path = pretrained_sovits_name.get(version, None)
    sovits_d_path = pretrained_sovits_d_models.get(version, None)
    gpt_path = pretrained_gpt_name.get(version, None)
    return sovits_path, gpt_path, sovits_d_path

# 预处理数据集
def preprocess_dataset(version, data_list, raw_wav_dir, exp_name, gpu_n_1a, gpu_n_1Ba, gpu_n_1c):
    if version in ["v1", "v2", "v3", "v4", "v2Pro", "v2ProPlus"]:
        set_version(version)
        sovits, _, _ = get_pretrained_paths(version)
        for progress, *_ in open1abc(
            version,
            data_list,
            raw_wav_dir,
            exp_name,
            gpu_n_1a,
            gpu_n_1Ba,
            gpu_n_1c,
            bert_path,
            cnhubert_path,
            sovits,
            ):
            print(progress)
    else:
        raise ValueError("Unsupported version: %s" % version)
    
# 训练 SoVITS 模型
def train_sovits(version, batch_size, total_epoch, exp_name, text_low_lr_rate, if_save_latest, if_save_every_weights, save_every_epoch, gpu_numbers1Ba, if_grad_ckpt, lora_rank):
    if version in ["v1", "v2", "v3", "v4", "v2Pro", "v2ProPlus"]:
        set_version(version)
        sovits_g, _, sovits_d = get_pretrained_paths(version)
        for progress, *_ in open1Ba(
            version,
            batch_size,
            total_epoch,
            exp_name,
            text_low_lr_rate,
            if_save_latest,
            if_save_every_weights,
            save_every_epoch,
            gpu_numbers1Ba,
            sovits_g,
            sovits_d,
            if_grad_ckpt,
            lora_rank
        ):
            print(progress)
    else:
        raise ValueError("Unsupported version: %s" % version)
    
# 训练 GPT 模型
def train_gpt(version, batch_size, total_epoch, exp_name, if_dpo, if_save_latest, if_save_every_weights, save_every_epoch, gpu_numbers):
    if version in ["v1", "v2", "v3", "v4", "v2Pro", "v2ProPlus"]:
        set_version(version)
        _, gpt, _ = get_pretrained_paths(version)
        for progress, *_ in open1Bb(
            version,
            batch_size,
            total_epoch,
            exp_name,
            if_dpo,
            if_save_latest,
            if_save_every_weights,
            save_every_epoch,
            gpu_numbers,
            gpt,
        ):
            print(progress)
    else:
        raise ValueError("Unsupported version: %s" % version)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-SoVITS 命令行训练工具")
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    # 预处理数据集子命令
    preprocess_parser = subparsers.add_parser("preprocess", help="预处理数据集")
    preprocess_parser.add_argument("--version", type=str, default="v2ProPlus", help="模型版本 (v1, v2, v3, v4, v2Pro, v2ProPlus)")
    preprocess_parser.add_argument("-l","--data_list", type=str, required=True, help="数据列表文件路径，扩展名 .list，每行格式：完整音频路径|说话人名称|语言标签(ZH/EN/JA/KO/YUE)|文本内容")
    preprocess_parser.add_argument("-raw","--raw_wav_dir", type=str, default="", help="原始 WAV 文件目录，默认使用 list 中的完整路径")
    preprocess_parser.add_argument("-exp","--exp_name", type=str, required=True, help="实验名称")
    preprocess_parser.add_argument("--gpu_n_1a", type=str, default="0-0-0-0-0-0-0-0", help="用于预处理的 GPU 编号 (1a)，单卡多进程示例：0-0-0-0，多卡多进程示例：0-0-1-1")
    preprocess_parser.add_argument("--gpu_n_1Ba", type=str, default="0-0-0-0-0-0-0-0", help="用于预处理的 GPU 编号 (1Ba)，单卡多进程示例：0-0-0-0，多卡多进程示例：0-0-1-1")
    preprocess_parser.add_argument("--gpu_n_1c", type=str, default="0-0-0-0-0-0-0-0", help="用于预处理的 GPU 编号 (1c)，单卡多进程示例：0-0-0-0，多卡多进程示例：0-0-1-1")
    # 训练 SoVITS 模型子命令
    train_sovits_parser = subparsers.add_parser("train_sovits", help="训练 SoVITS 模型")
    train_sovits_parser.add_argument("--version", type=str, default="v2ProPlus", help="模型版本 (v1, v2, v3, v4, v2Pro, v2ProPlus)")
    train_sovits_parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    train_sovits_parser.add_argument("--total_epoch", type=int, default=10, help="总训练轮数")
    train_sovits_parser.add_argument("--exp_name", type=str, required=True, help="实验名称，要和预处理时的保持一致")
    train_sovits_parser.add_argument("--text_low_lr_rate", type=float, default=0.4, help="文本编码器学习率")
    train_sovits_parser.add_argument("--if_save_latest", action="store_false", help="是否保存最新模型，默认保存")
    train_sovits_parser.add_argument("--if_save_every_weights", action="store_false", help="是否保存每次权重，默认保存")
    train_sovits_parser.add_argument("--save_every_epoch", type=int, default=10, help="每隔多少轮保存一次权重")
    train_sovits_parser.add_argument("--gpu_numbers1Ba", type=str, default="0", help="用于训练的 GPU 编号 (1Ba)，单卡多进程示例：0-0-0-0，多卡多进程示例：0-0-1-1")
    train_sovits_parser.add_argument("--if_grad_ckpt", action="store_true", help="是否启用梯度检查点以节省显存，默认不启用")
    train_sovits_parser.add_argument("--lora_rank", type=int, default=32, help="LoRA 低秩矩阵分解的秩，仅v3/v4版本支持")
    # 训练 GPT 模型子命令
    train_gpt_parser = subparsers.add_parser("train_gpt", help="训练 GPT 模型")
    train_gpt_parser.add_argument("--version", type=str, default="v2ProPlus", help="模型版本 (v1, v2, v3, v4, v2Pro, v2ProPlus)")
    train_gpt_parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    train_gpt_parser.add_argument("--total_epoch", type=int, default=10, help="总训练轮数")
    train_gpt_parser.add_argument("--exp_name", type=str, required=True, help="实验名称，要和预处理时的保持一致")
    train_gpt_parser.add_argument("--if_dpo", action="store_true", help="是否使用 DPO 训练")
    train_gpt_parser.add_argument("--if_save_latest", action="store_false", help="是否保存最新模型，默认保存")
    train_gpt_parser.add_argument("--if_save_every_weights", action="store_false", help="是否保存每次权重，默认保存")
    train_gpt_parser.add_argument("--save_every_epoch", type=int, default=10, help="每隔多少轮保存一次权重")
    train_gpt_parser.add_argument("--gpu_numbers", type=str, default="0", help="用于训练的 GPU 编号，单卡多进程示例：0-0-0-0，多卡多进程示例：0-0-1-1")
    args = parser.parse_args()
    
    if args.command == "preprocess":
        preprocess_dataset(args.version, args.data_list, args.raw_wav_dir, args.exp_name, args.gpu_n_1a, args.gpu_n_1Ba, args.gpu_n_1c)
    elif args.command == "train_sovits":
        train_sovits(args.version, args.batch_size, args.total_epoch, args.exp_name, args.text_low_lr_rate, args.if_save_latest, args.if_save_every_weights, args.save_every_epoch, args.gpu_numbers1Ba, args.if_grad_ckpt, args.lora_rank)
    elif args.command == "train_gpt":
        train_gpt(args.version, args.batch_size, args.total_epoch, args.exp_name, args.if_dpo, args.if_save_latest, args.if_save_every_weights, args.save_every_epoch, args.gpu_numbers)
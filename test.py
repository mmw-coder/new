import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # 设置离线模式

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 现在默认使用本地文件，无需每次都指定参数
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

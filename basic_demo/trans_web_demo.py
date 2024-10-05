"""
This script creates an interactive web demo for the GLM-4-9B model using Gradio,
a Python library for building quick and easy UI components for machine learning models.
It's designed to showcase the capabilities of the GLM-4-9B model in a user-friendly interface,
allowing users to interact with the model through a chat-like interface.
"""

import os
import gradio as gr
import torch
from threading import Thread

from typing import Union
from pathlib import Path
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM     # 用于处理微调大型语言模型并实现高效的参数，高效微调的库
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer
)

# 变量类型可以是多种类型的一种
ModelType = Union[PreTrainedModel, PeftModelForCausalLM]    # 加载模型
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]     # 加载分词器模型

MODEL_PATH = os.environ.get('MODEL_PATH', '/home/zhaocb/GLM-4/basic_demo/model_glm4/glm-4-9b-chat')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)


# 将给定的路径字符串转换为绝对路径，处理用户目录并解析路径中的任何符号链接或相对部分。这样可以确保后续操作时使用的路径是准确和有效的。
def _resolve_path(path: Union[str, Path]) -> Path:
    # path创建一个路径对象，传入路径的字符串，可以使相对路径也可以是绝对路径
    # expanduser将路径中的用户目录展开为实际的主目录
    # resolve将路径解析为绝对路径
    return Path(path).expanduser().resolve()


# 加载模型和分析器模型，返回模型对象和分词器对象
def load_model_and_tokenizer(
        model_dir: Union[str, Path], trust_remote_code: bool = True
) -> tuple[ModelType, TokenizerType]:
    model_dir = _resolve_path(model_dir)
    # AutoPeftModelForCausalLM自动加载适用于因果语言模型的PRFT模型，加载预训练语言模型并进行微调，且无序手动设置复杂的模型配置
    if (model_dir / 'adapter_config.json').exists():
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code, device_map='auto'
        )
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code, device_map='auto'
        )
        tokenizer_dir = model_dir
    # 加载分词器模型
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=trust_remote_code, use_fast=False
    )
    return model, tokenizer


model, tokenizer = load_model_and_tokenizer(MODEL_PATH, trust_remote_code=True)


# 自定义停止标准，用于在生成文本时决定何时停止生成。用于检测是否达到了生成的结束条件，防止模型生成无穷无尽的文本，确保生成过程在适当的位置停止。
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = model.config.eos_token_id    # 获取模型配置中的结束token ID
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:     # 当模型输入的token ID和结束token ID相同，则停止生成文本
                return True
        return False


# 解析文本 将文本转化为一种HTML格式
def parse_text(text):
    lines = text.split("\n")    # 文本用换行分割，生成列表
    lines = [line for line in lines if line != ""]  # 去除空行
    count = 0   # 初始化计数器，用于计数代码块的开始和结束
    for i, line in enumerate(lines):    # 遍历每一行，检查是否包含代码快的标记  ```
        if "```" in line:   # 检查当前行是否包含代码块的标记
            count += 1  # 每一次找到代码块标记，计数器加1
            items = line.split('`') # 将行按照反引号分割，以便提取语言标识符
            # 根据count的奇偶决定是开始新的代码块还是结束当前代码块
            if count % 2 == 1:  # 如果是奇数，表示是代码块的开始，
                lines[i] = f'<pre><code class="language-{items[-1]}">'  # 替换改行内容为HTML的<pre><code>标签，并添加相应的语言类
            else:   # 如果是偶数，表示是代码块的结束
                lines[i] = f'<br></code></pre>'     # 替换该行内容为 </code></pre>
        else:   # 如果这一行中不包含代码块
            if i > 0:   # 只有在不是第一行的情况下才进行这些替换
                if count % 2 == 1:
                    # 将文本中的某些特殊字符替换为对应的HTML实体，每一行之前添加 <br> 标签，使其在 HTML 中显示为换行
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)   # 将处理后的行合并为一个字符串
    return text     # 返回最终的HTML格式字符串


#
def predict(history, max_length, top_p, temperature):
    stop = StopOnTokens()
    messages = []
    for idx, (user_msg, model_msg) in enumerate(history):
        if idx == len(history) - 1 and not model_msg:
            messages.append({"role": "user", "content": user_msg})
            break
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if model_msg:
            messages.append({"role": "assistant", "content": model_msg})

    model_inputs = tokenizer.apply_chat_template(messages,
                                                 add_generation_prompt=True,
                                                 tokenize=True,
                                                 return_tensors="pt").to(next(model.parameters()).device)
    streamer = TextIteratorStreamer(tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = {
        "input_ids": model_inputs,
        "streamer": streamer,
        "max_new_tokens": max_length,
        "do_sample": True,
        "top_p": top_p,
        "temperature": temperature,
        "stopping_criteria": StoppingCriteriaList([stop]),
        "repetition_penalty": 1.2,
        "eos_token_id": model.config.eos_token_id,
    }
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()
    for new_token in streamer:
        if new_token:
            history[-1][1] += new_token
        yield history


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">GLM-4-9B Gradio Simple Chat Demo</h1>""")
    chatbot = gr.Chatbot()

    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10, container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0.01, 1, value=0.6, step=0.01, label="Temperature", interactive=True)


    def user(query, history):
        return "", history + [[parse_text(query), ""]]


    submitBtn.click(user, [user_input, chatbot], [user_input, chatbot], queue=False).then(
        predict, [chatbot, max_length, top_p, temperature], chatbot
    )
    emptyBtn.click(lambda: None, None, chatbot, queue=False)

demo.queue()
demo.launch(server_name="0.0.0.0", server_port=8035, inbrowser=True, share=True)


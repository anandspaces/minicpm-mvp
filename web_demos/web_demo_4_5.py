#!/usr/bin/env python
# encoding: utf-8
"""
MiniCPM-V 4.5 Web Demo (openbmb/MiniCPM-V-4_5).
Same UX as web_demo.py but uses the 4.5 API: msgs with content as [image, text] or [text].
"""
# Shim for Gradio: huggingface_hub removed HfFolder; Gradio 4.x still imports it.
import huggingface_hub
if not hasattr(huggingface_hub, 'HfFolder'):
    class _HfFolder:
        @staticmethod
        def get_token():
            return getattr(huggingface_hub, 'get_token', lambda: None)()
        @staticmethod
        def save_token(token):
            getattr(huggingface_hub, 'login', lambda **_: None)(token=token)
        @staticmethod
        def delete_token():
            getattr(huggingface_hub, 'logout', lambda: None)()
    huggingface_hub.HfFolder = _HfFolder
if not hasattr(huggingface_hub, 'whoami'):
    def _whoami():
        try:
            return getattr(huggingface_hub, 'get_token', lambda: None)() and True or {}
        except Exception:
            return {}
    huggingface_hub.whoami = _whoami

import os
import gradio as gr
from PIL import Image
import traceback
import re
import torch
import argparse
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel

# Compatibility: MiniCPM-V 4.5 custom model uses _tied_weights_keys; newer transformers
# expects all_tied_weights_keys. Only patch when this method exists (newer transformers).
if hasattr(PreTrainedModel, '_adjust_tied_keys_with_tied_pointers'):
    _orig_adjust_tied = PreTrainedModel._adjust_tied_keys_with_tied_pointers

    def _patched_adjust_tied_keys(self, missing_keys):
        tied = getattr(self, 'all_tied_weights_keys', None) or getattr(self, '_tied_weights_keys', None)
        if tied is None or not isinstance(tied, dict):
            self.all_tied_weights_keys = {}
        elif not hasattr(self, 'all_tied_weights_keys'):
            self.all_tied_weights_keys = tied
        _orig_adjust_tied(self, missing_keys)

    PreTrainedModel._adjust_tied_keys_with_tied_pointers = _patched_adjust_tied_keys

# How to run:
# Nvidia GPUs with BF16:  python web_demo_4_5.py --device cuda --dtype bf16
# Nvidia GPUs FP16 only:  python web_demo_4_5.py --device cuda --dtype fp16
# Mac MPS:                 PYTORCH_ENABLE_MPS_FALLBACK=1 python web_demo_4_5.py --device mps --dtype fp16

parser = argparse.ArgumentParser(description='MiniCPM-V 4.5 demo')
parser.add_argument('--device', type=str, default='cuda', help='cuda, mps, or cpu')
parser.add_argument('--dtype', type=str, default='bf16', help='bf16 or fp16')
parser.add_argument('--model-path', type=str, default='openbmb/MiniCPM-V-4_5',
                    help='HuggingFace model id or local path (e.g. ./models/MiniCPM-V-4_5)')
args = parser.parse_args()
device = args.device
if device == 'cuda' and not torch.cuda.is_available():
    print('CUDA not available, using CPU (slow). Install NVIDIA Container Toolkit for GPU in Docker.')
    device = 'cpu'
assert device in ['cuda', 'mps', 'cpu']
if args.dtype == 'bf16':
    if device in ('mps', 'cpu'):
        dtype = torch.float16
        if device == 'mps':
            print('Warning: MPS does not support bf16, using fp16')
    else:
        dtype = torch.bfloat16
else:
    dtype = torch.float16

# Load MiniCPM-V 4.5: use --model-path with a local dir to avoid re-downloading from HF
model_path = args.model_path
local_only = os.path.isdir(model_path)
if local_only:
    print(f'Loading from local path: {model_path}')
else:
    print(f'Loading {model_path} (from HuggingFace if not cached)...')
kwargs = dict(
    trust_remote_code=True,
    attn_implementation='sdpa',
    torch_dtype=dtype,
)
if local_only:
    kwargs['local_files_only'] = True
model = AutoModel.from_pretrained(model_path, **kwargs)
model = model.to(device=device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=local_only)
print('Model loaded.')

ERROR_MSG = "Error, please retry"
model_name = 'MiniCPM-V 4.5'
DEFAULT_LANGUAGE_INSTRUCTION = "Please respond in English only."

# UI config (same as web_demo.py)
form_radio = {
    'choices': ['Beam Search', 'Sampling'],
    'value': 'Sampling',
    'interactive': True,
    'label': 'Decode Type',
}
num_beams_slider = {'minimum': 0, 'maximum': 5, 'value': 3, 'step': 1, 'interactive': True, 'label': 'Num Beams'}
repetition_penalty_slider = {'minimum': 0, 'maximum': 3, 'value': 1.2, 'step': 0.01, 'interactive': True, 'label': 'Repetition Penalty'}
repetition_penalty_slider2 = {'minimum': 0, 'maximum': 3, 'value': 1.05, 'step': 0.01, 'interactive': True, 'label': 'Repetition Penalty'}
max_new_tokens_slider = {'minimum': 1, 'maximum': 4096, 'value': 1024, 'step': 1, 'interactive': True, 'label': 'Max New Tokens'}
top_p_slider = {'minimum': 0, 'maximum': 1, 'value': 0.8, 'step': 0.05, 'interactive': True, 'label': 'Top P'}
top_k_slider = {'minimum': 0, 'maximum': 200, 'value': 100, 'step': 1, 'interactive': True, 'label': 'Top K'}
temperature_slider = {'minimum': 0, 'maximum': 2, 'value': 0.7, 'step': 0.05, 'interactive': True, 'label': 'Temperature'}


def create_component(params, comp='Slider'):
    if comp == 'Slider':
        return gr.Slider(
            minimum=params['minimum'], maximum=params['maximum'], value=params['value'],
            step=params['step'], interactive=params['interactive'], label=params['label'],
        )
    elif comp == 'Radio':
        return gr.Radio(
            choices=params['choices'], value=params['value'],
            interactive=params['interactive'], label=params['label'],
        )
    elif comp == 'Button':
        return gr.Button(value=params['value'], interactive=True)
    elif comp == 'Checkbox':
        return gr.Checkbox(value=params.get('value', False), label=params.get('label', ''))


def build_msgs_4_5(img, context):
    """Build message list for MiniCPM-V 4.5: content is [image, text] or [text]."""
    msgs = []
    for i, msg in enumerate(context):
        if msg['role'] == 'user':
            text = msg['content']
            if i == 0 and img is not None:
                msgs.append({'role': 'user', 'content': [img.convert('RGB'), text]})
            else:
                msgs.append({'role': 'user', 'content': [text]})
        else:
            msgs.append({'role': 'assistant', 'content': [msg['content']]})
    return msgs


def chat(img, msgs, enable_thinking, stream=False, **kwargs):
    """Call MiniCPM-V 4.5 chat. msgs: list of {role, content} (content = string)."""
    if img is None:
        return -1, "Error, invalid image, please upload a new image", None
    try:
        msgs_45 = build_msgs_4_5(img, msgs)
        # 4.5 API: no image=, msgs with content as list; enable_thinking, stream
        out = model.chat(
            msgs=msgs_45,
            tokenizer=tokenizer,
            enable_thinking=enable_thinking,
            stream=stream,
            **kwargs,
        )
        if stream and hasattr(out, '__iter__') and not isinstance(out, str):
            answer = ''.join(out)
        else:
            answer = out if isinstance(out, str) else str(out)
        answer = re.sub(r'(<box>.*</box>)', '', answer)
        answer = answer.replace('<ref>', '').replace('</ref>', '').replace('<box>', '').replace('</box>', '')
        return 0, answer, None
    except Exception as err:
        print(err)
        traceback.print_exc()
        return -1, ERROR_MSG, None


def upload_img(image, _chatbot, _app_session):
    image = Image.fromarray(image)
    _app_session['ctx'] = []
    _app_session['img'] = image
    if _chatbot is None:
        _chatbot = []
    _chatbot.append({'role': 'assistant', 'content': 'Image uploaded successfully, you can talk to me now'})
    return _chatbot, _app_session


def respond(_question, _chat_bot, _app_cfg, params_form, num_beams, repetition_penalty, repetition_penalty_2,
            top_p, top_k, temperature, enable_thinking):
    if _chat_bot is None:
        _chat_bot = []
    if _app_cfg.get('ctx') is None:
        _chat_bot.append({'role': 'assistant', 'content': 'Please upload an image to start'})
        return '', _chat_bot, _app_cfg

    _context = _app_cfg['ctx'].copy()
    if _context:
        _context.append({'role': 'user', 'content': _question})
    else:
        _context = [{'role': 'user', 'content': DEFAULT_LANGUAGE_INSTRUCTION + '\n\n' + _question}]
    print('<User>:', _question)

    if params_form == 'Beam Search':
        params = {'sampling': False, 'num_beams': num_beams, 'repetition_penalty': repetition_penalty, 'max_new_tokens': 896}
    else:
        params = {'sampling': True, 'top_p': top_p, 'top_k': top_k, 'temperature': temperature,
                  'repetition_penalty': repetition_penalty_2, 'max_new_tokens': 896}

    code, _answer, _ = chat(_app_cfg['img'], _context, enable_thinking, stream=False, **params)
    print('<Assistant>:', _answer)

    _context.append({'role': 'assistant', 'content': _answer})
    _chat_bot.append({'role': 'user', 'content': _question})
    _chat_bot.append({'role': 'assistant', 'content': _answer})
    if code == 0:
        _app_cfg['ctx'] = _context
    return '', _chat_bot, _app_cfg


def regenerate_button_clicked(_question, _chat_bot, _app_cfg, params_form, num_beams, repetition_penalty,
                               repetition_penalty_2, top_p, top_k, temperature, enable_thinking):
    if _chat_bot is None:
        _chat_bot = []
    if len(_chat_bot) <= 2:
        _chat_bot.append({'role': 'assistant', 'content': 'No question for regeneration.'})
        return '', _chat_bot, _app_cfg
    if _chat_bot[-1].get('content') == 'No question for regeneration.':
        return '', _chat_bot, _app_cfg
    # Last two messages are user + assistant; get the user question and remove both
    _question = _chat_bot[-2]['content'] if _chat_bot[-2].get('role') == 'user' else _chat_bot[-1]['content']
    _chat_bot = _chat_bot[:-2]
    _app_cfg['ctx'] = _app_cfg['ctx'][:-2]
    return respond(_question, _chat_bot, _app_cfg, params_form, num_beams, repetition_penalty,
                   repetition_penalty_2, top_p, top_k, temperature, enable_thinking)


with gr.Blocks(title=model_name) as demo:
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            params_form = create_component(form_radio, comp='Radio')
            enable_thinking = create_component({'value': False, 'label': 'Enable thinking mode (slower, deeper reasoning)'}, comp='Checkbox')
            with gr.Accordion('Beam Search'):
                num_beams = create_component(num_beams_slider)
                repetition_penalty = create_component(repetition_penalty_slider)
            with gr.Accordion('Sampling'):
                top_p = create_component(top_p_slider)
                top_k = create_component(top_k_slider)
                temperature = create_component(temperature_slider)
                repetition_penalty_2 = create_component(repetition_penalty_slider2)
            regenerate = create_component({'value': 'Regenerate'}, comp='Button')
        with gr.Column(scale=3, min_width=500):
            app_session = gr.State({'ctx': None, 'img': None})
            bt_pic = gr.Image(label='Upload an image to start')
            chat_bot = gr.Chatbot(label=f'Chat with {model_name}', type='messages')
            txt_message = gr.Textbox(label='Input text')

            inputs = [txt_message, chat_bot, app_session, params_form, num_beams, repetition_penalty,
                      repetition_penalty_2, top_p, top_k, temperature, enable_thinking]
            regenerate.click(regenerate_button_clicked, inputs, [txt_message, chat_bot, app_session])
            txt_message.submit(respond, inputs, [txt_message, chat_bot, app_session])
            bt_pic.upload(lambda: None, None, chat_bot, queue=False).then(
                upload_img, [bt_pic, chat_bot, app_session], [chat_bot, app_session],
            )

demo.launch(share=False, debug=True, server_port=8081, server_name='0.0.0.0')

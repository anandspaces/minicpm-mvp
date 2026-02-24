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
import logging
import gradio as gr
from PIL import Image
import traceback
import re
import torch
import argparse
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from decord import VideoReader, cpu

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
parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                    help='Logging level')
args = parser.parse_args()

logging.basicConfig(
    level=getattr(logging, args.log_level),
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger('web_demo_4_5')

device = args.device
if device == 'cuda' and not torch.cuda.is_available():
    log.warning('CUDA not available, using CPU (slow). Install NVIDIA Container Toolkit for GPU in Docker.')
    device = 'cpu'
log.info('Using device=%s', device)
assert device in ['cuda', 'mps', 'cpu']
if args.dtype == 'bf16':
    if device in ('mps', 'cpu'):
        dtype = torch.float16
        if device == 'mps':
            log.warning('MPS does not support bf16, using fp16')
    else:
        dtype = torch.bfloat16
else:
    dtype = torch.float16

# Load MiniCPM-V 4.5: use --model-path with a local dir to avoid re-downloading from HF
model_path = args.model_path
# Paths that look like filesystem paths (absolute or ./ ../) are treated as local
looks_local = (
    model_path.startswith('/')
    or model_path.startswith('./')
    or model_path.startswith('../')
)
local_only = looks_local and os.path.isdir(model_path)
if looks_local and not os.path.isdir(model_path):
    raise FileNotFoundError(
        f"Local model path does not exist: {model_path}\n"
        "Mount the model when running Docker, e.g.:\n"
        "  docker run -v /host/models/MiniCPM-V-4_5:/app/models/MiniCPM-V-4_5 ...\n"
        "Or use a HuggingFace model id: --model-path openbmb/MiniCPM-V-4_5"
    )
if local_only:
    log.info('Loading from local path: %s', model_path)
else:
    log.info('Loading %s (from HuggingFace if not cached)', model_path)
kwargs = dict(
    trust_remote_code=True,
    attn_implementation='sdpa',
    torch_dtype=dtype,
)
if local_only:
    kwargs['local_files_only'] = True
log.info('Loading model weights...')
model = AutoModel.from_pretrained(model_path, **kwargs)
log.info('Moving model to device=%s', device)
model = model.to(device=device).eval()
log.info('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=local_only)
log.info('Model loaded.')

ERROR_MSG = "Error, please retry"
model_name = 'MiniCPM-V 4.5'
DEFAULT_LANGUAGE_INSTRUCTION = "Please respond in English only."
MAX_NUM_FRAMES = 64

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


def _encode_image_for_frame(image):
    """Resize image/frame if too large (for video frames)."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image).convert('RGB')
    else:
        image = image.convert('RGB')
    max_size = 448 * 16
    if max(image.size) > max_size:
        w, h = image.size
        if w > h:
            new_w = max_size
            new_h = int(h * max_size / w)
        else:
            new_h = max_size
            new_w = int(w * max_size / h)
        image = image.resize((new_w, new_h), resample=Image.BICUBIC)
    return image


def encode_video(video_path):
    """Encode video file to list of PIL Images (frames). Gradio Video returns path string or (path, subtitle)."""
    if video_path is None:
        return None
    path = video_path
    if isinstance(video_path, (list, tuple)):
        path = video_path[0] if video_path else None
    if not path or not os.path.isfile(path):
        return None

    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    log.debug('Opening video: %s', path)
    vr = VideoReader(path, ctx=cpu(0))
    num_frames_total = len(vr)
    sample_fps = max(1, round(vr.get_avg_fps() / 1))
    frame_idx = [i for i in range(0, num_frames_total, sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    log.info('Decoding %d video frames (total in file: %d)', len(frame_idx), num_frames_total)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(f.astype('uint8')) for f in frames]
    frames = [_encode_image_for_frame(f) for f in frames]
    log.info('Encoded video: %d frames', len(frames))
    return frames


def build_msgs_4_5(img, video_frames, context):
    """Build message list for MiniCPM-V 4.5: content is [image(s), text] or [text]."""
    msgs = []
    for i, msg in enumerate(context):
        if msg['role'] == 'user':
            text = msg['content']
            if i == 0:
                if video_frames:
                    msgs.append({'role': 'user', 'content': video_frames + [text]})
                elif img is not None:
                    msgs.append({'role': 'user', 'content': [img.convert('RGB'), text]})
                else:
                    msgs.append({'role': 'user', 'content': [text]})
            else:
                msgs.append({'role': 'user', 'content': [text]})
        else:
            msgs.append({'role': 'assistant', 'content': [msg['content']]})
    return msgs


def _count_video_frames_in_context(context):
    """Count PIL images in first user message (for video params)."""
    for msg in context:
        if msg['role'] == 'user' and isinstance(msg.get('content'), list):
            n = sum(1 for c in msg['content'] if isinstance(c, Image.Image))
            if n > 0:
                return n
    return 0


def chat(img, video_frames, msgs, enable_thinking, stream=False, **kwargs):
    """Call MiniCPM-V 4.5 chat. msgs: list of {role, content} (content = string). Needs img or video_frames for first turn."""
    has_media = (img is not None) or (video_frames and len(video_frames) > 0)
    if not has_media:
        return -1, "Error, invalid image or video, please upload an image or video", None
    try:
        msgs_45 = build_msgs_4_5(img, video_frames, msgs)
        chat_kwargs = dict(
            msgs=msgs_45,
            tokenizer=tokenizer,
            enable_thinking=enable_thinking,
            stream=stream,
            **kwargs,
        )
        if video_frames and len(video_frames) > 0:
            num_frames = _count_video_frames_in_context(msgs_45) or len(video_frames)
            chat_kwargs["max_inp_length"] = 4352
            chat_kwargs["use_image_id"] = False
            chat_kwargs["max_slice_nums"] = 1 if num_frames > 16 else 2
        out = model.chat(**chat_kwargs)
        if stream and hasattr(out, '__iter__') and not isinstance(out, str):
            answer = ''.join(out)
        else:
            answer = out if isinstance(out, str) else str(out)
        answer = re.sub(r'(<box>.*</box>)', '', answer)
        answer = answer.replace('<ref>', '').replace('</ref>', '').replace('<box>', '').replace('</box>', '')
        return 0, answer, None
    except Exception as err:
        log.exception('chat() failed: %s', err)
        return -1, ERROR_MSG, None


def upload_img(image, _chatbot, _app_session):
    log.info('Upload: image')
    image = Image.fromarray(image)
    _app_session['ctx'] = []
    _app_session['img'] = image
    _app_session['video_frames'] = None
    if _chatbot is None:
        _chatbot = []
    _chatbot.append({'role': 'assistant', 'content': 'Image uploaded successfully, you can talk to me now'})
    return _chatbot, _app_session


def upload_video(video, _chatbot, _app_session):
    if video is None:
        return _chatbot, _app_session
    log.info('Upload: video, encoding...')
    frames = encode_video(video)
    if not frames:
        if _chatbot is None:
            _chatbot = []
        _chatbot.append({'role': 'assistant', 'content': 'Failed to encode video. Please try another file.'})
        return _chatbot, _app_session
    _app_session['ctx'] = []
    _app_session['img'] = None
    _app_session['video_frames'] = frames
    if _chatbot is None:
        _chatbot = []
    _chatbot.append({'role': 'assistant', 'content': 'Video uploaded and encoded; you can ask questions now.'})
    return _chatbot, _app_session


def respond(_question, _chat_bot, _app_cfg, params_form, num_beams, repetition_penalty, repetition_penalty_2,
            top_p, top_k, temperature, enable_thinking):
    if _chat_bot is None:
        _chat_bot = []
    has_media = _app_cfg.get('img') is not None or (_app_cfg.get('video_frames') and len(_app_cfg.get('video_frames', [])) > 0)
    if not has_media:
        _chat_bot.append({'role': 'assistant', 'content': 'Please upload an image or video to start'})
        return '', _chat_bot, _app_cfg
    if _app_cfg.get('ctx') is None:
        _app_cfg['ctx'] = []

    _context = _app_cfg['ctx'].copy()
    if _context:
        _context.append({'role': 'user', 'content': _question})
    else:
        _context = [{'role': 'user', 'content': DEFAULT_LANGUAGE_INSTRUCTION + '\n\n' + _question}]
    log.info('Request: user query (len=%d)', len(_question))

    if params_form == 'Beam Search':
        params = {'sampling': False, 'num_beams': num_beams, 'repetition_penalty': repetition_penalty, 'max_new_tokens': 896}
    else:
        params = {'sampling': True, 'top_p': top_p, 'top_k': top_k, 'temperature': temperature,
                  'repetition_penalty': repetition_penalty_2, 'max_new_tokens': 896}

    log.info('Running model.chat()...')
    code, _answer, _ = chat(_app_cfg.get('img'), _app_cfg.get('video_frames'), _context, enable_thinking, stream=False, **params)
    log.info('Response: code=%d, answer_len=%d', code, len(_answer) if _answer else 0)

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
            app_session = gr.State({'ctx': None, 'img': None, 'video_frames': None})
            bt_pic = gr.Image(label='Upload an image to start')
            bt_video = gr.Video(label='Upload or record a video', sources=['upload', 'webcam'])
            chat_bot = gr.Chatbot(label=f'Chat with {model_name}', type='messages', allow_tags=False)
            txt_message = gr.Textbox(label='Input text')

            inputs = [txt_message, chat_bot, app_session, params_form, num_beams, repetition_penalty,
                      repetition_penalty_2, top_p, top_k, temperature, enable_thinking]
            regenerate.click(regenerate_button_clicked, inputs, [txt_message, chat_bot, app_session])
            txt_message.submit(respond, inputs, [txt_message, chat_bot, app_session])
            bt_pic.upload(lambda: None, None, chat_bot, queue=False).then(
                upload_img, [bt_pic, chat_bot, app_session], [chat_bot, app_session],
            ).then(lambda: gr.update(value=None), None, bt_video)
            bt_video.upload(lambda: None, None, chat_bot, queue=False).then(
                upload_video, [bt_video, chat_bot, app_session], [chat_bot, app_session],
            ).then(lambda: gr.update(value=None), None, bt_pic)

log.info('Starting Gradio server on 0.0.0.0:8081')
demo.launch(share=False, debug=True, server_port=8081, server_name='0.0.0.0')

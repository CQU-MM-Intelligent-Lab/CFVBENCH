import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from moviepy.video.io.VideoFileClip import VideoFileClip
import base64
from io import BytesIO
from .._config import (
    MINICPM_MODEL_PATH,
    get_effective_visible_gpu_count,
    get_effective_visible_gpu_ids,
    get_transformers_device_map,
)

# Import the model path from the new central config file

torch.set_float32_matmul_precision('medium')

# Prefer fast image processor in Transformers
os.environ.setdefault("TRANSFORMERS_IMAGE_PROCESSING_USE_FAST", "1")

def encode_video(video, frame_times):
    frames = []
    duration = getattr(video, "duration", None)
    if duration is None:
        duration = 0.0
    max_t = max(0.0, float(duration) - 1e-3)

    # Clamp sampling times to valid range to avoid boundary decoding issues
    safe_times = frame_times
    try:
        if len(frame_times) > 0:
            safe_times = np.clip(frame_times, 0.0, max_t)
    except Exception:
        # If frame_times is not a numpy array, fall back to per-item clamp
        safe_times = [min(max(float(t), 0.0), max_t) for t in frame_times]

    for t in safe_times:
        t0 = float(t)
        try:
            frame = video.get_frame(t0)
        except Exception:
            # Nudge slightly earlier to stay within decodable range without reducing frame count
            t1 = max(0.0, min(max_t, t0 - 1e-2))
            frame = video.get_frame(t1)
        frames.append(frame)

    # Convert to RGB PIL images at target resolution
    frames = [Image.fromarray(np.asarray(v).astype('uint8')).convert('RGB').resize((1280, 720)) for v in frames]
    return frames
    
def _load_minicpm():
    """Prefer GPU (RTX 3080 Ti single card) and fall back to CPU only on OOM.
    - Primary: float16 + config-driven visible GPUs + device_map='auto'
    - Fallback (ONLY on CUDA OOM): float32 + CPU
    """
    model_path = MINICPM_MODEL_PATH
    if not model_path:
        raise RuntimeError(
            "MINICPM model path not provided. Set MINICPM_MODEL_PATH in "
            "videorag/_config.py or via the environment variable MINICPM_MODEL_PATH."
        )
    try:
        visible_gpu_count = get_effective_visible_gpu_count()
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if visible_gpu_count > 0 else torch.float32,
            "device_map": get_transformers_device_map(),
        }
        if visible_gpu_count > 0:
            print(
                f"[GPU] Loading MiniCPM with visible GPUs={','.join(get_effective_visible_gpu_ids())} "
                f"device_map={load_kwargs['device_map']}"
            )
        model = AutoModel.from_pretrained(
            model_path,
            **load_kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        return model, tokenizer
    except Exception as e:
        # Only fall back to CPU when it's an OOM error
        is_oom = False
        try:
            import torch
            is_oom = isinstance(e, torch.cuda.OutOfMemoryError)
        except Exception:
            pass
        text = str(e).lower()
        if (not is_oom) and ("out of memory" in text or "cuda oom" in text):
            is_oom = True
        if not is_oom:
            # Not an OOM -> respect policy: do not auto-CPU fallback
            raise
        # OOM fallback: CPU float32
        try:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map={'': 'cpu'}
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            model.eval()
            return model, tokenizer
        except Exception:
            # Surface the original OOM for clarity
            raise e
    

def segment_caption(video_name, video_path, segment_index2name, transcripts, segment_times_info, caption_result, error_queue):
    try:
        model, tokenizer = _load_minicpm()
        
        with VideoFileClip(video_path) as video:
            for index in tqdm(segment_index2name, desc=f"Captioning Video {video_name}"):
                frame_times = segment_times_info[index]["frame_times"]
                video_frames = encode_video(video, frame_times)
                segment_transcript = transcripts[index]
                query = f"The transcript of the current video:\n{segment_transcript}.\nNow provide a description (caption) of the video in English."
                # Pass images via dedicated parameter to avoid empty image batch parsing
                msgs = [{'role': 'user', 'content': query}]
                params = {
                    "use_image_id": False,
                    "max_slice_nums": 2,
                    "temperature": 0.2,
                    "do_sample": False,
                    "max_new_tokens": 256,
                }
                # Convert PIL frames to base64 (no data:image prefix) and prefer images_base64 API
                def _pil_list_to_base64(pil_list):
                    out = []
                    for im in pil_list:
                        try:
                            buf = BytesIO()
                            im.save(buf, format='JPEG')
                            b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                            out.append(b64)
                        except Exception:
                            continue
                    return out

                images_b64 = _pil_list_to_base64(video_frames)

                # Remove any mtmd* parameters to avoid mtmd warnings/behaviour
                cleaned_params = {k: v for k, v in params.items() if not (isinstance(k, str) and k.startswith('mtmd'))}

                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                    # Always send images as base64; if none available, send images_base64=None
                    seg_cap = model.chat(
                        msgs=msgs,
                        tokenizer=tokenizer,
                        images_base64=images_b64 if images_b64 else None,
                        **cleaned_params
                    )
                caption_result[index] = seg_cap.replace("\n", "").replace("<|endoftext|>", "")
                torch.cuda.empty_cache()
    except Exception as e:
        error_queue.put(f"Error in segment_caption:\n {str(e)}")
        raise RuntimeError


def merge_segment_information(segment_index2name, segment_times_info, transcripts, captions):
    inserting_segments = {}
    for index in segment_index2name:
        inserting_segments[index] = {"content": None, "time": None}
        segment_name = segment_index2name[index]
        inserting_segments[index]["time"] = '-'.join(segment_name.split('-')[-2:])
        inserting_segments[index]["content"] = f"Caption:\n{captions[index]}\nTranscript:\n{transcripts[index]}\n\n"
        inserting_segments[index]["transcript"] = transcripts[index]
        inserting_segments[index]["frame_times"] = segment_times_info[index]["frame_times"].tolist()
    return inserting_segments
        

def retrieved_segment_caption(caption_model, caption_tokenizer, refine_knowledge, retrieved_segments, video_path_db, video_segments, num_sampled_frames, extra_text_map=None):
    # caption_model / caption_tokenizer are provided by VideoRAG.load_caption_model()
    
    caption_result = {}
    for this_segment in tqdm(retrieved_segments, desc='Captioning Segments for Given Query'):
        video_name = '_'.join(this_segment.split('_')[:-1])
        index = this_segment.split('_')[-1]
        video_path = video_path_db._data[video_name]
        timestamp = video_segments._data[video_name][index]["time"].split('-')
        start, end = eval(timestamp[0]), eval(timestamp[1])
        video = VideoFileClip(video_path)
        # Guard against degenerate intervals by slightly nudging end and clamping later
        if end <= start:
            end = min(getattr(video, "duration", end) or end, start + 1e-3)
        frame_times = np.linspace(start, end, num_sampled_frames, endpoint=False)
        video_frames = encode_video(video, frame_times)
        if not isinstance(video_frames, list) or len(video_frames) == 0:
            raise RuntimeError(f"Empty frames for segment {this_segment} with time [{start}, {end}] from {video_path}")
        segment_transcript = video_segments._data[video_name][index]["transcript"]
        extra_text = ""
        if extra_text_map is not None and this_segment in extra_text_map and extra_text_map[this_segment].strip():
            extra_text = f"\nAdditional Visual Text/Objects:\n{extra_text_map[this_segment]}\n"
        # remove trailing stray quote and set stable generation params
        query = (
            f"The transcript of the current video:\n{segment_transcript}.{extra_text}\n"
            f"Now provide a very detailed description (caption) of the video in English and extract relevant information about: {refine_knowledge}."
        )
        # Pass images via dedicated parameter to avoid empty image batch parsing
        msgs = [{'role': 'user', 'content': query}]
        params = {
            "use_image_id": False,
            "max_slice_nums": 2,
            # remove unsupported sampling flags for this model interface
            "max_new_tokens": 256,
        }
        params = {
            "use_image_id": False,
            "max_slice_nums": 2,
            # remove unsupported sampling flags for this model interface
            "max_new_tokens": 256,
        }
        # Convert frames to base64 and call via images_base64 when possible
        def _pil_list_to_base64(pil_list):
            out = []
            for im in pil_list:
                try:
                    buf = BytesIO()
                    im.save(buf, format='JPEG')
                    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                    out.append(b64)
                except Exception:
                    continue
            return out

        images_b64 = _pil_list_to_base64(video_frames)
        cleaned_params = {k: v for k, v in params.items() if not (isinstance(k, str) and k.startswith('mtmd'))}
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            # Always send images as base64; if none available, send images_base64=None
            segment_caption = caption_model.chat(
                msgs=msgs,
                tokenizer=caption_tokenizer,
                images_base64=images_b64 if images_b64 else None,
                **cleaned_params
            )
        this_caption = segment_caption.replace("\n", "").replace("<|endoftext|>", "")
        caption_result[this_segment] = f"Caption:\n{this_caption}\nTranscript:\n{segment_transcript}\n\n"
        torch.cuda.empty_cache()
    
    return caption_result

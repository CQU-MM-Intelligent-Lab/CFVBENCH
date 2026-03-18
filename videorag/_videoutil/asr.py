import os
import torch
import logging
from tqdm import tqdm
from faster_whisper import WhisperModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def speech_to_text(video_name, working_dir, segment_index2name, audio_output_format):
    try:
        from videorag._config import WHISPER_MODEL_PATH, WHISPER_FALLBACK_MODEL_ID
    except Exception:
        WHISPER_MODEL_PATH = None
        WHISPER_FALLBACK_MODEL_ID = "distil-large-v3"
    model_path = WHISPER_MODEL_PATH or "./faster-distil-whisper-large-v3"
    if not os.path.exists(model_path):
        print(f"[ASR] Local model path not found in speech_to_text: {model_path}")
        model_path = WHISPER_FALLBACK_MODEL_ID
    model = WhisperModel(model_path)
    model.logger.setLevel(logging.WARNING)
    
    cache_path = os.path.join(working_dir, '_cache', video_name)
    
    transcripts = {}
    for index in tqdm(segment_index2name, desc=f"Speech Recognition {video_name}"):
        segment_name = segment_index2name[index]
        audio_file = os.path.join(cache_path, f"{segment_name}.{audio_output_format}")
        segments, info = model.transcribe(audio_file)
        result = ""
        for segment in segments:
            result += "[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text)
        transcripts[index] = result
    
    return transcripts

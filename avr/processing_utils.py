import os
import glob
import shutil
import subprocess
from typing import List, Tuple

import numpy as np
from PIL import Image

from faster_whisper import WhisperModel  # 保持与原文件一致
from test_media_utils import (
    get_video_duration,
    get_video_resolution,
)


def _frame_cache_dir_for_segment(video_path: str, output_dir: str, start_time: float, end_time: float) -> str:
    """
    为单个视频片段生成独立的帧缓存目录，避免不同片段互相污染。
    使用毫秒时间戳以保证唯一性。
    """
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    def _ms(t) -> str:
        try:
            return f"{int(round(float(t) * 1000))}"
        except Exception:
            return "na"
    return os.path.join(output_dir, f"{video_id}_{_ms(start_time)}_{_ms(end_time)}_frames")


def extract_frames_and_compress(
    video_path: str,
    output_dir: str,
    start_time: float = 0.0,
    end_time: float | None = None,
    num_frames: int = 5,
    target_height: int | None = 360,
    show_compress_log: bool = False,
    existing_frames: list[tuple[str, float]] | None = None,
) -> tuple[list[tuple[str, float]], float | None]:
    """
    Extracts frames, resizes them, and returns paths with timestamps.
    If existing_frames are provided, it calculates the new timestamps,
    extracts only the missing frames, and merges them with existing ones.
    """
    frame_dir = _frame_cache_dir_for_segment(video_path, output_dir, float(start_time), float(end_time))
    os.makedirs(frame_dir, exist_ok=True)

    if end_time is None:
        vid_dur = get_video_duration(video_path)
        start_time = 0.0 if start_time is None else float(start_time)
        end_time = min(30.0, vid_dur) if vid_dur else 30.0
    
    duration = float(end_time) - float(start_time)
    if duration <= 0:
        print(f"[Frames] Invalid duration ({duration}) for segment. Skipping frame extraction.")
        return [], None

    all_target_timestamps = np.linspace(start_time, end_time, num_frames, endpoint=False).tolist()
    
    # --- 新增逻辑：基于现有帧进行增量提取 ---
    if existing_frames:
        print(f"[Frames] Augmenting {len(existing_frames)} existing frames to {num_frames} total.")
        existing_timestamps = {round(ts, 5) for _, ts in existing_frames}
        
        # 找出需要新提取的帧的时间戳
        new_timestamps_to_extract = []
        for ts in all_target_timestamps:
            if round(ts, 5) not in existing_timestamps:
                new_timestamps_to_extract.append(ts)

        if new_timestamps_to_extract:
            print(f"[Frames] Extracting {len(new_timestamps_to_extract)} new frames...")
            # 使用 FFmpeg 的 select filter 精确提取指定时间戳的帧
            # 注意：时间戳需要相对于视频开头，而不是片段开头
            select_expr = "+".join([f'eq(n,{int(round((ts - start_time) * (num_frames/duration)))})' for ts in new_timestamps_to_extract])
            
            # 为了避免文件名冲突和排序问题，我们将新帧保存到临时目录
            temp_frame_dir = os.path.join(frame_dir, "temp_new_frames")
            os.makedirs(temp_frame_dir, exist_ok=True)

            # 使用更可靠的fps和select过滤器组合
            # 计算一个足够高的FPS以确保能捕捉到所有时间点
            required_fps = (num_frames / duration) * 2 
            
            # 生成 select 表达式，选择最接近目标时间戳的帧
            # 'select' filter is 1-based index, pts_time is in seconds
            select_filter_parts = []
            for ts in new_timestamps_to_extract:
                 # We need to find the frame whose presentation time is closest to ts
                 # This is complex with select, so we extract at high rate and then pick
                 pass # We will select frames after extraction

            vf = f"fps={required_fps}"
            if target_height and int(target_height) > 0:
                vf = vf + f",scale=-1:{int(target_height)}"
            cmd_extract = [
                "ffmpeg", "-y", "-ss", str(start_time), "-to", str(end_time), "-i", video_path,
                "-vf", vf,
                os.path.join(temp_frame_dir, "tempframe_%04d.jpg")
            ]
            try:
                from avr.test_media_utils import run_ffmpeg_with_fallback
                res = run_ffmpeg_with_fallback(cmd_extract, fallback_hwaccel=True, verbose=False)
                if res.returncode != 0:
                    raise subprocess.CalledProcessError(
                        res.returncode, cmd_extract, output=getattr(res, 'stdout', ''), stderr=getattr(res, 'stderr', '')
                    )

                # 从临时目录中挑选最接近目标时间戳的帧
                temp_frames = sorted(glob.glob(os.path.join(temp_frame_dir, "*.jpg")))
                temp_timestamps = np.linspace(start_time, end_time, len(temp_frames), endpoint=False)

                for ts_target in new_timestamps_to_extract:
                    # 找到时间最接近的已提取帧
                    best_match_idx = int(np.argmin(np.abs(temp_timestamps - ts_target)))
                    src_path = temp_frames[best_match_idx]
                    # 用目标时间戳重命名并移动到主帧目录
                    # 使用毫秒级精度命名以保证唯一性和排序
                    dest_filename = f"frame_{int(ts_target * 1000):08d}.jpg"
                    dest_path = os.path.join(frame_dir, dest_filename)
                    if not os.path.exists(dest_path):
                        # 使用复制而非移动，避免同一临时帧被多次选中后源文件缺失
                        try:
                            shutil.copy2(src_path, dest_path)
                        except Exception as copy_err:
                            # 若复制失败，尝试回退为移动（少数平台权限限制）
                            try:
                                os.replace(src_path, dest_path)
                            except Exception:
                                print(f"[Frames] Failed to materialize frame for ts={ts_target}: {copy_err}")

                # 清理临时目录
                shutil.rmtree(temp_frame_dir)
            except subprocess.CalledProcessError as e:
                print(f"[FFmpeg] Error during incremental frame extraction: {getattr(e, 'stderr', e)}")
            except Exception as e:
                print(f"[Frames] Unexpected error during incremental extraction: {e}")
        else:
            print("[Frames] No new frames needed.")

    # --- 如果没有现有帧，执行原始逻辑 ---
    else:
        # 检查是否已满足所需帧数
        existing_frame_files = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
        if len(existing_frame_files) == num_frames:
            print(f"[Frames] Frames already extracted for segment: {os.path.basename(frame_dir)}")
        else:
            # 清理旧帧，因为参数可能已变
            for f in existing_frame_files:
                os.remove(f)
            
            print(f"[Frames] Extracting {num_frames} frames for segment...")
            fps = num_frames / duration
            vf = f"fps={fps}"
            if target_height and int(target_height) > 0:
                vf = vf + f",scale=-1:{int(target_height)}"
            cmd = [
                "ffmpeg", "-y", "-ss", str(start_time), "-to", str(end_time), "-i", video_path,
                "-vf", vf,
                os.path.join(frame_dir, "frame_%04d.jpg")
            ]
            try:
                from avr.test_media_utils import run_ffmpeg_with_fallback
                res = run_ffmpeg_with_fallback(cmd, fallback_hwaccel=True, verbose=False)
                if res.returncode != 0:
                    print(f"[FFmpeg] Error during frame extraction: {getattr(res, 'stderr', '')}")
            except subprocess.CalledProcessError as e:
                print(f"[FFmpeg] Error during frame extraction: {getattr(e, 'stderr', e)}")
    
    final_frame_paths = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
    
    # 基于最终帧数重新计算精确时间戳
    final_timestamps = np.linspace(start_time, end_time, len(final_frame_paths), endpoint=False).tolist()
    frames_with_ts = list(zip(final_frame_paths, final_timestamps))

    # 计算像素压缩比
    original_res = get_video_resolution(video_path)
    ratio = None
    if original_res and final_frame_paths:
        with Image.open(final_frame_paths[0]) as img:
            new_res = img.size
        original_pixels = original_res[0] * original_res[1]
        new_pixels = new_res[0] * new_res[1]
        if original_pixels > 0:
            ratio = new_pixels / original_pixels

    return frames_with_ts, ratio


def transcribe_segment_audio(video_path: str, asr_model: WhisperModel, start_time: float, end_time: float) -> str:
    """Extracts the specific 30s audio segment and transcribes it using faster-whisper."""
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    seg_ms_start = int(round(float(start_time) * 1000))
    seg_ms_end = int(round(float(end_time) * 1000))
    audio_path = os.path.join(os.path.dirname(video_path), f"{video_id}_{seg_ms_start}_{seg_ms_end}.mp3")

    if not os.path.exists(audio_path):
        try:
            cmd = ["ffmpeg", "-y", "-ss", str(start_time), "-to", str(end_time), "-i", video_path, "-q:a", "0", "-map", "a", audio_path]
            print(f"[FFmpeg] Extracting audio segment {video_id} [{start_time}-{end_time}]...")
            from avr.test_media_utils import run_ffmpeg_with_fallback
            res = run_ffmpeg_with_fallback(cmd, fallback_hwaccel=True, verbose=False)
            if res.returncode != 0:
                print(f"[FFmpeg] Error extracting audio for segment {video_path}: {getattr(res, 'stderr', '')}")
                return ""
        except subprocess.CalledProcessError as e:
            print(f"[FFmpeg] Error extracting audio for segment {video_path}: {e.stderr.decode()}")
            return ""

    try:
        print(f"[ASR] Transcribing segment {video_id} [{start_time}-{end_time}]...")
        segments, _ = asr_model.transcribe(audio_path, beam_size=5)
        transcript = " ".join([seg.text for seg in segments]).strip()
        return transcript
    except Exception as e:
        print(f"[ASR] Error during transcription for {audio_path}: {e}")
        return ""

import os
import glob
import shutil
import subprocess
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
from PIL import Image

from avr.media_utils import (
    get_video_duration,
    get_video_resolution,
    run_ffmpeg_with_fallback,
)

if TYPE_CHECKING:
    from faster_whisper import WhisperModel


def _frame_cache_dir_for_segment(
    video_path: str,
    output_dir: str,
    start_time: float,
    end_time: float,
    target_height: int | None = None,
) -> str:
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
    height_tag = f"h{int(target_height)}" if target_height and int(target_height) > 0 else "horig"
    return os.path.join(output_dir, f"{video_id}_{_ms(start_time)}_{_ms(end_time)}_{height_tag}_frames")


def _extract_frames_via_cv2(
    video_path: str,
    frame_dir: str,
    timestamps: list[float],
    target_height: int | None = None,
) -> list[str]:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        print(f"[Frames] OpenCV fallback unavailable: {exc}")
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap or not cap.isOpened():
        print(f"[Frames] OpenCV could not open video for fallback extraction: {video_path}")
        return []

    saved_paths: list[str] = []
    try:
        for idx, ts in enumerate(timestamps):
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(ts)) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            if target_height and int(target_height) > 0 and img.height > 0:
                new_w = max(1, int(round(img.width * (int(target_height) / img.height))))
                img = img.resize((new_w, int(target_height)))
            out_path = os.path.join(frame_dir, f"frame_{idx + 1:04d}.jpg")
            img.save(out_path, format="JPEG")
            saved_paths.append(out_path)
    finally:
        try:
            cap.release()
        except Exception:
            pass
    return saved_paths


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
    frame_dir = _frame_cache_dir_for_segment(
        video_path,
        output_dir,
        float(start_time),
        float(end_time),
        target_height=target_height,
    )
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
                _extract_frames_via_cv2(video_path, frame_dir, new_timestamps_to_extract, target_height)
            except Exception as e:
                print(f"[Frames] Unexpected error during incremental extraction: {e}")
                _extract_frames_via_cv2(video_path, frame_dir, new_timestamps_to_extract, target_height)
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
                res = run_ffmpeg_with_fallback(cmd, fallback_hwaccel=True, verbose=False)
                if res.returncode != 0:
                    print(f"[FFmpeg] Error during frame extraction: {getattr(res, 'stderr', '')}")
                    _extract_frames_via_cv2(video_path, frame_dir, all_target_timestamps, target_height)
            except subprocess.CalledProcessError as e:
                print(f"[FFmpeg] Error during frame extraction: {getattr(e, 'stderr', e)}")
                _extract_frames_via_cv2(video_path, frame_dir, all_target_timestamps, target_height)
            except Exception as e:
                print(f"[Frames] Unexpected error during frame extraction: {e}")
                _extract_frames_via_cv2(video_path, frame_dir, all_target_timestamps, target_height)
    
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


def transcribe_segment_audio(
    video_path: str,
    asr_model: "WhisperModel",
    start_time: float,
    end_time: float,
    work_dir: str | None = None,
) -> str:
    """Extracts the specific 30s audio segment and transcribes it using faster-whisper."""
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    seg_ms_start = int(round(float(start_time) * 1000))
    seg_ms_end = int(round(float(end_time) * 1000))
    audio_root = work_dir or os.path.dirname(video_path)
    os.makedirs(audio_root, exist_ok=True)
    audio_path = os.path.join(audio_root, f"{video_id}_{seg_ms_start}_{seg_ms_end}.mp3")

    if not os.path.exists(audio_path):
        try:
            cmd = ["ffmpeg", "-y", "-ss", str(start_time), "-to", str(end_time), "-i", video_path, "-q:a", "0", "-map", "a", audio_path]
            print(f"[FFmpeg] Extracting audio segment {video_id} [{start_time}-{end_time}]...")
            res = run_ffmpeg_with_fallback(cmd, fallback_hwaccel=True, verbose=False)
            if res.returncode != 0:
                print(f"[FFmpeg] Error extracting audio for segment {video_path}: {getattr(res, 'stderr', '')}")
                return ""
        except subprocess.CalledProcessError as e:
            try:
                stderr = e.stderr.decode()
            except Exception:
                stderr = str(e)
            print(f"[FFmpeg] Error extracting audio for segment {video_path}: {stderr}")
            return ""
        except Exception as e:
            print(f"[FFmpeg] Unexpected audio extraction error for {video_path}: {e}")
            return ""

    try:
        print(f"[ASR] Transcribing segment {video_id} [{start_time}-{end_time}]...")
        segments, _ = asr_model.transcribe(audio_path, beam_size=5)
        transcript = " ".join([seg.text for seg in segments]).strip()
        return transcript
    except Exception as e:
        print(f"[ASR] Error during transcription for {audio_path}: {e}")
        return ""

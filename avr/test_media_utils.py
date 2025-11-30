import os
import subprocess
import urllib
import shutil
import glob
from PIL import Image
import numpy as np


def _make_completed_proc(returncode: int, stdout: str = "", stderr: str = ""):
    class _R:
        def __init__(self, rc, so, se):
            self.returncode = rc
            self.stdout = so
            self.stderr = se
    return _R(returncode, stdout, stderr)


def run_ffmpeg_with_fallback(cmd, fallback_hwaccel: bool = True, verbose: bool = True, timeout: int | None = None):
    """Run an ffmpeg command (cmd as list). On failure, attempt safe fallbacks:
    1) Try inserting a software AV1 decoder (libdav1d) for input if possible.
    2) If still failing, attempt a re-encode to H.264 (libx264) using libdav1d to decode.

    Returns an object with attributes: returncode, stdout, stderr (like subprocess.CompletedProcess).
    """
    import subprocess
    import shlex

    if isinstance(cmd, str):
        cmd_list = shlex.split(cmd)
    else:
        cmd_list = list(cmd)

    try:
        if verbose:
            print(f"[FFmpeg] Running: {' '.join(cmd_list)}")
        proc = subprocess.run(cmd_list, capture_output=True, text=True, timeout=timeout)
    except Exception as e:
        if verbose:
            print(f"[FFmpeg] Exception when running ffmpeg: {e}")
        return _make_completed_proc(1, stdout="", stderr=str(e))

    # success
    if proc.returncode == 0:
        return proc

    stderr = (proc.stderr or "")
    # If fallback not desired, return original result
    if not fallback_hwaccel:
        return proc

    # Heuristics: detect AV1 / hwaccel related failures
    low = stderr.lower()
    trigger = False
    av1_indicators = ["missing sequence header", "doesn't support hardware accelerated av1", "failed to get pixel format", "av1", "cuvid", "nvdec"]
    for k in av1_indicators:
        if k in low:
            trigger = True
            break

    if not trigger:
        # still try a soft fallback for common decoder issues
        trigger = True

    # attempt 1: insert software AV1 decoder (libdav1d) before -i if not present
    try:
        if "-i" in cmd_list:
            i_idx = cmd_list.index("-i")
            # only insert if no explicit input decoder set earlier
            pre_segment = cmd_list[:i_idx]
            if "-c:v" not in pre_segment and "-codec:v" not in pre_segment:
                new_cmd = pre_segment + ["-c:v", "libdav1d"] + cmd_list[i_idx:]
            else:
                new_cmd = cmd_list
        else:
            new_cmd = cmd_list

        if verbose:
            print(f"[FFmpeg] Fallback attempt 1 (software AV1 decode): {' '.join(new_cmd)}")
        proc2 = subprocess.run(new_cmd, capture_output=True, text=True, timeout=timeout)
        if proc2.returncode == 0:
            return proc2
    except Exception as e:
        if verbose:
            print(f"[FFmpeg] Fallback attempt 1 exception: {e}")

    # attempt 2: full re-encode to H.264 (libx264) using libdav1d decode.
    # Parse input and output from original command (simple heuristic: -i <input> and last token as output)
    try:
        inp = None
        out = None
        if "-i" in cmd_list:
            idx = cmd_list.index("-i")
            if idx + 1 < len(cmd_list):
                inp = cmd_list[idx + 1]
        if len(cmd_list) >= 1:
            out = cmd_list[-1]

        if inp and out:
            reenc_cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-c:v", "libdav1d", "-i", inp,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                out,
            ]
            if verbose:
                print(f"[FFmpeg] Fallback attempt 2 (re-encode to H.264): {' '.join(reenc_cmd)}")
            proc3 = subprocess.run(reenc_cmd, capture_output=True, text=True, timeout=timeout)
            return proc3
    except Exception as e:
        if verbose:
            print(f"[FFmpeg] Fallback attempt 2 exception: {e}")

    # all attempts failed; return original proc
    return proc


def get_bench_video_path(video_id: str) -> str | None:
    """
    Locate the video file in the ./Bench directory (relative to repo root).
    Supports flexible extensions.
    """
    # Assuming this file is in avr/, so repo root is one level up
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bench_dir = os.path.join(repo_root, "Bench")
    
    # Search for video_id.* in Bench
    # video_id might be a path or just an ID. We assume ID.
    # If video_id contains path separators, take basename
    vid = os.path.basename(video_id)
    # Remove extension if present (though usually video_id shouldn't have it)
    if '.' in vid:
        vid = os.path.splitext(vid)[0]
        
    pattern = os.path.join(bench_dir, f"{vid}.*")
    candidates = glob.glob(pattern)
    
    # Filter for video extensions if needed, or just take the first file that looks like a video
    video_exts = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv'}
    for c in candidates:
        if os.path.splitext(c)[1].lower() in video_exts:
            return c
            
    return None


def download_file(url: str, target_dir: str) -> str:
    """
    Resolves a video file. 
    First checks ./Bench for a local file matching the ID (derived from url).
    If not found, falls back to downloading (legacy behavior).
    """
    import urllib.request
    import urllib.parse
    
    # Try to extract video ID from URL or use URL as ID
    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme:
        # Treat as ID
        video_id = url
    else:
        # Try to get ID from filename in URL
        filename = os.path.basename(parsed.path)
        video_id = os.path.splitext(filename)[0]
        
    # Check local Bench
    local_path = get_bench_video_path(video_id)
    if local_path and os.path.exists(local_path):
        print(f"[Video] Found local video in Bench: {local_path}")
        return local_path

    # Fallback to original download logic
    filename = os.path.basename(urllib.parse.unquote(url).split('?')[0])
    filepath = os.path.join(target_dir, filename)
    if os.path.exists(filepath):
        print(f"[Download] File already exists: {filepath}")
        return filepath
    print(f"[Download] Downloading {url} to {filepath}")
    try:
        urllib.request.urlretrieve(url, filepath)
        return filepath
    except Exception as e:
        print(f"[Download] Error downloading {url}: {e}")
        return ""


def get_video_resolution(video_path: str) -> tuple[int, int] | None:
    """Gets video resolution using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        width, height = map(int, result.stdout.strip().split('x'))
        return width, height
    except Exception as e:
        print(f"[FFprobe] Error getting resolution for {video_path}: {e}")
        return None


def get_video_duration(video_path: str) -> float | None:
    """Gets video duration in seconds using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"[FFprobe] Error getting duration for {video_path}: {e}")
        return None


def is_valid_video(video_path: str) -> bool:
    dur = get_video_duration(video_path)
    return dur is not None and dur > 0


def repair_mp4_faststart(src_path: str) -> str | None:
    """Try to repair an MP4 by remuxing with faststart. Returns repaired path or None."""
    try:
        if not os.path.exists(src_path):
            return None
        repaired = os.path.splitext(src_path)[0] + "_fixed.mp4"
        cmd = [
            "ffmpeg", "-y", "-i", src_path,
            "-c", "copy", "-movflags", "+faststart",
            repaired
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return repaired if os.path.exists(repaired) else None
    except Exception as e:
        print(f"[FFmpeg] Repair failed for {src_path}: {e}")
        return None


def ensure_valid_video_or_skip(url: str, work_dir: str, video_path: str) -> str | None:
    """Ensure the MP4 at video_path is valid. If invalid, force redownload; if still invalid, try faststart repair. Returns a valid path or None to skip."""
    if is_valid_video(video_path):
        return video_path
    
    # If it's a local Bench file, we can't "redownload" it easily, but we can try to repair it.
    # Check if it is in Bench
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bench_dir = os.path.join(repo_root, "Bench")
    is_bench_file = False
    try:
        is_bench_file = os.path.commonprefix([os.path.abspath(video_path), os.path.abspath(bench_dir)]) == os.path.abspath(bench_dir)
    except Exception:
        pass

    if is_bench_file:
        print(f"[Validate] Local Bench file invalid: {video_path}. Attempting repair...")
        # Repair to work_dir, not modifying Bench
        filename = os.path.basename(video_path)
        repaired_path = os.path.join(work_dir, f"repaired_{filename}")
        
        # Copy to work_dir first
        try:
            shutil.copy2(video_path, repaired_path)
            repaired = repair_mp4_faststart(repaired_path)
            if repaired and is_valid_video(repaired):
                return repaired
        except Exception as e:
            print(f"[Validate] Repair failed: {e}")
        return None

    print(f"[Validate] Detected invalid video file: {video_path}. Forcing re-download...")
    try:
        if os.path.exists(video_path):
            os.remove(video_path)
    except Exception:
        pass
    new_path = download_file(url, work_dir)
    if new_path and is_valid_video(new_path):
        return new_path
    print(f"[Validate] Re-download did not produce a valid MP4. Attempting repair...")
    repaired = repair_mp4_faststart(new_path or video_path)
    if repaired and is_valid_video(repaired):
        # Replace original path reference with repaired file
        try:
            shutil.move(repaired, new_path or video_path)
            final_path = new_path or video_path
        except Exception:
            final_path = repaired
        print(f"[Validate] Repair succeeded: {final_path}")
        return final_path if is_valid_video(final_path) else None
    print(f"[Validate] Repair failed. Skipping this segment: {url}")
    return None


def image_compression_ratio(video_path: str, frame_paths: list[str]) -> float | None:
    original_res = get_video_resolution(video_path)
    if not original_res or not frame_paths:
        return None
    with Image.open(frame_paths[0]) as img:
        new_res = img.size
    original_pixels = original_res[0] * original_res[1]
    new_pixels = new_res[0] * new_res[1]
    if original_pixels > 0:
        return new_pixels / original_pixels
    return None

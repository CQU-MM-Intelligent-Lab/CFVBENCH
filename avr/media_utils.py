import os
import subprocess
import urllib
import shutil
import glob
from PIL import Image
import numpy as np


def resolve_external_binary(name: str) -> str | None:
    """Resolve an external executable path, with a best-effort ffmpeg fallback."""
    system_path = shutil.which(name)
    if system_path:
        return system_path

    if name == "ffmpeg":
        try:
            import imageio_ffmpeg  # type: ignore

            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            if ffmpeg_path and os.path.exists(ffmpeg_path):
                return ffmpeg_path
        except Exception:
            return None

    if name == "ffprobe":
        ffmpeg_path = resolve_external_binary("ffmpeg")
        if ffmpeg_path:
            ext = ".exe" if os.name == "nt" else ""
            sibling = os.path.join(os.path.dirname(ffmpeg_path), f"ffprobe{ext}")
            if os.path.exists(sibling):
                return sibling

    return None


def resolve_ffmpeg_binary() -> str | None:
    return resolve_external_binary("ffmpeg")


def resolve_ffprobe_binary() -> str | None:
    return resolve_external_binary("ffprobe")


def _replace_binary(cmd_list: list[str], binary_name: str, resolved_path: str) -> list[str]:
    if not cmd_list:
        return cmd_list

    first = os.path.basename(str(cmd_list[0])).lower()
    expected = {binary_name.lower(), f"{binary_name.lower()}.exe"}
    if first in expected:
        new_cmd = list(cmd_list)
        new_cmd[0] = resolved_path
        return new_cmd
    return cmd_list


def _get_video_metadata_via_cv2(video_path: str) -> tuple[tuple[int, int] | None, float | None]:
    """Best-effort OpenCV fallback when ffprobe is unavailable."""
    try:
        import cv2  # type: ignore
    except Exception:
        return None, None

    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap or not cap.isOpened():
            return None, None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)

        resolution = (width, height) if width > 0 and height > 0 else None
        duration = (frame_count / fps) if fps > 0 and frame_count > 0 else None
        return resolution, duration
    except Exception:
        return None, None
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass


def _make_completed_proc(returncode: int, stdout: str = "", stderr: str = ""):
    # small shim to return an object similar to subprocess.CompletedProcess for callers
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

    ffmpeg_bin = resolve_ffmpeg_binary()
    if ffmpeg_bin is None:
        return _make_completed_proc(
            1,
            stdout="",
            stderr=(
                "ffmpeg executable not found. Install the system ffmpeg binary or make it available on PATH. "
                "The Python package `ffmpeg` is not the executable used by this pipeline."
            ),
        )
    cmd_list = _replace_binary(cmd_list, "ffmpeg", ffmpeg_bin)

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
                ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
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


def download_file(url: str, target_dir: str) -> str:
    """Downloads a file from a URL to a target directory."""
    import urllib.request
    import urllib.parse
    local_candidate = os.path.abspath(os.path.expanduser(str(url)))
    if os.path.isfile(local_candidate):
        print(f"[Download] Using local file: {local_candidate}")
        return local_candidate
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
        ffprobe_bin = resolve_ffprobe_binary()
        if ffprobe_bin is None:
            resolution, _ = _get_video_metadata_via_cv2(video_path)
            if resolution:
                print(f"[FFprobe] ffprobe binary unavailable for {video_path}, falling back to OpenCV metadata.")
                return resolution
            print(f"[FFprobe] ffprobe binary unavailable for {video_path}, and OpenCV fallback could not read metadata.")
            return None

        cmd = [
            ffprobe_bin, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        width, height = map(int, result.stdout.strip().split('x'))
        return width, height
    except Exception as e:
        resolution, _ = _get_video_metadata_via_cv2(video_path)
        if resolution:
            print(f"[FFprobe] Resolution probe failed for {video_path}, falling back to OpenCV metadata.")
            return resolution
        print(f"[FFprobe] Error getting resolution for {video_path}: {e}")
        return None


def get_video_duration(video_path: str) -> float | None:
    """Gets video duration in seconds using ffprobe."""
    try:
        ffprobe_bin = resolve_ffprobe_binary()
        if ffprobe_bin is None:
            _, duration = _get_video_metadata_via_cv2(video_path)
            if duration and duration > 0:
                print(f"[FFprobe] ffprobe binary unavailable for {video_path}, falling back to OpenCV metadata.")
                return duration
            print(f"[FFprobe] ffprobe binary unavailable for {video_path}, and OpenCV fallback could not read metadata.")
            return None

        cmd = [
            ffprobe_bin, "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        _, duration = _get_video_metadata_via_cv2(video_path)
        if duration and duration > 0:
            print(f"[FFprobe] Duration probe failed for {video_path}, falling back to OpenCV metadata.")
            return duration
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
        ffmpeg_bin = resolve_ffmpeg_binary()
        if ffmpeg_bin is None:
            print(f"[FFmpeg] Repair skipped for {src_path}: ffmpeg executable not available.")
            return None
        repaired = os.path.splitext(src_path)[0] + "_fixed.mp4"
        cmd = [
            ffmpeg_bin, "-y", "-i", src_path,
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

    local_source = os.path.abspath(os.path.expanduser(str(url)))
    real_video_path = os.path.realpath(video_path)
    real_workdir = os.path.realpath(work_dir)
    is_local_source = os.path.isfile(local_source)
    same_as_local_source = False
    try:
        same_as_local_source = is_local_source and os.path.samefile(local_source, video_path)
    except Exception:
        same_as_local_source = False
    video_is_in_workdir = False
    try:
        video_is_in_workdir = os.path.commonpath([real_video_path, real_workdir]) == real_workdir
    except Exception:
        video_is_in_workdir = False

    # Never delete original local dataset files outside the workdir. For local sources,
    # try repair in place (or on the current file) rather than pretending the path is a URL.
    if same_as_local_source:
        print(f"[Validate] Detected invalid local source video: {video_path}. Trying repair instead of re-download.")
        repaired = repair_mp4_faststart(video_path)
        if repaired and is_valid_video(repaired):
            print(f"[Validate] Repair succeeded: {repaired}")
            return repaired
        print(f"[Validate] Local source repair failed. Skipping this segment: {video_path}")
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

"""Simple diagnostic script to help debug segment -> video resolution and basic ffmpeg frame extraction.

Usage examples (from repo root, Windows cmd):
    python test\diagnose_segment.py q2KehxKQ3Zk_0_4 --print-map
    python test\diagnose_segment.py R7N5a476DKQ_3 --ffmpeg-check

What it does:
- Parse the segment id into (video_name, start, end) using existing logic where possible.
- Try conservative key resolution against video_urls.video_urls_multi_segment (exact, .mp4, strip suffixes, FULL variants).
- Print attempted candidates and the matched key/url (if any).
- If matched url is a local file path and --ffmpeg-check is passed, try to extract one frame using ffmpeg to a temp jpg and report success.

This script is safe: it will not download remote URLs. For remote URLs it reports the matched URL and suggests next steps.
"""

import argparse
import os
import subprocess
import tempfile
import sys
from typing import Tuple, List

try:
    from video_urls import video_urls_multi_segment as VMAP
except Exception:
    VMAP = {}

def _parse_segment_timing(segment_id: str) -> Tuple[str, int, int]:
    try:
        if '_' in segment_id and segment_id.count('_') >= 2:
            parts = segment_id.split('_')
            if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
                base = '_'.join(parts[:-2])
                three_idx = int(parts[-2])
                thirty_idx = int(parts[-1])
                if base.endswith('FULL'):
                    video_name = base[:-4]
                    start_time = three_idx * 180 + thirty_idx * 30
                else:
                    video_name = f"{base}_{three_idx}"
                    start_time = thirty_idx * 30
                return video_name, start_time, start_time + 30
    except Exception:
        pass
    try:
        video_name, idx = segment_id.rsplit('_', 1)
        start_time = int(idx) * 30
        return video_name, start_time, start_time + 30
    except Exception:
        return segment_id, 0, 30


def resolve_video_candidates(vmap: dict, video_name: str) -> Tuple[str, str, List[str]]:
    """Try conservative candidates; return (matched_key, matched_value, tried_list)"""
    tried = []
    candidates = [video_name]
    if not video_name.endswith('.mp4'):
        candidates.append(video_name + '.mp4')
    else:
        candidates.append(video_name[:-4])
    parts = video_name.split('_') if '_' in video_name else [video_name]
    for i in range(1, len(parts)):
        prefix = '_'.join(parts[:-i])
        if prefix:
            candidates.append(prefix)
            candidates.append(prefix + '.mp4')
            candidates.append(prefix + 'FULL')
            candidates.append(prefix + 'FULL.mp4')
    # dedupe preserving order
    seen = set(); uniq = []
    for c in candidates:
        if c not in seen:
            seen.add(c); uniq.append(c)
    for c in uniq:
        tried.append(c)
        try:
            if c in vmap and vmap.get(c):
                return c, vmap.get(c), tried
        except Exception:
            continue
    # fallback: first available
    try:
        for k, v in vmap.items():
            if v:
                tried.append(f"<fallback:{k}>")
                return k, v, tried
    except Exception:
        pass
    return None, None, tried


def ffmpeg_extract_one_frame(video_path: str, t: float) -> Tuple[bool, str]:
    """Attempt to extract one frame at time t (seconds) to a temp jpg. Returns (success, message).
    Requires ffmpeg available in PATH. Only run for local files.
    """
    if not os.path.exists(video_path):
        return False, f"File not found: {video_path}"
    out_fd, out_path = tempfile.mkstemp(suffix='.jpg')
    os.close(out_fd)
    cmd = [
        'ffmpeg', '-y', '-ss', str(float(t)), '-i', video_path,
        '-frames:v', '1', '-q:v', '2', out_path
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        if proc.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            os.remove(out_path)
            return True, 'frame extraction OK'
        else:
            stderr = proc.stderr.decode('utf-8', errors='ignore')
            try:
                os.remove(out_path)
            except Exception:
                pass
            return False, f'ffmpeg failed (rc={proc.returncode}): ' + (stderr.splitlines()[-1] if stderr else '')
    except FileNotFoundError:
        try:
            os.remove(out_path)
        except Exception:
            pass
        return False, 'ffmpeg not found in PATH'
    except Exception as e:
        try:
            os.remove(out_path)
        except Exception:
            pass
        return False, f'ffmpeg exception: {e}'


def main():
    p = argparse.ArgumentParser(description='Diagnose segment -> video resolution and basic ffmpeg frame extraction')
    p.add_argument('segment_id', help='Segment id to diagnose, e.g. q2KehxKQ3Zk_0_4 or R7N5a476DKQ_3')
    p.add_argument('--ffmpeg-check', action='store_true', help='If matched video is local file, try to extract one frame')
    p.add_argument('--print-map', action='store_true', help='Print the video_urls entry (if present)')
    p.add_argument('--download', action='store_true', help='If matched value is remote, attempt to download and validate to ./downloads/')
    args = p.parse_args()

    seg = args.segment_id
    print(f'Parsing segment id: {seg}')
    vname, st, et = _parse_segment_timing(seg)
    print(f'  -> parsed video_name="{vname}", start={st}, end={et}')

    print('\nLooking up in video_urls mapping (video_urls.video_urls_multi_segment) ...')
    if not isinstance(VMAP, dict) or not VMAP:
        print('  WARNING: video_urls_multi_segment appears empty or not importable in this environment.')
    matched_key, matched_val, tried = resolve_video_candidates(VMAP or {}, vname)
    print('  Tried candidates:')
    for t in tried:
        print('   -', t)
    if matched_key:
        print(f'  MATCH: key="{matched_key}" -> value="{matched_val}"')
    else:
        print('  No match found in video_urls mapping.')

    if args.print_map:
        print('\nFull mapping for base id (if exists):')
        try:
            base = vname.split('_')[0]
            print('  base =', base)
            print('  entry =', VMAP.get(base))
        except Exception:
            pass

    if args.ffmpeg_check:
        if not matched_val:
            print('\nNo matched value to ffmpeg-check.')
            return
        # if matched_val looks like a local path
        if isinstance(matched_val, str) and os.path.exists(matched_val):
            mid = (st + et) / 2.0
            ok, msg = ffmpeg_extract_one_frame(matched_val, mid)
            print('\nFFMPEG CHECK:')
            print('  target file:', matched_val)
            print('  time (s):', mid)
            print('  result:', ok, msg)
        else:
            print('\nMatched value is not a local file (likely a remote URL).')
            print('  value =', matched_val)
            print('  Suggestion: ensure the video has been downloaded and video_path_db contains local path. You can run the pipeline to download or manually download the URL and re-run ffmpeg check against the local file.')

    if args.download:
        print('\n--download requested: attempting to download matched value (if remote)')
        try:
            from avr.test_media_utils import download_file, ensure_valid_video_or_skip
        except Exception as e:
            print('  ERROR: download helpers not available:', e)
            return

        downloads_dir = os.path.join(os.getcwd(), 'downloads')
        os.makedirs(downloads_dir, exist_ok=True)

        def try_download_url(url: str):
            try:
                print('  Downloading URL:', url)
                cand = download_file(url, downloads_dir)
                if not cand:
                    print('   -> download_file returned empty')
                    return None
                cand2 = ensure_valid_video_or_skip(url, downloads_dir, cand)
                if cand2:
                    print('   -> validated and available at:', cand2)
                    return cand2
                print('   -> validation/repair failed for:', cand)
            except Exception as e:
                print('   -> download exception:', e)
            return None

        downloaded_path = None
        # If matched_val is an inner-map, attempt to pick a candidate
        if isinstance(matched_val, dict):
            inner_map = matched_val
            # build candidates similar to resolver
            parts = vname.split('_') if '_' in vname else [vname]
            inner_candidates = []
            if len(parts) >= 2 and parts[-1].isdigit():
                base = '_'.join(parts[:-1])
                try:
                    three_idx = int(parts[-1])
                except Exception:
                    three_idx = None
                if base and three_idx is not None:
                    inner_candidates.extend([
                        f"{base}_{three_idx}.mp4",
                        f"{base}_{three_idx}",
                    ])
                    inner_candidates.append(base + '.mp4')
                    inner_candidates.append(base)
            else:
                base = vname
                inner_candidates.extend([base + '.mp4', base])

            # try ordered candidates
            for ic in inner_candidates:
                if ic in inner_map and inner_map.get(ic):
                    val = inner_map.get(ic)
                    if isinstance(val, str) and os.path.exists(val):
                        downloaded_path = val; break
                    if isinstance(val, str) and val.startswith(('http://','https://')):
                        downloaded_path = try_download_url(val)
                        if downloaded_path:
                            break

            # fallback: try first available inner_map entry
            if not downloaded_path:
                for k, v in inner_map.items():
                    if isinstance(v, str) and v.startswith(('http://','https://')):
                        downloaded_path = try_download_url(v)
                        if downloaded_path:
                            break
        elif isinstance(matched_val, str):
            if os.path.exists(matched_val):
                downloaded_path = matched_val
            elif matched_val.startswith(('http://','https://')):
                downloaded_path = try_download_url(matched_val)

        if downloaded_path:
            print('Download result: OK ->', downloaded_path)
            # optionally run ffmpeg check if requested together
            if args.ffmpeg_check:
                mid = (st + et) / 2.0
                ok, msg = ffmpeg_extract_one_frame(downloaded_path, mid)
                print('\nFFMPEG CHECK ON DOWNLOADED FILE:')
                print('  target file:', downloaded_path)
                print('  time (s):', mid)
                print('  result:', ok, msg)
        else:
            print('Download result: FAILED or no suitable remote URL found to download.')

if __name__ == '__main__':
    main()

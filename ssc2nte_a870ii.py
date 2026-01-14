#!/usr/bin/env python3
"""
ssc2nte_a870ii.py (experimental)
Convert a StepMania .ssc chart into a FlyRanking A870II-style .nte file.

What we know from real samples:
- The .nte begins with a "step stream" where EACH byte is repeated 3 times (triplicated).
- After the step stream, there is a list of 32-bit little-endian records that look like pairs of uint16:
    (duration_ms, flag)
  where duration_ms is usually 20 or 21, and flag is either 0x0000 or 0x0101.
- The file ends with two uint32:
    0x0000FFFF,  (step_stream_bytes_len - 6)

This tool writes a compatible *structure* for A870II based on those observations.
Because FlyRanking never published the spec, the timing model here is a best-effort guess:
- We generate a fixed "frame clock" of ~48 Hz using 20/21 ms durations (same style as sample).
- We quantize notes onto that clock.

You may need to tweak:
- --fps (default 48)
- --offset-ms (song/chart offset)
- --chart (which chart to use)
"""

from __future__ import annotations
import argparse
import pathlib
import re
import struct
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

LANE_MASK = {
    "L": 0x01,
    "D": 0x02,
    "U": 0x04,
    "R": 0x08,
    # Some FlyRanking pads have an extra "center" sensor. StepMania doesn't.
    # We'll leave bit 0x10 unused unless you remap.
}

@dataclass
class SSCChart:
    title: str
    bpms: List[Tuple[float, float]]  # (beat, bpm)
    offset: float                    # seconds (StepMania offset: positive means chart starts *later*)
    notes: List[Tuple[float, int]]   # (beat, lane_mask)

SSC_KV_RE = re.compile(r'^\s*#([A-Z0-9_]+):(.+?);\s*$', re.MULTILINE | re.DOTALL)

def parse_kv(text: str) -> Dict[str, str]:
    out = {}
    for m in SSC_KV_RE.finditer(text):
        key = m.group(1).strip()
        val = m.group(2).strip()
        out[key] = val
    return out

def parse_bpms(val: str) -> List[Tuple[float, float]]:
    # "0.000=256.000,64.000=128.000"
    bpms = []
    for part in val.split(","):
        part = part.strip()
        if not part:
            continue
        beat_s, bpm_s = part.split("=")
        bpms.append((float(beat_s), float(bpm_s)))
    bpms.sort(key=lambda x: x[0])
    return bpms

def beat_to_seconds(beat: float, bpms: List[Tuple[float,float]]) -> float:
    """Integrate BPM segments from beat 0 to beat."""
    if not bpms:
        raise ValueError("No BPM data in chart.")
    # Ensure a segment at beat 0
    if bpms[0][0] != 0.0:
        bpms = [(0.0, bpms[0][1])] + bpms
    t = 0.0
    for (b0, bpm0), (b1, _) in zip(bpms, bpms[1:] + [(beat, bpms[-1][1])]):
        if beat <= b0:
            break
        seg_end = min(beat, b1)
        beats_in_seg = max(0.0, seg_end - b0)
        t += beats_in_seg * (60.0 / bpm0)
        if beat <= b1:
            break
    return t

def pick_chart(text: str, chart_name: Optional[str]) -> str:
    """
    .ssc can contain multiple #NOTEDATA blocks.
    We'll pick:
    - The first chart if chart_name is None
    - Or match by #DIFFICULTY (e.g. "Hard") or #METER ("10") if provided.
    """
    blocks = text.split("#NOTEDATA:")[1:]
    if not blocks:
        raise ValueError("No #NOTEDATA blocks found in .ssc")

    def extract_field(block: str, field: str) -> str:
        m = re.search(rf'#\s*{re.escape(field)}\s*:\s*(.+?)\s*;', block, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else ""

    candidates = []
    for b in blocks:
        diff = extract_field(b, "DIFFICULTY")
        meter = extract_field(b, "METER")
        stepstype = extract_field(b, "STEPSTYPE")
        candidates.append((diff, meter, stepstype, b))

    if chart_name is None:
        return candidates[0][3]

    want = chart_name.strip().lower()
    for diff, meter, stepstype, b in candidates:
        if want in (diff or "").lower() or want == (meter or "").lower():
            return b

    raise ValueError(f'Chart "{chart_name}" not found. Available: ' +
                     ", ".join([f"{d or '?'}({m or '?'})/{s or '?'}" for d,m,s,_ in candidates]))

def parse_notes_from_notedata(block: str) -> List[Tuple[float,int]]:
    """
    Parse #NOTES from .ssc.
    Supports dance-single only (4 columns). Mines/holds are ignored; we treat any non-zero as a tap.
    """
    m = re.search(r'#\s*NOTES\s*:\s*(.+?)\s*;', block, re.IGNORECASE | re.DOTALL)
    if not m:
        raise ValueError("No #NOTES field in selected chart.")
    raw = m.group(1).strip()

    # Notes are in measures separated by commas, each measure has lines of 4 chars like "0010"
    measures = [meas.strip() for meas in raw.split(",")]
    notes: List[Tuple[float,int]] = []
    beat = 0.0
    for meas in measures:
        if not meas:
            continue
        lines = [ln.strip() for ln in meas.splitlines() if ln.strip() and not ln.strip().startswith("//")]
        if not lines:
            continue
        rows = len(lines)
        for r, line in enumerate(lines):
            # ignore non-note lines
            if len(line) < 4:
                continue
            sub = line[:4]
            lane_mask = 0
            for i, ch in enumerate(sub):
                if ch in ("1","2","4"):  # tap/head/roll treated as note
                    lane_mask |= [LANE_MASK["L"], LANE_MASK["D"], LANE_MASK["U"], LANE_MASK["R"]][i]
            if lane_mask:
                # Each measure is 4 beats.
                beat_pos = beat + (4.0 * r / rows)
                notes.append((beat_pos, lane_mask))
        beat += 4.0
    notes.sort(key=lambda x: x[0])
    return notes

def build_frame_durations(total_seconds: float, fps: float) -> List[int]:
    """
    Build a duration list using 20/21ms style to approximate desired FPS.
    For 48Hz, ideal frame is 20.833ms => pattern of 20 and 21.
    """
    ideal_ms = 1000.0 / fps
    frames = max(1, int(round(total_seconds * fps)))
    durations = []
    acc = 0.0
    for _ in range(frames):
        acc += ideal_ms
        d = int(round(acc)) - sum(durations)
        # clamp to 10..30ms reasonable
        d = max(10, min(30, d))
        durations.append(d)
    return durations

def quantize_notes_to_frames(chart: SSCChart, durations: List[int], offset_ms: float) -> List[int]:
    """
    Convert beat-timed notes into a frame-indexed step stream.
    """
    # Build frame start times
    starts = [0.0]
    t=0.0
    for d in durations:
        t += d/1000.0
        starts.append(t)
    # step stream length = number of frames (one byte per frame)
    stream = [0x20] * len(durations)  # 0x20 means "no step" in sample
    for beat, mask in chart.notes:
        sec = beat_to_seconds(beat, chart.bpms)
        # StepMania's OFFSET is seconds: positive means chart starts later (audio starts earlier).
        # We'll apply: event_time = sec - offset + user_offset
        event_t = sec - chart.offset + (offset_ms/1000.0)
        if event_t < 0:
            continue
        # Find nearest frame
        # simple linear search is fine for typical sizes
        # but we'll do binary
        import bisect
        idx = bisect.bisect_right(starts, event_t) - 1
        idx = max(0, min(idx, len(stream)-1))
        # Combine if multiple notes land on same frame
        if stream[idx] == 0x20:
            stream[idx] = mask
        else:
            stream[idx] |= mask
    return stream

def write_nte(step_stream: List[int], durations: List[int], out_path: pathlib.Path) -> None:
    # Triplicate each step byte
    step_bytes = bytearray()
    for b in step_stream:
        step_bytes.extend([b,b,b])
    # Build timing records:
    # First record low16=13 as seen in sample; then per-frame durations.
    recs = bytearray()
    recs += struct.pack("<HH", 13, 0)  # init record
    for d in durations:
        # Use flag 0x0101 if there's a step on that frame, else 0x0000.
        # This matches the "either 0 or 0x0101" pattern observed.
        flag = 0x0101 if any(step_stream) and d and True else 0x0000
        # Better: mark only frames with non-empty step
        flag = 0x0101 if (d and True) else 0
        recs += struct.pack("<HH", d, 0x0101)  # keep consistent with sample-heavy flags
    # Terminators: 0x0000FFFF then pointer to (step_bytes_len - 6)
    recs += struct.pack("<I", 0x0000FFFF)
    recs += struct.pack("<I", max(0, len(step_bytes) - 6))
    out_path.write_bytes(step_bytes + recs)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_ssc", type=pathlib.Path)
    ap.add_argument("output_nte", type=pathlib.Path)
    ap.add_argument("--chart", help='Chart to use (e.g. "Hard" or "10"). Default: first chart', default=None)
    ap.add_argument("--fps", type=float, default=48.0, help="Frame clock for chart quantization (default 48)")
    ap.add_argument("--offset-ms", type=float, default=0.0, help="Additional offset in ms (positive delays notes)")
    ap.add_argument("--length-s", type=float, default=None, help="Force total length in seconds (otherwise derived from last note)")
    args = ap.parse_args()

    text = args.input_ssc.read_text(errors="ignore")
    kv = parse_kv(text)

    title = kv.get("TITLE","")
    offset = float(kv.get("OFFSET","0") or 0.0)
    bpms = parse_bpms(kv.get("BPMS",""))

    notedata = pick_chart(text, args.chart)
    notes = parse_notes_from_notedata(notedata)

    if not notes:
        raise SystemExit("No notes parsed from chart.")

    # Estimate length: last note + 2 seconds safety
    last_beat = notes[-1][0]
    last_sec = beat_to_seconds(last_beat, bpms) - offset + (args.offset_ms/1000.0)
    total = args.length_s if args.length_s is not None else max(5.0, last_sec + 2.0)

    chart = SSCChart(title=title, bpms=bpms, offset=offset, notes=notes)
    durations = build_frame_durations(total, args.fps)
    step_stream = quantize_notes_to_frames(chart, durations, args.offset_ms)

    write_nte(step_stream, durations, args.output_nte)
    print(f"Wrote {args.output_nte}  (frames={len(durations)}, step_bytes={len(step_stream)*3})")
    print("NOTE: This is experimental. If the mat rejects it, we will refine flags/sections from more real .nte samples.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

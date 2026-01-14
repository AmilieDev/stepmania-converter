#!/usr/bin/env python3
"""
starlevel_add_song.py (FlyRanking A870II)

Adds a new *indexed title slot* into starlevel.dat WITHOUT removing existing songs.

Key discovery from your provided file:
- starlevel.dat contains 228 fixed-size slots of 65 bytes starting at offset 0x207 (519).
- Most slots already contain an obfuscated ASCII title:
    <prefix byte> <title bytes (each ASCII + 1)> 0x01
  Prefix is commonly 0x03, sometimes 0x02.
- There are still empty slots. We can fill one empty slot by reusing a template slot's
  non-title bytes and only changing the title bytes IN-PLACE (no shifting offsets).

This script:
- Finds an empty slot (or uses --slot)
- Copies a template slot (default: slot with the largest title field)
- Replaces the title IN-PLACE (padded with spaces to fit the template's fixed title length)
- Writes a new starlevel.dat

IMPORTANT:
This updates ONLY the displayed/indexed title. Your console likely maps songs to slots by
index number (slot), so you may also need to place your song files using that slot's
expected naming scheme on the SD card.

Usage:
  python3 starlevel_add_song.py starlevel.dat starlevel_patched.dat --title "My Custom Song"

Optional:
  --slot 9        # choose a specific empty slot
  --template 143  # choose which existing slot to clone (default auto)
"""

from __future__ import annotations
import argparse
import pathlib

SLOTS_OFFSET = 519
SLOT_SIZE = 65
TOTAL_SLOTS = 228

def decode(enc: bytes) -> str:
    return bytes([(b-1) & 0xFF for b in enc]).decode('latin1', errors='replace')

def encode(txt: str) -> bytes:
    raw = txt.encode('latin1', errors='replace')
    return bytes([(b+1) & 0xFF for b in raw])

def parse_title_field(slot_bytes: bytes):
    """
    Returns (prefix, enc_title_bytes, end_pos) or None if no recognizable title.
    """
    prefix = slot_bytes[0]
    if prefix not in (2,3,4,5):
        return None
    try:
        end = slot_bytes.index(0x01, 1, 60)
    except ValueError:
        return None
    enc = slot_bytes[1:end]
    if not enc:
        return None
    # basic sanity: decoded should be mostly printable
    dec = decode(enc)
    printable = sum(32 <= ord(c) < 127 for c in dec)
    if printable / max(1, len(dec)) < 0.8:
        return None
    return (prefix, enc, end)

def iter_slots(data: bytes):
    for i in range(TOTAL_SLOTS):
        off = SLOTS_OFFSET + i*SLOT_SIZE
        yield i, off, data[off:off+SLOT_SIZE]

def find_empty_slots(data: bytes):
    empties = []
    for i, off, slot in iter_slots(data):
        if parse_title_field(slot) is None:
            empties.append(i)
    return empties

def pick_best_template(data: bytes) -> int:
    best = (-1, -1)  # (title_len, slot_index)
    for i, off, slot in iter_slots(data):
        info = parse_title_field(slot)
        if not info:
            continue
        _, enc, _ = info
        if len(enc) > best[0]:
            best = (len(enc), i)
    if best[1] < 0:
        raise SystemExit("Could not find any existing template slot with a title.")
    return best[1]

def write_title_into_slot(slot: bytearray, template_slot: bytes, title: str):
    info = parse_title_field(template_slot)
    if not info:
        raise SystemExit("Template slot does not contain a recognizable title.")
    prefix, enc, end = info
    title_len = len(enc)

    new_enc = encode(title)
    if len(new_enc) > title_len:
        raise SystemExit(f'Title too long. Max {title_len} chars for this template. '
                         f'Pick a different --template or shorten the title.')

    # pad with spaces to exact title_len
    pad = encode(" " * (title_len - len(new_enc)))
    new_fixed = new_enc + pad

    # Copy full template first
    slot[:] = template_slot

    # Overwrite prefix and title bytes in-place, keep 0x01 terminator position unchanged
    slot[0] = prefix
    slot[1:1+title_len] = new_fixed
    slot[end] = 0x01

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("in_starlevel", type=pathlib.Path)
    ap.add_argument("out_starlevel", type=pathlib.Path)
    ap.add_argument("--title", required=True, help="New song title (ASCII/latin1)")
    ap.add_argument("--slot", type=int, default=None, help="Which empty slot to fill (default: first empty)")
    ap.add_argument("--template", type=int, default=None, help="Which existing slot to clone (default: auto best)")
    args = ap.parse_args()

    data = bytearray(args.in_starlevel.read_bytes())

    empties = find_empty_slots(data)
    if not empties:
        raise SystemExit("No empty slots found. This starlevel.dat appears full.")
    slot_index = args.slot if args.slot is not None else empties[0]
    if slot_index not in empties:
        raise SystemExit(f"Slot {slot_index} is not empty. Empty slots include: {empties[:20]}{'...' if len(empties)>20 else ''}")

    template_index = args.template if args.template is not None else pick_best_template(data)

    # Extract template slot bytes
    t_off = SLOTS_OFFSET + template_index*SLOT_SIZE
    template_slot = bytes(data[t_off:t_off+SLOT_SIZE])

    # Fill target slot
    s_off = SLOTS_OFFSET + slot_index*SLOT_SIZE
    new_slot = bytearray(SLOT_SIZE)
    write_title_into_slot(new_slot, template_slot, args.title)
    data[s_off:s_off+SLOT_SIZE] = new_slot

    args.out_starlevel.write_bytes(data)

    print(f"Added title '{args.title}' into slot {slot_index} (cloned from template slot {template_index}).")
    print(f"Wrote: {args.out_starlevel}")
    print("\nNext steps:")
    print("1) Copy the patched starlevel.dat back onto the SD card (replace original).")
    print("2) Put your song files into the location your console expects, using the naming scheme for slot index "
          f"{slot_index}. (Often slot index maps to a numbered filename or entry.)")
    print("If you show me how existing song files are named for nearby slots, I can tell you exactly what to name yours.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

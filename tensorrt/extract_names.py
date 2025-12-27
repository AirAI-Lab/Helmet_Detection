#!/usr/bin/env python3
"""
Extract class names from an Ultralytics .pt checkpoint (if available) and write to a names.txt file (one per line).

Usage:
  python tools/tensorrt/extract_names.py --pt /path/to/best.pt --out names.txt
"""
import argparse
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pt', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    try:
        from ultralytics import YOLO
        model = YOLO(args.pt)
        names = model.names if hasattr(model, 'names') else None
    except Exception as e:
        print('Failed to load via ultralytics:', e)
        names = None

    if names is None:
        print('Could not extract names from checkpoint.')
        return

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open('w', encoding='utf8') as f:
        for i in range(len(names)):
            f.write(str(names[i]) + '\n')
    print('Wrote', outp)

if __name__ == '__main__':
    main()

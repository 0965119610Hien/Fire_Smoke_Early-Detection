"""
Script to merge videos from numbered subdirectories (1 to 13) into their respective parent category folders.
Also handles removing redundant uncut 2HOX video and cleaning up empty numbered folders.
"""
import os
import sys
import shutil
from pathlib import Path

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(r"e:\NCKH\Code\Fire_Smoke_Early-Detection\crawled_early_fire_transition\curated_continuous_incidents")

def merge_numbered_directories():
    print("=" * 60)
    print(">> BAT DAU GOP CAC THU MUC CON DUOC DANH SO (1 -> 13) VAO DU LIEU GOC")
    print("=" * 60)
    
    moved_count = 0
    skipped_count = 0
    numbered_dirs = []
    
    for root, dirs, files in os.walk(BASE_DIR):
        for d in dirs:
            if d.isdigit():
                numbered_dirs.append(Path(root) / d)
                
    print(f"Tim thay {len(numbered_dirs)} thu muc con duoc danh so: {[d.name for d in numbered_dirs]}")
    
    for nd in numbered_dirs:
        parent_dir = nd.parent
        print(f"\nDang xu ly thu muc {nd.name}: {nd.relative_to(BASE_DIR)}")
        
        mp4_files = list(nd.glob("*.mp4"))
        for f in mp4_files:
            # Handle special case: uncut 2HOX video
            if f.name == "lab_experiments_2HOX_wDAB3A_continuous_trimmed.mp4":
                print(f"   [INFO] Da co 2 file cat part1 va part2, bo qua file chua cat: {f.name}")
                f.unlink() # remove uncut version
                continue
                
            dest_file = parent_dir / f.name
            if dest_file.exists():
                # If file already exists and has same size, it was already copied
                if dest_file.stat().st_size == f.stat().st_size:
                    print(f"   [SKIP] File da ton tai o thu muc cha ({f.name}), xoa file trung o thu muc con.")
                    f.unlink()
                    skipped_count += 1
                else:
                    print(f"   [WARN] File trung ten nhung khac dung luong: {f.name} (giu nguyen o thu muc cha)")
                    f.unlink()
            else:
                shutil.move(str(f), str(dest_file))
                print(f"   [OK] Da chuyen: {f.name} -> {parent_dir.name}/")
                moved_count += 1
                
        # Check if directory is now empty
        remaining = list(nd.iterdir())
        if not remaining:
            nd.rmdir()
            print(f"   [DEL] Da xoa thu muc con trong: {nd.name}")
        else:
            print(f"   [WARN] Thu muc con con lai file: {[r.name for r in remaining]}")
            
    print("\n" + "=" * 60)
    print("HOAN THANH GOP THU MUC:")
    print(f"   - Da chuyen {moved_count} video ra thu muc cha.")
    print(f"   - Da xu ly {skipped_count} video trung khop.")
    print("=" * 60)

if __name__ == "__main__":
    merge_numbered_directories()

import json
from pathlib import Path

out = Path('output/full_run')

print('=== image_assets.json ===')
assets_path = out / 'images' / 'image_assets.json'
if assets_path.exists():
    assets = json.loads(assets_path.read_text(encoding='utf-8'))
    for a in assets:
        fp = a.get('file_path', '')
        exists = Path(fp).exists() if fp else False
        print(f"  {a['anchor_id']}: placeholder={a['is_placeholder']}, file_exists={exists}")
else:
    print('  NOT FOUND')

print()
print('=== PNG files on disk ===')
img_dir = out / 'images'
if img_dir.exists():
    for f in sorted(img_dir.glob('*.png')):
        print(f'  {f.name} ({f.stat().st_size} bytes)')
else:
    print('  images/ dir not found')

print()
print('=== run_state ===')
rs = json.loads((out / 'run_state.json').read_text(encoding='utf-8'))
print(f"  phase={rs['phase']}, chapter={rs['current_chapter']}")
cover = rs.get('cover_asset')
print(f"  cover_asset={'yes fp='+str(cover.get('file_path',''))[:50] if cover else 'None'}")
accepted = [s['chapter_number'] for s in rs.get('chapter_statuses', []) if s.get('accepted_draft')]
print(f'  accepted chapters: {accepted}')

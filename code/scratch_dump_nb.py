import json, sys
sys.stdout.reconfigure(encoding='utf-8')

nb_path = sys.argv[1]
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"==================== {nb_path} ====================")
for i, c in enumerate(nb['cells']):
    ctype = c.get('cell_type')
    src = ''.join(c.get('source', []))
    print(f"\n[Cell {i}: {ctype}]")
    print(src)
    outputs = c.get('outputs', [])
    if outputs:
        print("\n--- OUTPUTS ---")
        for out in outputs:
            if 'text' in out:
                print(''.join(out['text']))
            elif 'data' in out:
                if 'text/plain' in out['data']:
                    print(''.join(out['data']['text/plain']))

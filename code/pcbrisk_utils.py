import os, json, hashlib, random, shutil, tempfile, subprocess
import numpy as np

D = '/content/drive/MyDrive/pcbrisk'
L = '/content'

def sh(cmd):
    return subprocess.run(cmd, shell=True, check=True)

def seed_all(s=42):
    random.seed(s); np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    except ImportError:
        pass

def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()

def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]

def safe_copy(src, dst, verify=True):
    # unique temp in the destination dir -> verify hash -> atomic rename
    parent = os.path.dirname(dst)
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(dst) + '.', suffix='.tmp', dir=parent)
    os.close(fd)
    try:
        shutil.copy2(src, tmp)
        if verify and sha256_file(src) != sha256_file(tmp):
            raise IOError(f'checksum mismatch copying {src}')
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return dst

def cache_path(name, **parts):
    return f'{D}/03_cache/{name}_{sha256_json(parts)}'

def cached_npz(name, fn, **parts):
    path = cache_path(name, **parts) + '.npz'
    if not os.path.exists(path):
        out = fn()
        tmp = path + '.part'
        with open(tmp, 'wb') as f:          # file handle: numpy won't append a suffix
            np.savez_compressed(f, **out)
        os.replace(tmp, path)
    with np.load(path) as z:
        return {k: z[k] for k in z.files}   # materialise, no open handle

def append_checksum(path, ledger=f'{D}/00_raw/checksums.txt'):
    line = f'{sha256_file(path)}  {os.path.basename(path)}\n'
    existing = open(ledger).read() if os.path.exists(ledger) else ''
    if line not in existing:
        with open(ledger, 'a') as f:
            f.write(line)
    return line.strip()

def restore_tar(tar_on_drive, dest, marker='.ready'):
    # copy one big archive to local SSD and untar (fast), skip if already done
    if os.path.exists(f'{dest}/{marker}'):
        return dest
    os.makedirs(dest, exist_ok=True)
    local = f'{L}/{os.path.basename(tar_on_drive)}'
    shutil.copy2(tar_on_drive, local)
    sh(f'tar -xf {local} -C {dest}')
    open(f'{dest}/{marker}', 'w').close()
    os.remove(local)
    return dest

"""Random access into the iSign pose archive without downloading all 170 GB.

iSign ships its poses as ONE zip file split into four parts
(iSign-poses_v1.1_part_aa .. _ad; the dataset card says to `cat` them together).
A zip keeps its index (the "central directory") at the very end, so we can:

  1. treat the four parts as one virtual file, read via HTTP Range requests;
  2. let Python's zipfile parse the central directory from the tail of part_ad;
  3. fetch any member file on its own with one Range request.

The paper (arXiv 2609.12993) trained on "the iSign poses v1.1 Part AD subset"
(18,867 samples): the members whose data lives in part_ad. `members_in_part`
reproduces exactly that selection.
"""

import io
import random
import struct
import threading
import time
import zipfile
import zlib

import requests

HF_REPO = "Exploration-Lab/iSign"
PART_NAMES = [f"iSign-poses_v1.1_part_{s}" for s in ("aa", "ab", "ac", "ad")]


def hf_url(filename, repo=HF_REPO):
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{filename}"


def part_sizes(token, repo=HF_REPO):
    """Exact byte size of each part, from the Hub API."""
    r = requests.get(f"https://huggingface.co/api/datasets/{repo}/tree/main",
                     headers={"Authorization": f"Bearer {token}"}, timeout=60)
    r.raise_for_status()
    sizes = {f["path"]: f["size"] for f in r.json()}
    return [sizes[name] for name in PART_NAMES]


class MultiPartSource:
    """Byte-range reader over several consecutive parts (HTTP or local files)."""

    def __init__(self, parts, sizes, fetch):
        # parts: list of identifiers; sizes: their byte sizes;
        # fetch(part, start, end_inclusive) -> bytes
        self.parts, self.sizes, self.fetch = parts, sizes, fetch
        self.starts = []
        pos = 0
        for s in sizes:
            self.starts.append(pos)
            pos += s
        self.total = pos

    def part_start(self, index):
        return self.starts[index]

    def read_range(self, start, length):
        """Read `length` bytes at virtual offset `start`, crossing parts if needed."""
        out, pos, end = [], start, min(start + length, self.total)
        while pos < end:
            i = max(k for k, s in enumerate(self.starts) if s <= pos)
            local = pos - self.starts[i]
            take = min(end - pos, self.sizes[i] - local)
            out.append(self.fetch(self.parts[i], local, local + take - 1))
            pos += take
        return b"".join(out)


def http_fetcher(token, retries=10):
    """fetch(part, start, end) via HTTP Range on the gated HF dataset.

    HuggingFace answers HTTP 429 when too many requests arrive at once; we then
    wait as long as its Retry-After header asks (or back off exponentially).
    One requests.Session per thread, since sessions aren't guaranteed thread-safe.
    """
    local = threading.local()

    def session():
        if not hasattr(local, "s"):
            local.s = requests.Session()
            local.s.headers["Authorization"] = f"Bearer {token}"
        return local.s

    def fetch(part, start, end):
        want = end - start + 1
        for attempt in range(retries):
            try:
                r = session().get(hf_url(part), headers={"Range": f"bytes={start}-{end}"},
                                  timeout=120)
                if r.status_code == 206 and len(r.content) == want:
                    return r.content
                if r.status_code in (401, 403):
                    raise PermissionError(
                        f"HTTP {r.status_code}: accept the iSign terms on HuggingFace and "
                        "check your HF token.")
                if r.status_code == 429:
                    wait = float(r.headers.get("Retry-After", 0) or 0) or min(60, 5 * 2 ** attempt)
                    time.sleep(wait + random.random() * 3)
                    continue
                raise IOError(f"HTTP {r.status_code}, got {len(r.content)}/{want} bytes")
            except PermissionError:
                raise
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2 * (attempt + 1))
        raise IOError(f"still rate-limited after {retries} tries ({part} {start}-{end})")
    return fetch


def local_fetcher():
    """fetch(path, start, end) from local files (used by the offline tests)."""
    def fetch(path, start, end):
        with open(path, "rb") as f:
            f.seek(start)
            return f.read(end - start + 1)
    return fetch


class _SeekableView(io.RawIOBase):
    """Minimal seekable file over a MultiPartSource, just so zipfile can parse
    the central directory. Reads are cached in 8 MB blocks."""

    BLOCK = 8 << 20

    def __init__(self, source):
        self.src, self.pos, self.cache = source, 0, {}

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        base = {0: 0, 1: self.pos, 2: self.src.total}[whence]
        self.pos = base + offset
        return self.pos

    def _block(self, b):
        if b not in self.cache:
            self.cache[b] = self.src.read_range(b * self.BLOCK, self.BLOCK)
        return self.cache[b]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.src.total - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.src.total:
            b, off = divmod(self.pos, self.BLOCK)
            chunk = self._block(b)[off:off + n]
            if not chunk:
                break
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, buf):
        data = self.read(len(buf))
        buf[:len(data)] = data
        return len(data)


def list_members(source):
    """All zip members (zipfile.ZipInfo), parsed from the archive's tail."""
    with zipfile.ZipFile(_SeekableView(source)) as zf:
        return [i for i in zf.infolist() if not i.is_dir()]


def members_in_part(members, source, part_index):
    """Members whose local header starts inside part `part_index` (the paper's
    "Part AD subset" is part_index=3)."""
    lo = source.part_start(part_index)
    hi = lo + source.sizes[part_index]
    return [m for m in members if lo <= m.header_offset < hi]


def read_member(source, info):
    """Bytes of one member, with a single range request (plus one if the local
    header's extra field is unusually large)."""
    guess = 30 + len(info.filename.encode("utf-8")) + 256
    head = source.read_range(info.header_offset, guess + info.compress_size)
    if head[:4] != b"PK\x03\x04":
        raise IOError(f"bad local header for {info.filename}")
    name_len, extra_len = struct.unpack("<HH", head[26:30])
    data_start = 30 + name_len + extra_len
    need = data_start + info.compress_size
    if need > len(head):
        head = source.read_range(info.header_offset, need)
    raw = head[data_start:need]
    if info.compress_type == zipfile.ZIP_STORED:
        data = raw
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(raw, -15)
    else:
        raise NotImplementedError(f"compression type {info.compress_type}")
    if zlib.crc32(data) & 0xFFFFFFFF != info.CRC:
        raise IOError(f"CRC mismatch for {info.filename}")
    return data

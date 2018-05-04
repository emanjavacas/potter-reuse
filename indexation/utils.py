
import re


def read_lines(*paths):
    for path in paths:
        idx = 0
        with open(path, 'r') as f:
            for line in f:
                yield path, idx, line.strip()
                idx += 1


def chunks(it, size):
    buf = []
    for s in it:
        buf.append(s)
        if len(buf) == size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def take(it, n):
    tot = 0
    for i in it:
        if tot + 1 > n:
            break
        yield i
        tot += 1


def ensure_ext(path, ext):
    if path.endswith(ext):
        return path
    return path + ".{}".format(re.sub("^\.", "", ext))

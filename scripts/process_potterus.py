
import os
import re
from process_utils import package_tar


def join_dots(line):
    return re.sub('\. \. \.', '...', line)


def read_books(path):
    for f in os.listdir(path):
        p = os.path.join(path, f)
        with open(p, 'r') as f:
            lines = (join_dots(line.strip()) for line in f if line.strip())
            yield os.path.basename(p), lines


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ucto')
    parser.add_argument('--tokenize', action='store_true')
    parser.add_argument('--root', '/home/manjavacas/corpora/potterus/books/')
    args = parser.parse_args()

    tokenizer = None
    if args.tokenize:
        from process_utils import tokenizer
        tokenizer = tokenizer(model=args.model)
        fname = 'potterus_{}.tar.gz'.format(args.model)
    else:
        fname = 'potterus.raw.tar.gz'

    package_tar(fname, read_books(args.root), tokenizer)

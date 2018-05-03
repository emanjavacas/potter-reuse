
import os
from process_utils import package_tar


def read_books(path='/home/manjavacas/corpora/potterus/books/'):
    for f in os.listdir(path):
        p = os.path.join(path, f)
        with open(p, 'r') as f:
            yield os.path.basename(p), [line.strip() for line in f if line.strip()]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ucto')
    parser.add_argument('--tokenize', action='store_true')
    args = parser.parse_args()

    tokenizer = None
    if args.tokenize:
        from process_utils import tokenizer
        tokenizer = tokenizer(model=args.model)
        fname = 'potterus_{}.tar.gz'.format(args.model)
    else:
        fname = 'potterus.raw.tar.gz'

    package_tar(fname, read_books(), tokenizer)


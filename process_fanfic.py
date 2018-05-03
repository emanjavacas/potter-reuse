
import json
from collections import defaultdict


def read_fanfic(store_meta, path='/home/mike/GitRepos/fanfic/fanfics.json'):
    with open(path) as f:
        for idx, line in enumerate(f):
            entry = json.loads(line.strip())
            store_meta[entry['author_id']] += 1
            work_id = store_meta[entry['author_id']]
            fname = '{}.{}.{}'.format(work_id, entry['author_id'], idx)
            yield fname, [l for c in entry['chapters'].values() for l in c.split('\n')]


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
        fname = 'fanfic_{}.tar.gz'.format(args.model)
    else:
        fname = 'fanfic.raw.tar.gz'

    from process_utils import package_tar
    store_meta = defaultdict(int)
    try:
        package_tar(fname, read_fanfic(store_meta), tokenizer)
    except Exception as e:
        print("Exception!", str(e))

    with open('fanfic.meta.csv', 'w+') as f:
        for k, v in store_meta.items():
            f.write('{}\t{}\n'.format(k, v))


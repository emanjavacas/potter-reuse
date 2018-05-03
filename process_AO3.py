
import csv
import sys
csv.field_size_limit(sys.maxsize)


def read_csv(store_meta, path='/home/manjavacas/corpora/AO3/en_fanfic.csv'):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for idx, line in enumerate(reader):
            entry = dict(zip(header, line))
            if entry['author'] in store_meta:
                author_id = store_meta[entry['author']]
            else:
                author_id = len(store_meta) + 1
                store_meta[entry['author']] = author_id
            fname = '{}.{}.{}'.format(entry['work_id'], author_id, idx)
            yield fname, entry['body'].split('\n')


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
        fname = 'AO3_{}.tar.gz'.format(args.model)
    else:
        fname = 'AO3.raw.tar.gz'

    from process_utils import package_tar
    store_meta = {}
    try:
        package_tar(fname, read_csv(store_meta), tokenizer)
    except Exception as e:
        print("Exception!", str(e))

    with open('AO3.meta.csv', 'w+') as f:
        for k, v in store_meta.items():
            f.write('{}\t{}\n'.format(k, v))

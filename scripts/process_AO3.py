
import csv
import sys
csv.field_size_limit(sys.maxsize)


def read_csv(path, store_meta, filter_meta=None):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for idx, line in enumerate(reader):
            if filter_meta is not None and not filter_meta.get(idx):
                continue
            entry = dict(zip(header, line))
            if entry['author'] in store_meta:
                author_id = store_meta[entry['author']]
            else:
                author_id = len(store_meta) + 1
                store_meta[entry['author']] = author_id
            fname = '{}.{}.{}'.format(entry['work_id'], author_id, idx)
            yield fname, entry['body'].split('\n')


def get_HP_meta(path):
    filter_meta = {}
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for idx, line in enumerate(reader):
            entry = dict(zip(header, line))
            if entry['fandom'] == 'Harry Potter - J. K. Rowling':
                if idx not in set(442, 443, 0, 1):  # non-english entries
                    filter_meta[idx] = True

    return filter_meta


def read_processed(path, idxs):
    import tarfile
    with tarfile.open(path, 'r:gz')as tar:
        for m in tar.getmembers():
            work_id, author_id, idx = m.name.split('.')
            if int(idx) in idxs:
                yield tar.extractfile(m).read()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ucto')
    parser.add_argument('--tokenize', action='store_true')
    parser.add_argument('--ignore_quotes', action='store_true')
    parser.add_argument('--root', default='/home/manjavacas/corpora/AO3/en_fanfic.csv')
    args = parser.parse_args()

    tokenizer = None
    if args.tokenize:
        from process_utils import tokenizer
        tokenizer = tokenizer(model=args.model)
        fname = 'AO3_{}.{}tar.gz'.format(
            args.model, 'quote.' if not args.ignore_quotes else '')
    else:
        fname = 'AO3.raw.tar.gz'

    from process_utils import package_tar
    store_meta, filter_meta = {}, get_HP_meta(args.root)
    print(len(filter_meta))
    try:
        package_tar(fname, read_csv(args.root, store_meta, filter_meta=filter_meta),
                    tokenizer, args.ignore_quotes)
    except Exception as e:
        print("Exception!", str(e))

    with open('AO3.meta.csv', 'w+') as f:
        for k, v in store_meta.items():
            f.write('{}\t{}\n'.format(k, v))

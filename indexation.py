
import sys
import re
import logging
import os
import shutil
import uuid
import json
import tarfile
import gzip

import numpy as np
import faiss


class Indexer(object):
    """
    Indexer class to manage FAISS indices.
    """
    INDEX = 'index.faiss'
    TEXT = 'text.zip'
    META = 'meta.json.zip'

    def __init__(self, dim=None):
        self.index = None
        if dim is not None:
            self.index = faiss.IndexFlatIP(dim)
        self.text = []
        self.meta = []

    def index_batch(self, batch, texts, meta=None):
        """
        Index a single batch of texts
        """
        if len(batch) != len(texts):
            raise ValueError(
                "Unequal number of items. Got #{} embeddings but only text for #{}"
                .format(len(batch), len(texts)))
        if meta is not None:
            if len(batch) != len(meta):
                raise ValueError(
                    "Unequal number of items. "
                    "Got #{} embeddings but only metadata for #{}"
                    .format(len(batch), len(meta)))
            if self.index.ntotal != len(self.meta):
                raise ValueError(
                    "Seems like index wasn't expecting metadata, "
                    "but metadata was provided")

        self.index.add(batch)
        self.text.extend(texts)
        if meta is not None:
            self.meta.extend(meta)

    def index_generator(self, encoder, inp, bsize=5000, **kwargs):
        """
        Index a generator over tuples of (sentence, sentence metadata)
        """
        for chunk in chunks(inp, bsize):
            texts, meta = zip(*chunk)
            embs = encoder(texts, **kwargs).astype(np.float32)
            faiss.normalize_L2(embs)
            self.index_batch(embs, texts, meta)

    def index_files(self, encoder, *paths, **kwargs):
        """
        Index files
        """
        for path in paths:
            inp = ((line, {'num': idx, 'path': path})
                   for idx, line in read_lines(path))
            self.index_generator(encoder, inp, **kwargs)

    def serialize(self, path):
        """
        Serialize index to single tar file with 2 members: 
            - index.faiss
            - meta.json.zip
        """
        fid = str(uuid.uuid1())

        # index
        indexname = '/tmp/{}-index'.format(fid)
        faiss.write_index(self.index, indexname)

        # texts (temporary fix, in the future it should use a database)
        textname = '/tmp/{}-texts'.format(fid)
        with gzip.GzipFile(textname, 'w') as f:
            for text in self.text:
                f.write((text + "\n").encode())

        # meta
        metaname = '/tmp/{}-index.meta.zip'.format(fid)
        with gzip.GzipFile(metaname, 'w') as f:
            f.write(json.dumps(self.meta).encode())

        # package into a tarfile
        with tarfile.open(ensure_ext(path, 'tar'), 'w') as f:
            f.add(indexname, arcname=Indexer.INDEX)
            f.add(textname, arcname=Indexer.TEXT)
            f.add(metaname, arcname=Indexer.META)

        # cleanup
        os.remove(indexname)
        os.remove(textname)
        os.remove(metaname)

    @classmethod
    def load(cls, path):
        """
        Instantiates Indexer from serialized tar
        """
        index, meta = None, None

        with tarfile.open(ensure_ext(path, 'tar'), 'r') as tar:
            # read index
            indextmp = '/tmp/{}-index'.format(str(uuid.uuid1()))
            tar.extract(cls.INDEX, path=indextmp)
            index = faiss.read_index(os.path.join(indextmp, cls.INDEX))
            shutil.rmtree(indextmp)

            # read text
            text = gzip.open(tar.extractfile(cls.TEXT)).read().decode().strip()

            # read meta
            meta = json.loads(gzip.open(tar.extractfile(cls.META)).read().decode())

        inst = cls()
        inst.index = index
        inst.text = text.split('\n')
        inst.meta = meta

        return inst

    def _query(self, encoder, inp, fp, NNs=10, bsize=5000, **kwargs):
        """
        Run query over batch of input sentences
        
        Parameters
        ===========
        encoder : function that encodes input text into sentence embeddings
        inp : iterator over dictionaries with entries "num", "path" and "text"
        """
        if self.index.ntotal == 0:
            raise ValueError("Empty index.")

        for chunk in chunks(inp, bsize):
            # encode
            text, meta = zip(*chunk)
            embs = encoder(text, **kwargs).astype(np.float32)
            faiss.normalize_L2(embs)
            D, I = self.index.search(embs, NNs)

            # serialize
            for bd, bi, bdata in zip(D, I, meta):
                # path num [source_id:similarity]+
                fp.write('{path}\t{num}\t{sims}\n'.format(
                    path=bdata['path'],
                    num=bdata['num'],
                    sims='+'.join('{}:{:g}'.format(i, d) for d, i in zip(bd, bi))))

    def query_files(self, encoder, outpath, *paths, NNs=10, bsize=500, **kwargs):
        """
        Conveniently process query spread across (possibly) multiple files in a memory
        efficient way
        """
        query_file = ensure_ext(outpath, 'tsv')
        if os.path.isfile(query_file):
            raise ValueError("Output query file {} already exists".format(query_file))

        with open(query_file, 'a+') as f:
            # write metadata about files for efficient storage
            paths_ = {path: idx for idx, path in enumerate(paths)}
            f.write("#{}\n".format('\t'.join(paths)))

            # process files
            for path in paths:
                logging.debug("Processing {}".format(path))
                inp = ((line, {'num': idx, 'path': paths_[path]})
                        for idx, line in read_lines(path))
                self._query(encoder, inp, f, NNs=NNs, bsize=bsize, **kwargs)

    @staticmethod
    def _parse_sims(sims):
        """
        Parse similarity queries from the query output file
        """
        nn, sims = zip(*map(lambda nn: nn.split(':'), sims.split('+')))
        nn, sims = list(map(int, nn)), list(map(float, sims))
        return nn, sims

    def inspect(self, results_file, *query_files, threshold=0.5, max_NNs=5):
        """
        Get a generator over matches based on query file. `query_files` must be passed
        in the same order as they were passed during querying.
        """
        with open(ensure_ext(results_file, 'tsv'), 'r') as f:
            paths_ = next(f).strip()[1:].split('\t')

            for line, (idx, trg) in zip(f, read_lines(*query_files)):
                # idx should be equal to num
                path_, num, sims = line.strip().split('\t')
                path = paths_[int(path_)]
                nns, sims = Indexer._parse_sims(sims)

                # check threshold
                if max(sims) < args.threshold:
                    continue

                # package
                match = {'trg': trg, 'matches': []}
                for nn, sim in take(zip(nns, sims), max_NNs):
                    if sim < args.threshold:
                        break

                    match['matches'].append(
                        {'src': self.text[nn],
                         'meta': self.meta[nn],
                         'sim': sim})

                yield match


def read_lines(*paths):
    for path in paths:
        idx = 0
        with open(path, 'r') as f:
            for line in f:
                yield idx, line.strip()
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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action')
    parser.add_argument('--indexer', required=True)
    parser.add_argument('--index_files', nargs='*')
    parser.add_argument('--bsize', type=int, default=10000)
    parser.add_argument('--NNs', type=int, default=5)
    parser.add_argument('--dim', type=int, default=4800)
    parser.add_argument('--query_files', nargs='*')
    parser.add_argument('--results_file')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    actions = set(args.action.lower().split())

    # print("Loading model...")
    # import skipthoughts
    # model = skipthoughts.load_model()

    # def encoder(sents):
    #     embs = np.array(skipthoughts.encode(model, sents, use_norm=False))
    #     return embs

    def encoder(sents):
        return np.random.randn(len(sents), args.dim)

    indexer = None
    try:
        indexer = Indexer.load(args.indexer)
    except:
        pass

    if 'index' in actions:
        print("Indexing...")
        do_serialize = indexer is None and args.indexer

        if len(args.index_files) == 0:
            raise ValueError("Indexing requires `index_files`")

        indexer = indexer or Indexer(dim=args.dim)
        indexer.index_files(encoder, *args.index_files)
        if do_serialize:
            indexer.serialize(args.indexer)

    if 'query' in actions:
        print("Querying...")

        if indexer is None:
            raise ValueError("Couldn't initialize indexer")
        if not args.results_file:
            raise ValueError("Querying requires `results_file`")
        if len(args.query_files) == 0:
            raise ValueError("Querying requires `query_files`")

        indexer.query_files(encoder, args.results_file, *args.query_files, NNs=args.NNs)

    if 'inspect' in actions:
        if not args.results_file:
            raise ValueError("Inspecting requires `results_file`")
        if len(args.query_files) == 0:
            raise ValueError("Querying requires `query_files`")

        matches = indexer.inspect(args.results_file, *args.query_files,
                                  threshold=args.threshold, max_NNs=args.NNs)

        try:
            match = next(matches)
            print(" => {}".format(match['trg']), flush=True)
            # TODO: print metadata
            for src in match['matches']:
                print(" *** [{:.3f}] {}\n".format(src['sim'], src['src']), flush=True)
                # TODO: print matches

        except BrokenPipeError:
            print("Bye!", file=sys.stderr)

        except StopIteration:
            print("Finished!", file=sys.stderr)

    sys.stderr.close()

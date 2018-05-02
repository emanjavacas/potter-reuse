
import logging
import os
import shutil
import uuid
import json
import tarfile
import gzip

import numpy as np
import skipthoughts
import faiss


class Indexer(object):
    """
    Indexer class to manage FAISS indices.
    """
    INDEX = 'index.faiss'
    META = 'meta.json.zip'

    def __init__(self, dim=None):
        self.index = None
        if dim is not None:
            self.index = faiss.IndexFlatIP(dim)
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

    def index_files(self, encoder, *fpaths, **kwargs):
        """
        Index files
        """
        for fpath in fpaths:
            inp = ((line, {'num': idx, 'fpath': fpath})
                   for idx, line in enumerate(read_lines(fpath)))
            self.index_generator(encoder, inp, **kwargs)

    def serialize(self, prefix):
        """
        Serialize index to single tar file with 2 members: 
            - index.faiss
            - meta.json.zip
        """
        fid = str(uuid.uuid1())

        # index
        indexname = '/tmp/{}-index'.format(fid)
        faiss.write_index(self.index, indexname)

        # meta
        metaname = '/tmp/{}-index.meta.zip'.format(fid)
        with gzip.GzipFile(metaname, 'w') as f:
            f.write(json.dumps(self.meta).encode())

        # package into a tarfile
        with tarfile.open('{}.index.tar'.format(prefix), 'w') as f:
            f.add(indexname, arcname=Indexer.INDEX)
            f.add(metaname, arcname=Indexer.META)

        # cleanup
        os.remove(indexname)
        os.remove(metaname)

    @classmethod
    def load(cls, fpath):
        """
        Instantiates Indexer from serialized tar
        """
        index, meta = None, None

        with tarfile.open(fpath, 'r') as tar:
            # read index
            indextmp = '/tmp/{}-index'.format(str(uuid.uuid1()))
            tar.extract(cls.INDEX, path=indextmp)
            index = faiss.read_index(os.path.join(indextmp, cls.INDEX))
            shutil.rmtree(indextmp)

            # read meta
            meta = json.loads(gzip.open(tar.extractfile(cls.META)).read().decode())

        inst = cls()
        inst.index = index
        inst.meta = meta

        return inst

    def _query(self, encoder, inp, fp, NNs=10, bsize=5000, **kwargs):
        """
        Run query over batch of input sentences
        
        Parameters
        ===========
        encoder : function that encodes input text into sentence embeddings
        inp : iterator over dictionaries with entries "num", "fpath" and "text"
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
                # fpath num [source_id:similarity]+
                fp.write('{fpath}\t{num}\t{sims}\n'.format(
                    fpath=bdata['fpath'],
                    num=bdata['num'],
                    sims='+'.join('{}:{:g}'.format(i, d) for d, i in zip(bd, bi))))

    def query_from_files(self, encoder, prefix, *fpaths, NNs=10, bsize=500, **kwargs):
        """
        Conveniently process query spread across (possibly) multiple files in a memory
        efficient way
        """
        query_file = '{}.tsv'.format(prefix)
        if os.path.isfile(query_file):
            raise ValueError("Output query file {} already exists".format(query_file))

        with open(query_file, 'a+') as f:
            # write metadata about files for efficient storage
            fpaths_ = {fpath: idx for idx, fpath in enumerate(fpaths)}
            f.write("#{}\n".format('\t'.join(fpaths)))

            # process files
            for fpath in fpaths:
                logging.debug("Processing {}".format(fpath))
                inp = ((line, {'num': idx, 'fpath': fpaths_[fpath]})
                        for idx, line in enumerate(read_lines(fpath)))
                self._query(encoder, inp, f, NNs=NNs, bsize=bsize, **kwargs)

    def inspect(self, query_file):
        pass


def read_lines(*paths):
    for path in paths:
        with open(path, 'r') as f:
            for line in f:
                yield line.strip()


def chunks(it, size):
    buf = []
    for s in it:
        buf.append(s)
        if len(buf) == size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action')
    parser.add_argument('--indexer', required=True)
    parser.add_argument('--index_files')
    parser.add_argument('--bsize', type=int, default=10000)
    parser.add_argument('--NNs', type=int, default=5)
    parser.add_argument('--query_files')
    args = parser.parse_args()
    actions = set(args.action.lower().split())

    print("Loading model...")
    model = skipthoughts.load_model()

    def encoder(sents):
        embs = np.array(skipthoughts.encode(model, sents, use_norm=False))
        faiss.normalize_L2(embs)
        return embs

    if 'index' in actions:

        vecs = encode(list(read_lines(args.index_file)))
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        faiss.write_index(index, '{}.index'.format(args.index_file))

    elif 'query' in actions:
        index = None
        try:
            index = faiss.read_index(args.index_file)
        except:
            raise ValueError("Couldn't read index {}".format(args.index_file))

        idx = 0
        with open('{}.results'.format(args.query_file), 'w+') as q:
            for n, chunk in enumerate(chunks(read_lines(args.query_file), args.bsize)):
                print("Processing [{}/{}] lines".format(
                    n * args.bsize, (n + 1) * args.bsize))
                vecs = encode(chunk)
                D, I = index.search(vecs, args.NNs)
                for d, i in zip(D, I):
                    idx += 1
                    q.write("{}\t".format(idx))
                    q.write("\t".join("{}:{:g}".format(ii, dd) for dd, ii in zip(d, i)))
                    q.write("\n")

    else:
        raise ValueError("Unknown action", args.action)

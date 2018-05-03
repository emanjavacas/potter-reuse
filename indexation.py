
import io
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


class BaseTextIndex(object):
    """
    Abstract class defining functionality to store indexed text data.
    It consists of two collections. (1) Text collection indexing the source text
    against which queries will be run. This text is indexed with identifiers
    that refer to their position in the associated FAISS index. (2) Query
    collection(s) that contain the text and metadata of a query run against
    the FAISS index as well as references and similarity scores to sentences in (1).
    """
    def __len__(self):
        raise NotImplementedError

    @classmethod
    def load(cls, tar):
        """
        Load TextIndex (possibly) using data stored in the global index tar file
        during serialization
        """
        raise NotImplementedError

    def on_serialize(self, indexer, tar):
        """
        Optional callback to store index identifiers into the global index tar file
        """
        pass

    def add(self, text, meta):
        """
        Add text and metadata to TextIndex. `text` and `meta` can be single items
        or a batch of items. Each meta instance must have unique identifier that
        will be used to retrieved the instance in the future
        """
        raise NotImplementedError

    def dump_query(self, query_name, generator):
        """
        Dump a query of sentence(s) with related metadata
        """
        raise NotImplementedError

    def get_indexed_text(self, text_id):
        """
        Retrieve indexed text based on identifiers
        """
        raise NotImplementedError

    def inspect_query(self, query_name, threshold, max_NNs, skip=0, sort=False):
        """
        Retrieve matches from an existing query. Return a generator over
        matches 
        """
        raise NotImplementedError


class FileTextIndex(BaseTextIndex):
    """
    Basic TextIndex using files to store the data. Both index text and queries
    will be stored in compressed text files.
    """
    TEXT = 'text.zip'
    META = 'meta.zip'

    def __init__(self, **kwargs):
        self.text = []
        self.meta = []

    def __len__(self):
        return len(self.text)

    @classmethod
    def load(cls, tar):
        inst = cls()
        text = gzip.open(tar.extractfile(cls.TEXT)).read().decode()
        inst.text = text.strip().split('\n')
        inst.meta = json.loads(gzip.open(tar.extractfile(cls.META)).read().decode())

        return inst

    def on_serialize(self, indexer, tar):
        """
        Attach zipped FileTextIndex data to index tar file
        """
        fid = str(uuid.uuid1())

        # text
        textname = '/tmp/{}-text'.format(fid)
        with gzip.GzipFile(textname, 'w') as f:
            for text in self.text:
                f.write((text + "\n").encode())

        # meta
        metaname = '/tmp/{}-index.meta.zip'.format(fid)
        with gzip.GzipFile(metaname, 'w') as f:
            f.write(json.dumps(self.meta).encode())

        tar.add(textname, arcname=FileTextIndex.TEXT)
        tar.add(metaname, arcname=FileTextIndex.META)

        os.remove(textname)
        os.remove(metaname)

    def add(self, text, meta):
        if isinstance(text, list):
            self.text.extend(text)
            self.meta.extend(meta)
        elif isinstance(text, str):
            self.text.append(text)
            self.meta.append(meta)
        else:
            raise ValueError("Unknown input type to FileTextIndex.add: {}"
                             .format(type(text).__name__))

    def dump_query(self, query_name, generator):
        query_file = ensure_ext(query_name, 'tsv')
        if os.path.isfile(query_file):
            raise ValueError("Output query file {} already exists".format(query_file))

        logging.warn("Dumping query to {}".format(query_file))
        paths_ = {}

        with open(query_file, 'w+') as f:

            for target, meta, sources, scores in generator:

                if meta['path'] in paths_:
                    path = paths_[meta['path']]
                else:
                    path = len(paths_)
                    paths_[meta['path']] = path

                sims = '+'.join('{}:{:g}'.format(i, d) for i, d in zip(sources, scores))
                # text path num [source_id:similarity]+
                line = '{target}\t{path}\t{num}\t{sims}\n'.format(
                    target=target, path=path, num=meta['num'], sims=sims)
                f.write(line)

            # write metadata about files for efficient storage
            f.seek(0)           # write path metadata to front
            for path, key in paths_.items():
                f.write('#{}\t{}\n'.format(path, key))

    def get_indexed_text(self, text_id):
        return self.text[text_id], self.meta[text_id]

    @staticmethod
    def _parse_scores(scores):
        """
        Parse similarity queries from the query output file
        """
        nn, scores = zip(*map(lambda nn: nn.split(':'), scores.split('+')))
        nn, scores = list(map(int, nn)), list(map(float, scores))
        return nn, scores

    def inspect_query(self, query_name, threshold, max_NNs):

        if not os.path.isfile(ensure_ext(query_name, 'tsv')):
            raise ValueError("Query doesn't exist: {}".format(query_name))

        with open(ensure_ext(query_name, 'tsv'), 'r') as f:

            paths_ = next(f).strip()[1:].split('\t')

            for line in f:
                target, path_, num, scores = line.strip().split('\t')
                path = paths_[int(path_)]
                target_meta = {'path': path, 'num': num}
                nns, scores = FileTextIndex._parse_scores(scores)

                # check threshold
                if max(scores) < threshold:
                    continue

                # package
                for nn, score in take(zip(nns, scores), max_NNs):
                    if score < threshold:
                        break

                    source, source_meta = self.text[nn], self.meta[nn]

                    yield {'target': (target, target_meta),
                           'source': (source, source_meta),
                           'score': score}


class RelationalTextIndex(BaseTextIndex):
    def __init__(self):
        pass



class Indexer(object):
    """
    Indexer class to manage FAISS indices.
    
    Parameters
    ==========
    dim : int
    text_index : BaseTextIndex
    make_text_index : func

    Attributes
    ==========
    index : FaissIndex
    text_index : BaseTextIndex
    """
    INDEX = 'index.faiss'
    TEXT_INDEX = 'text_index'

    def __init__(self,
                 dim=None,
                 text_index=None,
                 name=None,
                 make_text_index=FileTextIndex):

        self.index = None if dim is None else faiss.IndexFlatIP(dim)
        self.text_index = text_index or make_text_index(name or str(uuid.uuid1()))

    def index_batch(self, batch, text, meta):
        """
        Index a single batch of text
        """
        if len(batch) != len(text) or len(batch) != len(meta):
            raise ValueError(
                "Unequal number of items. Got #{} embeddings but #{}/{} text/meta"
                .format(len(batch), len(text), len(meta)))

        if self.index.ntotal != len(self.text_index):
            if self.index.ntotal != len(self.text_index):
                raise ValueError("Seems like index metadata is out of sync")

        self.index.add(batch)
        self.text_index.add(list(text), list(meta))

    def index_generator(self, encoder, inp, bsize=5000, **kwargs):
        """
        Index a generator over tuples of (sentence, sentence metadata)
        """
        for chunk in chunks(inp, bsize):
            text, meta = zip(*chunk)
            embs = encoder(text, **kwargs).astype(np.float32)
            faiss.normalize_L2(embs)
            self.index_batch(embs, text, meta)

    def index_files(self, encoder, *paths, **kwargs):
        """
        Index files
        """
        for path in paths:
            inp = ((line, {'num': idx, 'path': path})
                   for idx, _, line in read_lines(path))
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

        # package into a tarfile
        with tarfile.open(ensure_ext(path, 'tar'), 'w') as f:
            # add faiss index
            f.add(indexname, arcname=Indexer.INDEX)
            # add metadata
            metainfo = tarfile.TarInfo(Indexer.TEXT_INDEX)
            metadata = type(self.text_index).__name__.encode()
            metainfo.size = len(metadata)
            f.addfile(metainfo, io.BytesIO(metadata))
            # add text_index eventual metadata
            self.text_index.on_serialize(self, f)

        # cleanup
        os.remove(indexname)

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

            # text index
            text_index_type = tar.extractfile(cls.TEXT_INDEX).read().decode()
            if text_index_type == 'FileTextIndex':
                text_index = FileTextIndex.load(tar)
            else:
                raise ValueError("Seems like this was the wrong text indexer: {}"
                                 .format(text_index_type))

        inst = cls()
        inst.index = index
        inst.text_index = text_index
        inst.path = path  # TODO: once serialized, we can free mem if required

        return inst

    def query_generator(self, encoder, inp, NNs=10, bsize=5000, **kwargs):
        """
        Create a generator over query items
        """
        if self.index.ntotal == 0:
            raise ValueError("Empty index.")

        for chunk in chunks(inp, bsize):
            # encode
            targets, meta = zip(*chunk)
            embs = encoder(targets, **kwargs).astype(np.float32)
            faiss.normalize_L2(embs)
            D, I = self.index.search(embs, NNs)

            # serialize
            for target, bmeta, sources, scores in zip(targets, meta, I, D):
                yield target, bmeta, sources, scores

    def dump_query(self, encoder, inp, query_name, **kwargs):
        """
        Run query over batch of input sentences
        
        Parameters
        ===========
        encoder : function that encodes input text into sentence embeddings
        inp : iterator over tuples of shape
            (query, {"num": int, "path": string}), where
            `query` is the query text, `num` is the line number of the query
            and `path` is the source file where the query belongs to.
        """
        self.text_index.dump_query(
            query_name, self.query_generator(encoder, inp, **kwargs))

    def query_from_files(self, encoder, query_name, *paths, **kwargs):
        """
        Run query over files with sentence per line format
        """
        inp = ((line, {'num': idx, 'path': path})
               for idx, path, line in read_lines(*paths))

        self.text_index.dump_query(
            query_name, self.query_generator(encoder, inp, **kwargs))

    def inspect_query(self, query_name, threshold=0.0, max_NNs=5):
        """
        Get a generator over matches where each match is
        {'source': (source, source_meta),
         'target': (target, target_meta), 
         'score': similarity}
        """
        yield from self.text_index.inspect_query(
            query_name, threshold=threshold, max_NNs=max_NNs)


def read_lines(*paths):
    for path in paths:
        idx = 0
        with open(path, 'r') as f:
            for line in f:
                yield idx, path, line.strip()
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
    parser.add_argument('--query_name')
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
        if not args.query_name:
            raise ValueError("Querying requires `query_name`")
        if len(args.query_files) == 0:
            raise ValueError("Querying requires `query_files`")

        indexer.query_from_files(
            encoder, args.query_name, *args.query_files, NNs=args.NNs)

    if 'inspect' in actions:
        if not args.results_file:
            raise ValueError("Inspecting requires `results_file`")
        if len(args.query_files) == 0:
            raise ValueError("Querying requires `query_files`")

        matches = indexer.inspect(
            args.results_file, threshold=args.threshold, max_NNs=args.NNs)

        try:
            match = next(matches)
            print(" => {}".format(match['source']), flush=True)
            # TODO: print metadata
            print(" *** [{:.3f}] {}\n".format(
                match['score'], match['target']), flush=True)

        except BrokenPipeError:
            print("Bye!", file=sys.stderr)

        except StopIteration:
            print("Finished!", file=sys.stderr)

    sys.stderr.close()

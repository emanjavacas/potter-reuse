
import io
import os
import shutil
import uuid
import tarfile

import numpy as np
import faiss

import utils

from text_indexer import FileTextIndex
from relational_indexer import RelationalTextIndex


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
    __id__ = 'name'
    __index__ = 'index.faiss'
    __text_index__ = 'text_index'

    def __init__(self,
                 dim=None,
                 text_index=None,
                 name=None,
                 make_text_index=RelationalTextIndex):

        self.name = name or str(uuid.uuid1())
        self.index = None if dim is None else faiss.IndexFlatIP(dim)
        self.text_index = text_index or make_text_index(self.name)
        # pointer to eventual serialized index; once it's serialized
        # we take advantage of it to offload faiss index and free memory
        self.path = None

    def index_batch(self, batch, text, meta):
        """
        Index a single batch of text
        """
        if len(batch) != len(text) or len(batch) != len(meta):
            raise ValueError(
                "Unequal number of items. Got #{} embeddings but #{}/{} text/meta"
                .format(len(batch), len(text), len(meta)))

        if self.index.ntotal != len(self.text_index):
            raise ValueError("Seems like index metadata is out of sync")

        start, stop = self.index.ntotal, self.index.ntotal + len(text)
        for n, idx in enumerate(range(start, stop)):
            meta[n]['id'] = idx
        self.index.add(batch)
        self.text_index.add(list(text), list(meta))

    def index_generator(self, encoder, inp, bsize=5000, **kwargs):
        """
        Index a generator over tuples of (sentence, sentence metadata)
        """
        for chunk in utils.chunks(inp, bsize):
            text, meta = zip(*chunk)
            embs = encoder(text, **kwargs).astype(np.float32)
            faiss.normalize_L2(embs)
            self.index_batch(embs, text, meta)

    def index_files(self, encoder, *paths, verbose=False, **kwargs):
        """
        Index files
        """
        inp = ((line, {'path': path, 'num': num})
               for path, num, line in utils.read_lines(*paths, verbose=verbose))

        self.index_generator(encoder, inp, **kwargs)

    def serialize(self, path):
        """
        Serialize index to single tar file and additional text index metadata
        """
        # index
        indexname = '/tmp/{}-index'.format(str(uuid.uuid1()))
        faiss.write_index(self.index, indexname)

        # package into a tarfile
        with tarfile.open(utils.ensure_ext(path, 'tar'), 'w') as f:
            # add index identifier
            index_name = tarfile.TarInfo(Indexer.__id__)
            index_name.size = len(self.name.encode())
            f.addfile(index_name, io.BytesIO(self.name.encode()))
            # add faiss index
            f.add(indexname, arcname=Indexer.__index__)
            # add text index type
            text_index_type = tarfile.TarInfo(Indexer.__text_index__)
            data = type(self.text_index).__name__.encode()
            text_index_type.size = len(data)
            f.addfile(text_index_type, io.BytesIO(data))
            # add text_index eventual metadata
            self.text_index.on_serialize(self, f)

        # cleanup
        os.remove(indexname)

    def unload_faiss(self):
        """
        Free memory by unloading the faiss index
        """
        if not self.path:
            raise ValueError("Index hasn't been serialized yet")

        self.index = None

    def load_faiss(self):
        """
        Auxiliary method to load the faiss index
        """
        if self.index is not None:
            raise ValueError("Index is already loaded")

        if self.path is None:
            raise ValueError("Index hasn't been serialized yet")

        with tarfile.open(utils.ensure_ext(self.path, 'tar'), 'r') as tar:
            # name
            name = tar.extractfile(Indexer.__id__).read().decode().strip()
            # validate name
            if name != self.name:
                raise ValueError("Wrong index: {} != {}".format(name, self.name))

            # read index
            indextmp = '/tmp/{}-index'.format(str(uuid.uuid1()))
            tar.extract(Indexer.__index__, path=indextmp)
            index = faiss.read_index(os.path.join(indextmp, Indexer.__index__))
            shutil.rmtree(indextmp)

        self.index = index

    @classmethod
    def load(cls, path):
        """
        Instantiates Indexer from serialized tar
        """
        with tarfile.open(utils.ensure_ext(path, 'tar'), 'r') as tar:
            # name
            name = tar.extractfile(cls.__id__).read().decode().strip()
            
            # read index
            indextmp = '/tmp/{}-index'.format(str(uuid.uuid1()))
            tar.extract(cls.__index__, path=indextmp)
            index = faiss.read_index(os.path.join(indextmp, cls.__index__))
            shutil.rmtree(indextmp)

            # text index
            text_index_type = tar.extractfile(cls.__text_index__).read().decode()
            if text_index_type == 'FileTextIndex':
                text_index = FileTextIndex.load(name, tar)
            elif text_index_type == 'RelationalTextIndex':
                text_index = RelationalTextIndex.load(name, tar)
            else:
                raise ValueError("Seems like this was the wrong text indexer: {}"
                                 .format(text_index_type))

        inst = cls()
        inst.index = index
        inst.text_index = text_index
        inst.path = path  # TODO: once serialized, we can free FAISS mem if required

        return inst

    def query_generator(self, encoder, inp, NNs=10, bsize=5000, **kwargs):
        """
        Create a generator over query items
        """
        if self.index.ntotal == 0:
            raise ValueError("Empty index.")

        for chunk in utils.chunks(inp, bsize):
            # encode
            targets, meta = zip(*chunk)
            embs = encoder(targets, **kwargs).astype(np.float32)
            faiss.normalize_L2(embs)
            D, I = self.index.search(embs, NNs)

            # generate matches
            yield from zip(targets, meta, I, D)

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
        inp = ((line, {'num': num, 'path': path})
               for path, num, line in utils.read_lines(*paths))

        self.text_index.dump_query(
            query_name, self.query_generator(encoder, inp, **kwargs))

    def inspect_query(self, query_name, threshold=0.0, max_NNs=5, **kwargs):
        """
        Get a generator over matches where each match is
        {'source': (source, source_meta),
         'target': (target, target_meta), 
         'score': similarity}
        """
        yield from self.text_index.inspect_query(
            query_name, threshold=threshold, max_NNs=max_NNs, **kwargs)

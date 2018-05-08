
import os
import json
import gzip
import logging
import uuid

import utils


class BaseTextIndex(object):
    """
    Abstract class defining functionality to store indexed text data.
    It consists of two collections. (1) Text collection indexing the source text
    against which queries will be run. This text is indexed with identifiers
    that refer to their position in the associated FAISS index. (2) Query
    collection(s) that contain the text and metadata of a query run against
    the FAISS index as well as references and similarity scores to sentences in (1).
    """
    def __init__(self, name):
        self.name = name

    def __len__(self):
        raise NotImplementedError

    @classmethod
    def load(cls, name, tar):
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
        Add text and metadata to TextIndex. All arguments can be single items
        or a batch of items. All meta objects have an `id` entry that refers to
        the corresponding row in the faiss index.
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

    def inspect_query(self, query_name, threshold, max_NNs,
                      by_source=False, skip=0, sort=False):
        """
        Retrieve matches from an existing query. Return a generator over
        matches 
        """
        raise NotImplementedError


def _parse_scores(scores):
    """
    Parse similarity queries from the query output file
    """
    nn, scores = zip(*map(lambda nn: nn.split(':'), scores.split('+')))
    nn, scores = list(map(int, nn)), list(map(float, scores))
    return nn, scores


class FileTextIndex(BaseTextIndex):
    """
    Basic TextIndex using files to store the data. Both index text and queries
    will be stored in compressed text files.
    """
    __text__ = 'text.zip'
    __meta__ = 'meta.zip'

    def __init__(self, name):
        self.name = name
        self.text = []
        self.meta = []

    def __len__(self):
        return len(self.text)

    @classmethod
    def load(cls, name, tar):
        inst = cls(name)
        text = gzip.open(tar.extractfile(cls.__text__)).read().decode()
        inst.text = text.strip().split('\n')
        inst.meta = json.loads(gzip.open(tar.extractfile(cls.__meta__)).read().decode())

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

        tar.add(textname, arcname=FileTextIndex.__text__)
        tar.add(metaname, arcname=FileTextIndex.__meta__)

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
        query_file = utils.ensure_ext(query_name, 'tsv')
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
            header = '#{}\n'.format(self.name)
            paths_ = list(map(lambda x: x[0], sorted(paths_.items(), key=lambda x: x[1])))
            header += '#{}\n'.format('\t'.join(paths_))
            f.seek(0)
            f.write(header)

    def get_indexed_text(self, text_id):
        return self.text[text_id], self.meta[text_id]

    def inspect_query(self, query_name, threshold, max_NNs, by_source=False, **kwargs):
        """
        Inspect query from file
        """
        if by_source:
            logging.warn("Ignoring `by_source`. {} doesn't support it."
                         .format(type(self).__name__))

        if not os.path.isfile(utils.ensure_ext(query_name, 'tsv')):
            raise ValueError("Query doesn't exist: {}".format(query_name))

        with open(utils.ensure_ext(query_name, 'tsv'), 'r') as f:

            # validate query
            name = next(f).strip()[1:]
            if name != self.name:
                raise ValueError("Seems like this query doesn't correspond "
                                 "to current indexer: {} != {}"
                                 .format(name, self.name))

            # extract path encoding
            paths_ = next(f).strip()[1:].split('\t')

            for line in f:
                target, path_, num, scores = line.strip().split('\t')
                tmeta = {'path': paths_[int(path_)], 'num': int(num)}
                nns, scores = _parse_scores(scores)

                # check threshold
                if max(scores) < threshold:
                    continue

                match = {'target': target, 'meta': tmeta, 'matches': []}

                # package
                for nn, score in utils.take(zip(nns, scores), max_NNs):
                    if score < threshold:
                        break

                    match['matches'].append(
                        {'source': self.text[nn],
                         'meta': self.meta[nn],
                         'score': score})

                yield match

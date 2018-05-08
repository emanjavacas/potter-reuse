
import json
import itertools
from contextlib import contextmanager

from sqlalchemy import create_engine, Column, types, TypeDecorator
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import and_

from text_indexer import BaseTextIndex
import utils

# sqlalchemy singletons
Base = declarative_base()

def get_dict(self):
    d = dict(self.__dict__)
    d.pop('_sa_instance_state', None)
    return d

setattr(Base, "get_dict", get_dict)

engine = create_engine('sqlite:///text_index.db')
Session = sessionmaker(bind=engine)


@contextmanager
def session_scope():
    session = Session()

    try:
        yield session
        session.commit()

    except:
        session.rollback()
        raise

    finally:
        session.close()


class JSONEncodedDict(TypeDecorator):
    "Represents an immutable structure as a json-encoded string."

    impl = types.String

    def process_bind_param(self, value, dialect):
        if value is not None:
            value = json.dumps(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            value = json.loads(value)
        return value


class Source(Base):
    __tablename__ = 'TextIndex'

    id = Column(types.String, primary_key=True)  # {index_id.row}
    index_id = Column(types.String)   # unique identifier for the index
    row = Column(types.Integer)       # text entry id (int mapping to faiss rows)
    text = Column(types.UnicodeText)  # text entry being indexed
    path = Column(types.String)       # path to file where the text comes from
    num = Column(types.Integer)       # line number
    meta = Column(JSONEncodedDict)    # extra metadata
    date = Column(types.Date)         # just some timestamp

    def __repr__(self):
        return '<Source row="{}" text="{}" location="{}:{}">'.format(
            self.row,
            self.text[:20] + '...' if len(self.text) > 20 else self.text,
            self.path,
            self.num)


class Query(Base):
    __tablename__ = 'QueryIndex'

    id = Column(types.Integer, primary_key=True)
    index_id = Column(types.String)   # same
    query_id = Column(types.String)   # unique identifier of the query
    text = Column(types.UnicodeText)  # text entry being queried
    source = Column(types.Integer)    # int mapping to faiss row of the source text
    score = Column(types.Float)       # score given to the match
    path = Column(types.String)       # path to file where query is coming from
    num = Column(types.Integer)       # line number
    meta = Column(JSONEncodedDict)    # extra metadata
    date = Column(types.Date)         # just some timestamp

    def __repr__(self):
        return '<Query match="{}" score="{:g}" text="{}" location="{}:{}">'.format(
            self.source,
            self.score,
            self.text[:20] + '...' if len(self.text) > 20 else self.text,
            self.path,
            self.num)


#Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)  # ensure tables exist


class RelationalTextIndex(BaseTextIndex):
    def __init__(self, name, _from_file=False):
        self.name = name

        if '.' in name:
            raise ValueError(
                "RelationalTextIndex doesn't support \"'\" in the index name: [{}]"
                .format(name))

        # check if index exists
        if len(self) != 0 and not _from_file:
            raise ValueError("Index [{}] already exists! Load it from file".format(name))


    def __len__(self):
        with session_scope() as session:
            return session.query(Source).filter(Source.index_id == self.name).count()

    @classmethod
    def load(cls, name, tar):
        return cls(name, _from_file=True)

    def add(self, text, meta):
        with session_scope() as session:
            if isinstance(text, str):
                text, meta = [text], [meta]

            for t, m in zip(text, meta):
                session.add(Source(
                    id='{}.{}'.format(self.name, m['id']),
                    row=m['id'],
                    index_id=self.name,
                    text=t,
                    path=m['path'],
                    num=m['num']))

    def dump_query(self, query_name, generator):
        with session_scope() as session:
            session.bulk_insert_mappings(
                Query,
                [
                    {'index_id': self.name,
                     'query_id': query_name,
                     'text': target,
                     'source': source,
                     'score': score,
                     'path': meta['path'],
                     'num': meta['num']}
                    for target, meta, sources, scores in generator
                    for source, score in zip(sources, scores)
                ]
            )
        # # this code would be just a little bit faster
        # engine.execute(Query.__table__.insert(),
        #     [
        #         {'index_id': self.name,
        #          'query_id': query_name,
        #          'text': target,
        #          'source': source,
        #          'score': score,
        #          'path': meta['path'],
        #          'num': meta['num']}
        #         for target, meta, sources, scores in generator
        #         for source, score in zip(sources, scores)
        #     ]
        # )

    def get_indexed_text(self, text_id):
        with session_scope() as session:
            match = session.query(Source).get(text_id)
            return match.text, {'path': match.path, 'num': match.num}

    def inspect_query(self, query_name, threshold, max_NNs):
        # TODO: RelationalTextIndex can do actual search based on further metadata
        with session_scope() as session:
            query = session \
                .query(Query) \
                .filter(and_(Query.query_id == query_name,
                             Query.score >= threshold)) \
                .order_by(Query.score.desc()) \
                .order_by(Query.source)

            for _, matches in itertools.groupby(query, lambda m: m.source):
                sid, path, num = matches[0].source
                source = session.query(Source).get('{}.{}'.format(self.name, sid))

                output = {'source': source.text,
                          'meta': {'path': source.path, 'num': source.num},
                          'matches': []}

                for match in utils.take(matches, max_NNs):
                    output['matches'].append(
                        {'target': match.text,
                         'meta': {'path': match.path, 'num': match.num},
                         'score': match.score})

                yield output

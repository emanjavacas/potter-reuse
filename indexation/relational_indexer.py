
import json
import itertools
from contextlib import contextmanager

from sqlalchemy import create_engine, Column, types, TypeDecorator, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

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
    __tablename__ = 'Source'

    id = Column(types.Integer, primary_key=True)
    index_id = Column(types.String)   # unique identifier for the index
    row = Column(types.Integer)       # text entry id (int mapping to faiss rows)
    text = Column(types.UnicodeText)  # text entry being indexed
    path = Column(types.String)       # path to file where the text comes from
    num = Column(types.Integer)       # line number
    meta = Column(JSONEncodedDict)    # extra metadata
    date = Column(types.Date)         # just some timestamp


class Query(Base):
    __tablename__ = 'Query'

    id = Column(types.Integer, primary_key=True)
    index_id = Column(types.String)   # same
    query_id = Column(types.String)   # unique identifier of the query
    text = Column(types.UnicodeText)  # text entry being queried
    path = Column(types.String)       # path to file where query is coming from
    num = Column(types.Integer)       # line number
    meta = Column(JSONEncodedDict)    # extra metadata
    date = Column(types.Date)         # just some timestamp
    matches = relationship("Match")


class Match(Base):
    __tablename__ = 'Match'

    id = Column(types.Integer, primary_key=True)
    source = Column(types.Integer)    # int mapping to faiss row of the source text
    score = Column(types.Float)       # score given to the match
    query_id = Column(types.Integer, ForeignKey("Query.id"))


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
                    row=m['id'],
                    index_id=self.name,
                    text=t,
                    path=m['path'],
                    num=m['num']))

    def dump_query(self, query_name, generator):
        for chunk in utils.chunks(generator, 5000):  # transact every n items
            with session_scope() as session:
                for target, meta, sources, scores in chunk:
                    q = Query(index_id=self.name,
                              query_id=query_name,
                              text=target,
                              path=meta['path'],
                              num=meta['num'])

                    for source, score in zip(sources, scores):
                        m = Match(source=source, score=score)
                        q.matches.append(m)
                        session.add(m)

                    session.add(q)

    def get_indexed_text(self, text_id):
        with session_scope() as session:
            match = session.query(Source) \
                           .filter_by(index_id=self.name, row=text_id) \
                           .first()

            return match.text, {'path': match.path, 'num': match.num}

    def inspect_query(self, query_name, threshold, max_NNs, by_source=False):
        session = Session()     # no need to create session scope
        query = session.query(Query) \
                       .filter(Query.query_id==query_name) \
                       .join(Query.matches, aliased=True) \
                       .filter(Match.score >= float(threshold))

        for q in query.yield_per(100):  # paginate over 100 items
            match = {'target': q.text,
                     'meta': {'path': q.path, 'num': q.num},
                     'matches': []}

            for m in session.query(Match).filter_by(query_id=q.id).all():
                if m.score >= threshold:
                    source, smeta = self.get_indexed_text(m.source)
                    match['matches'].append(
                        {'source': source, 'score': m.score, 'meta': smeta})

            yield match

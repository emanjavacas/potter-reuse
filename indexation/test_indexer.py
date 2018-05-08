
import lorem
import numpy as np

from indexation import Indexer
from relational_indexer import RelationalTextIndex


DIM=100


def generate_text(path, nlines):
    with open(path, 'w+') as f:
        for _ in range(nlines):
            f.write('{}\n'.format(lorem.sentence()))


def encoder(lines):
    return np.random.randn(len(lines), DIM)


index_file = '/tmp/index.file'
query_files = ['/tmp/query.file.{}'.format(i) for i in range(10)]
generate_text(index_file, 1000)
for f in query_files:
    generate_text(f, 1000)

indexer = Indexer(DIM,
                  make_text_index=RelationalTextIndex
)
import time
start = time.time()
indexer.index_files(encoder, index_file)
print("Indexing took: {:g}".format(time.time() - start))
start = time.time()
indexer.query_from_files(encoder, 'random_query', *query_files)
print("Querying took: {:g}".format(time.time() - start))
start = time.time()
indexer.serialize('random.index')
print("Serializing took: {:g}".format(time.time() - start))
start = time.time()
i2=Indexer.load("random.index.tar")
print("Loading took: {:g}".format(time.time() - start))


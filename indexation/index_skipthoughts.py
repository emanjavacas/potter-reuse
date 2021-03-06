
import os
import argparse
import sys

import numpy as np

from indexation import Indexer


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('action')
    parser.add_argument('--indexer', required=True, help="Name for the indexer. "
                        "It will be used to store/load indexer to/from disk.")
    parser.add_argument('--index_files', nargs='*')
    parser.add_argument('--bsize', type=int, default=10000, help="Buffer size")
    parser.add_argument('--NNs', type=int, default=5)
    parser.add_argument('--dim', type=int, default=4800)
    parser.add_argument('--query_files', nargs='*')
    parser.add_argument('--projection')
    parser.add_argument('--query_name')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    actions = set(args.action.lower().split())

    projection = None
    if os.path.isfile(args.projection or ''):
        np.load(args.projection)

    encoder, model = None, None
    if 'index' in actions or 'query' in actions:
        print("Loading model...")
        import skipthoughts
        model = skipthoughts.load_model()

        def encoder(sents):
            embs = np.array(skipthoughts.encode(model, sents, use_norm=False))
            if projection is not None:
                embs = np.einsum('ij,bj->bj', projection, embs)
            return embs

    indexer = None

    if 'index' in actions:
        print("Indexing...")

        try:
            indexer = Indexer.load(args.indexer)
        except:
            pass

        do_serialize = indexer is None and args.indexer

        if len(args.index_files) == 0:
            raise ValueError("Indexing requires `index_files`")

        indexer = indexer or Indexer(dim=args.dim, name=args.indexer)
        indexer.index_files(encoder, *args.index_files, verbose=True)
        if do_serialize:
            print("Serializing index...")
            indexer.serialize(args.indexer)

    if 'query' in actions:
        print("Querying...")

        if indexer is None:
            indexer = Indexer.load(args.indexer)
            raise ValueError("Couldn't initialize indexer")
        if not args.query_name:
            raise ValueError("Querying requires `query_name`")
        if len(args.query_files) == 0:
            raise ValueError("Querying requires `query_files`")

        indexer.query_from_files(
            encoder, args.query_name, *args.query_files, NNs=args.NNs, verbose=True)

    if 'inspect' in actions:
        indexer = indexer or Indexer.load(args.indexer, inspect_only=True)

        if not args.query_name:
            raise ValueError("Inspect requires `query_name`")

        matches = indexer.inspect_query(
            args.query_name, threshold=args.threshold, max_NNs=args.NNs)

        while True:
            try:
                match = next(matches)
                print(" => {}\n".format(match['target']), flush=True)
                for m in match['matches']:
                    print(" *** [{:.3f}] {}".format(
                        m['score'], m['source']), flush=True)
                print(flush=True)
    
            except BrokenPipeError:
                print("Bye!", file=sys.stderr)
                break
    
            except StopIteration:
                print("Finished!", file=sys.stderr)
                break

    sys.stderr.close()

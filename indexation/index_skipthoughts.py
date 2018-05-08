
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

    print("Loading model...")
    projection = None
    if os.path.isfile(args.projection or ''):
        np.load(args.projection)

    encoder, model = None, None
    if 'index' in actions or 'query' in actions:
        import skipthoughts
        model = skipthoughts.load_model()

        def encoder(sents):
            embs = np.array(skipthoughts.encode(model, sents, use_norm=False))
            if projection is not None:
                embs = np.einsum('ij,bj->bj', projection, embs)
            return embs

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

        indexer = indexer or Indexer(dim=args.dim, name=args.indexer)
        indexer.index_files(encoder, *args.index_files, verbose=True)
        if do_serialize:
            print("Serializing index...")
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
            encoder, args.query_name, *args.query_files, NNs=args.NNs, verbose=True)

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

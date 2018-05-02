
from nltk.tokenize import word_tokenize
import sys


def read_index(path):
    with open(path, 'r') as f:
        return {idx: line.strip() for idx, line in enumerate(f)}


def take(it, n):
    tot = 0
    for i in it:
        if tot + 1 > n:
            break
        yield i
        tot += 1


def parse_sims(sims):
    _, *sims = sims.split('\t')
    nn, sims = zip(*map(lambda nn: nn.split(':'), sims))
    nn, sims = list(map(int, nn)), list(map(float, sims))
    return nn, sims


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_result')
    parser.add_argument('--index_file')
    parser.add_argument('--min_len', type=int, default=1)
    parser.add_argument('--NNs', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    result = args.query_result
    query = '.'.join(result.split('.')[:-1])  # test.txt.results -> test.txt
    index = read_index(args.index_file)

    with open(query, 'r') as q, open(result, 'r') as r:
        for line, sims in zip(q, r):

            # skip short sentences
            if len(word_tokenize(line)) < args.min_len:
                continue

            try:
                line, (NNs, sims) = line.strip(), parse_sims(sims.strip())
                if max(sims) < args.threshold:
                    continue

                NNs_str = ''
                for nn, sim in take(zip(NNs, sims), args.NNs):
                    if sim < args.threshold:
                        continue
                    NNs_str += " *** [{:.3f}] {}\n".format(sim, index[nn])

                if NNs_str:
                    # print reference
                    print(" => {}".format(line), flush=True)
                    # print NNs
                    print(NNs_str, flush=True)

                            
            except BrokenPipeError:
                print('Bye', file=sys.stderr)
                break

    sys.stderr.close()

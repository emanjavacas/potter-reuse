
import os

def read_books(path='/home/manjavacas/corpora/potterus/'):
    for f in os.listdir(path):
        p = os.path.join(path, f)
        with open(p, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


if __name__ == '__main__':
    model, tokenize, tokenizer = 'ucto', True, None

    if tokenize:
        import tokenizer
        tokenizer = tokenizer.tokenizer(model)
        fname = 'potterus_{}.txt'.format(model)
    else:
        fname = 'potterus.raw.txt'

    with open(fname, 'w+') as f:
        for line in read_books():
            if tokenize:
                for subline in tokenizer(line):
                    f.write(subline)
                    f.write("\n")
            else:
                f.write(line)
                f.write("\n")


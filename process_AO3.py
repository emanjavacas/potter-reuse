
import csv
import sys
csv.field_size_limit(sys.maxsize)


def read_csv(path='/home/manjavacas/corpora/AO3/en_fanfic.csv'):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for line in reader:
            yield dict(zip(header, line))


if __name__ == '__main__':
    import tokenizer

    model, tokenize = "ucto", True

    if tokenize:
        tokenizer = tokenizer.tokenizer(model=model)
        fname = 'AO3_{}.txt'.format(model)
    else:
        fname = 'AO3.raw.txt'

    with open(fname, 'w+') as f:
        for entry in read_csv():
            for line in entry.get('body', '').split('\n'):
                line = line.strip()
                if line:
                    if tokenize:
                        for subline in tokenizer(line):
                            f.write(subline)
                            f.write("\n")
                    else:
                        f.write(line)
                        f.write("\n")

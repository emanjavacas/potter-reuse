
import re
import os


def is_timestamp(line):
    return re.match(r"[0-9:,]+ --> [0-9:,]+")


def read_lines(path):
    with open(path, 'r') as f:
        next(f)                 # skip number
        next(f)                 # skip timestamp
        sub = []
        for line in f:
            line = line.strip()
            if not line:
                yield ' '.join(sub)
                sub = []
                next(f)       # skip number
                next(f)       # skip timestamp
            else:
                sub.append(line)

def preprocess(lines):
    # . . . => ...
    def join_dots_subline(lines):
        for line in lines:
            yield re.sub('\. \. \.', '...', line)

    # since most of it is dialogue, quotes are rather misleading and making sentence
    # tokenization unnecessarily harder, ergo => dropped
    def drop_quotes(lines):
        for line in lines:
            yield line.replace('"', "")

    for line in drop_quotes(join_dots_subline(lines)):
        yield line


def postprocess(lines, tokenizer=None, calls=0):
    # remove xml
    def remove_xml(lines):
        for line in lines:
            if line.startswith('<font'):
                continue
            line = re.sub('</? ?[iI] ?>', '', line)
            yield line

    # ... plus ...
    def remove_dash(lines):
        for line in lines:
            yield re.sub('^[ ]*-[ ]+', '', line)

    # ... ... => ' '
    def remove_joint_dots(lines):
        for line in lines:
            yield re.sub('[.]{3,3} [.]{3,3}', ' ', line)

    # remove -
    def join_dots(lines):
        last = None
        for line in lines:
            if line.endswith('...') and not line.endswith('....'):
                if last:
                    if not line.startswith('...'):
                        print("Consecutive trailing dots", last + '---' + line)
                        yield last
                        yield line
                        last = None
                    else:
                        last = last[:-3] + ' ' + line[3:]
                else:
                    last = line
            elif line.startswith('...'):
                if last:
                    yield last[:-3] + ' ' + line[3:]
                    last = None
                else:
                    yield line
                    last = None
            else:
                yield line

    for line in remove_joint_dots(remove_xml(remove_dash(join_dots(lines)))):
        yield line


def read_subtitles(path, tokenizer):
    for p in os.listdir(path):
        preprocessed = preprocess(read_lines(os.path.join(path, p)))
        # disable quote detection
        lines = tokenizer(' '.join(preprocessed), ignore_quotes=True)
        fname = os.path.basename(os.path.join(path, p))
        yield (fname, list(postprocess(lines, tokenizer)))


if __name__ == '__main__':
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ucto')
    parser.add_argument('--root', default='/home/manjavacas/corpora/potterus/subtitles/')
    args = parser.parse_args()

    from process_utils import tokenizer, package_tar
    tokenizer = tokenizer(args.model)
    fname = 'subtitles_{}.tar.gz'.format(args.model)
    package_tar(fname, read_subtitles(args.root, tokenizer), None)

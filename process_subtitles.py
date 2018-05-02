
import re


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
            if '. . .' in line:
                line = re.sub('. . .', '...', line)
            yield line

    for line in join_dots_subline(lines):
        yield line


def postprocess(lines):
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
            if line.startswith('-'):
                yield line[2:]
            else:
                yield line

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


def get_subtitle_lines(path, model):
    from tokenizer import tokenizer
    tokenizer = tokenizer(model)
    lines = tokenizer(' '.join(preprocess(read_lines(path))))
    for line in postprocess(lines):
        yield line


if __name__ == '__main__':
    import os
    ROOT = './data/subtitles'
    model = 'ucto'
    with open('./data/subtitles_{}.txt'.format(model), 'w+') as f:
        for path in os.listdir(ROOT):
            print("Processing {}".format(os.path.join(ROOT, path)))
            for line in get_subtitle_lines(os.path.join(ROOT, path), model):
                f.write(line)
                f.write('\n')

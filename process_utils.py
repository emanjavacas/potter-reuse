
import tarfile
import io


def tokenizer(model='spacy'):
    tok = None

    if model == 'ucto':
        import ucto
        tok = ucto.Tokenizer(
            "tokconfig-eng",
            paragraphdetection=False,
            quotedetection=True)

        def to_lines(line, ignore_quotes=False):
            tok.process(line)
            lines, line, quote = [], '', False
            for token in tok:
                if token.nospace():
                    line += str(token)
                else:
                    line += '{} '.format(str(token))
                if token.isbeginofquote() and not ignore_quotes:
                    quote = True
                if token.isendofquote() and not ignore_quotes:
                    quote = False
                if token.iseos() and not quote:
                    lines.append(line.strip())
                    line = ''

            return lines

    elif model == 'nltk':
        from nltk.tokenize import sent_tokenize as tok

        def to_lines(line):
            return [line for line in tok(line)]

    elif model == 'spacy':
        import spacy
        nlp = spacy.load('en', disable=['parser', 'ner', 'tagger'])

        def to_lines(line):
            doc = nlp(line)
            return [sent.string.strip() for sent in doc.sents]

    else:
        raise ValueError("Unknown model: [{}]".format(model))

    return to_lines


def package_tar(tarname, it, tokenizer):
    fnames = set()

    with tarfile.open(tarname, 'w:gz') as tar:
        for fname, lines in it:

            if fname in fnames:
                print("Duplicate file: {}".format(fname))
                continue
            fnames.add(fname)

            if tokenizer is not None:
                lines = tokenizer('\n'.join(lines))
            lines = list(lines)

            print("Adding #{} lines to file {}".format(len(lines), fname))
            tarinfo = tarfile.TarInfo(fname)
            lines = '\n'.join(lines).encode()
            tarinfo.size = len(lines)  # length in bytes
            tar.addfile(tarinfo, io.BytesIO(lines))

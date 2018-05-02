

def tokenizer(model='spacy'):
    tok = None

    if model == 'ucto':
        import ucto
        tok = ucto.Tokenizer(
            "tokconfig-eng",
            paragraphdetection=False,
            quotedetection=True)

        def to_lines(line):
            tok.process(line)
            lines = []
            line = ''
            quote = False
            for token in tok:
                if token.nospace():
                    line += str(token)
                else:
                    line += '{} '.format(str(token))
                if token.isbeginofquote():
                    quote = True
                if token.isendofquote():
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

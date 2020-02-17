def clean_test(text, hint=None, do_lower=False):
    '''
    Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution
    '''
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&' + "\n\t"

    def clean_special_chars(text, punct):
        for p in punct:
            text = text.replace(p, ' ' + p + ' ')
        return text

    def intonation(text, hint, do_lower):
        new_text = []
        for token in text.split():
            if do_lower:
                new_text.append(token.lower())
            else:
                new_text.append(token)
            if hint and token.isupper() and len(token) > 1:
                new_text.append(hint)
        return " ".join(new_text)

    # data = data.astype(str).apply(lambda x: clean_special_chars(intonation(x, hint, do_lower), punct))
    return clean_special_chars(intonation(text, hint, do_lower), punct)
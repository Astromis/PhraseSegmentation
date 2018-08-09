from nltk.tokenize.texttiling import TextTilingTokenizer
from nltk.corpus import stopwords

# https://www.nltk.org/api/nltk.tokenize.html#nltk.tokenize.texttiling.TextTilingTokenizer
# http://amor.cms.hu-berlin.de/~robertsw/files/TextTilingSlides.pdf
#tt = TextTilingTokenizer(stopwords=stopwords.words('russian'))

def demo(text=None):
    from nltk.corpus import brown
    from matplotlib import pylab
    tt = TextTilingTokenizer(demo_mode=True)
    if text is None:
        text = brown.raw()[:10000]
    s, ss, d, b = tt.tokenize(text)
    pylab.xlabel("Sentence Gap index")
    pylab.ylabel("Gap Scores")
    pylab.plot(range(len(s)), s, label="Gap Scores")
    pylab.plot(range(len(ss)), ss, label="Smoothed Gap scores")
    pylab.plot(range(len(d)), d, label="Depth scores")
    pylab.stem(range(len(b)), b)
    pylab.legend()
    pylab.show()
    
demo()

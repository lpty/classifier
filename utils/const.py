import string
from zhon import hanzi


class Punctuations(object):
    ZHPunc = hanzi.punctuation
    ENPunc = string.punctuation
    CHARPunc = 'qwertyuioplkjhgfdsazxcvbnmQWERTYUIOPLKJHGFDSAZXCVBNM'
    NUMPunc = '1234567890'
    ANUMPunc = '１２３４５６７８９０'
    SPACEPunc = ' '
    ROMEPunc = 'ⅠⅡⅢⅣⅤⅥⅦⅧⅨ'
    BEPunc = ['BOS', 'EOS']
    PUNCTUATIONS = "".join(set(
        ENPunc + ZHPunc + u''':!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻︽︿﹁﹃﹙﹛﹝（｛“‘-—_…'''))
    AllPunc = "".join(set(
        NUMPunc + ANUMPunc + ROMEPunc + CHARPunc + SPACEPunc + ENPunc + ZHPunc + u''':!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻︽︿﹁﹃﹙﹛﹝（｛“‘-—_…'''))

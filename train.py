# -*- coding: utf-8 -*-

from gensim.models import word2vec
import logging

def main():

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus("out_54new.txt")
    model = word2vec.Word2Vec(sentences, size=260)
    model.save('new_out51.model')

if __name__ == "__main__":
    main()

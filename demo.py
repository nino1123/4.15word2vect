# -*- coding: utf-8 -*-
from gensim.models import word2vec
from gensim import models
import logging

def main():
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	
	# model = models.Word2Vec.load_word2vec_format('wiki.zh.model.bin',binary=True)
	model = word2vec.Word2Vec.load('new_out51.model')

	query = ("人民")
	q_list = query.split()
	try:
		if len(q_list) == 1:
			print("相似词前 20 排序")
			res = model.most_similar(q_list[0], topn = 20)
			for item in res:
				print(item[0]+","+str(item[1]))

		elif len(q_list) == 2:
			print("计算 Cosine 相似度")
			res = model.similarity(q_list[0], q_list[1])
			print(res)

		else:
			print("%s之于%s，如%s之于" % (q_list[0], q_list[2], q_list[1]))
			res = model.most_similar([q_list[0], q_list[1]], [q_list[2]], topn = 20)
			for item in res:
				print(item[0]+","+str(item[1]))

		print("----------------------------")
	except Exception as e:
		print(repr(e))
		
		
if __name__ == "__main__":
	main()

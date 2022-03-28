import numpy as np
import sys

filetype = sys.argv[1]
fname = './data/'+filetype+'.txt'
saved = './data/bi_'+filetype+'.txt'
label = []
with open(fname) as samples, open(saved, mode = 'w', encoding = 'utf-8') as bi:
	for line in samples.readlines():
		line.strip()
		img, label = line.split('\t')
		img = [1 if int(img_item) > 0 else 0 for img_item in img.split(' ')]
		img = [str(img_item) for img_item in img]
		img = ' '.join(img)
		bi.write(img+'\t'+label)
samples.close()
bi.close()

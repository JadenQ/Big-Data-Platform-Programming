import numpy as np

# fname = 'centroid_1229_all_15iters.txt'
# fname = 'centroid_1229_all_10iters.txt'
# fname = 'centroid_1001_all.txt'

fname = 'centroid_324_all.txt'

count = 0
iteration = 0
rec_count = {}
last_count = {}
with open(fname) as f:
	for line in f.readlines():
		count = count + 1
		line.strip()
		index, img_count = line.split(':')
		img, count = img_count.split(',')
		index = int(index[-1])
		count = int(count)
		cur_count = {index:count}
		rec_count.update(cur_count)
		if count % 10 == 0:
			iteration = iteration + 1
			if iteration == 1:
				last_count.update(rec_count)
				continue
			else:
				diff_count = {key: np.abs(last_count[key] - rec_count.get(key, 0)) for key in last_count}
				score = sum(diff_count.values()) / 60000
				print(score)
				last_count.update(rec_count)

print(diff_count)


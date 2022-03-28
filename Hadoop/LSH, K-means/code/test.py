# # import random

# # random.seed(1229)
# # img = [str(random.randint(0, 255)) for k in range(28*28)]
# # img = ' '.join(img)
# # print(img)

# # random_img = [[random.randint(0, 225) for k in range(28*28)] for i in range(10)]
# # print(random_img)

# # bb = '0 1 2 3 4'
# # bbs = [int(item) for item in bb.split(' ')]
# # print(bbs)

# bb = [1,2,3,4,5]
# cc = [5,4,3,2,1]
# aa = '1 2 4 5 6'
# aa = [img_item for img_item in aa.split(' ')]
# dd = [bb[i]+cc[i] for i in range(len(bb))]
# print(' '.join(str(i) for i in dd))
# print([dd[i]/6 for i in range(len(dd))])

# print(' '.join(aa))
# import numpy as np
# centroid = np.zeros(28*28)
# # centroid = centroid.reshape(784,)
# print(centroid)
# count = 2
# new_centroid = [str(centroid[i] / count) for i in range(28*28)]
# # print(new_centroid)
# import random
# import numpy as np
# def save_centroid(centroid_file, centroids, counts):
# 	with open(centroid_file, mode = 'w', encoding = 'utf-8') as cent:
# 		for i in range(0, 10):
# 			cent_i = 'Centroid ' + str(i) + ':'
# 			img = [str(centroids[i][j]) for j in range(28*28)]
# 			img = ' '.join(img)
# 			cent_i = cent_i + img + ',' + str(counts[i]) + '\n'
# 			cent.write(cent_i)
# 	cent.close()

# centroid_file = "test.txt"
# # centroids = np.zeros((10, 28 * 28))
# # counts = np.zeros((10, 1))

# random_img = [[random.randint(0, 225) for j in range(28*28)] for i in range(10)]
# # intialize as 0
# counts = [0 for i in range(10)]
# save_centroid(centroid_file, random_img, counts)
# from collection import Counter
import numpy as np
aa = {1:0, 2:9, 3:8}
bb = {1:7, 2:6, 3:9}

d = {key: np.abs(aa[key] - bb.get(key, 0)) for key in aa}
print(d)
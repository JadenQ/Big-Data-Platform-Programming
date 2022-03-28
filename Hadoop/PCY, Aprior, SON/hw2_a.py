import sys
import time
start_time = time.time()

fname = sys.argv[1] # input file
basket_num = 0 # num of baskets
word_count = {} # key: word (individual item), value: frequency
freq_item = {} # key:word (frequent individual item), value: frequency
threshold = 0.005 # support threshold
freq_pairs = {} # key: item pair, value: frequency
topK = 40

# Pass1
with open(fname) as f:
	for line in f.readlines():
		basket_num += 1 # count the number of baskets
		line = line.strip()
		words = line.split(' ')
		words = list(set(words)) # remove the duplicates in one basket
		for word in words:
			if word not in word_count:
				word_count[word] = 1
			else: word_count[word] += 1


# s = threshold * basket_num
s = threshold * basket_num
for word in word_count:
	if word_count[word] >= s:
		freq_item[word] = word_count[word]

# Pass2

# get the candidate pair
with open(fname) as f:
	for line in f.readlines():
		line = line.strip()
		words = line.split(' ')
		words = list(set(words)) # remove the duplicates in one basket
		for i in range(0, len(words) - 1):
			for j in range(i+1, len(words)):
				if (words[i] in freq_item) and (words[j] in freq_item): # both elements are frequent
					# there should be no order within item pair
					if (words[i] <= words[j]):
						pair = words[i] + " " + words[j]
					else: pair = words[j] + " " + words[i]
					if pair in freq_pairs:
						freq_pairs[pair] += 1
					else: freq_pairs[pair] = 1

# get the frequent pair
result = {}
for pair in freq_pairs:
	if freq_pairs[pair] > s:
		result[pair] = freq_pairs[pair]

# sort the frequent item pairs according to frequency
if len(freq_pairs) > topK:
	result = sorted(result.items(), key = lambda x:x[1], reverse = True)[:topK]
	# sort according to value
else:
	result = sorted(result.items(), key = lambda x:x[1], reverse = True)

for pair in result:
	print(pair)

print("--- %s seconds ---" % (time.time() - start_time))
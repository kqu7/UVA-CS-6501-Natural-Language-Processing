#Name: Kai Qu 
#Computing ID: kq4ff
#Date: 10/10/2019
import numpy as np 
import scipy

#Scanning data and turning the word whose frequency is less than 2 into UNK
tag2num = {'A' : 1, 'C' : 2, 'D' : 3, 'M' : 4, 'N' : 5, 'O' : 6, 'P' : 7, 'R' : 8, 'V' : 9, 'W' : 10, 'Start' : 11, 'End': 12}
tag_counts = np.zeros((13))
trans_counts, trans_prob = np.zeros((13, 13)), np.zeros((13, 13))
emission_counts, emission_prob = np.zeros((13, 24510)), np.zeros((13, 24510))
file_trn = 'trn.pos'
word_count = {}
word_index = {}
cnt = 1
all_lines = []
with open(file_trn) as tn: 
	l = tn.readline().strip()
	while l: 
		arr = l.split(' ')
		for word in arr: 
			index = word.find('/')
			word = word[:index]
			if word not in word_count: 
				word_count[word] = 1
			else: 
				word_count[word] += 1
		all_lines.append(arr)
		l = tn.readline().strip()
tn.close()

for i in range(len(all_lines)): 
	for j in range(len(all_lines[i])):
		index = all_lines[i][j].find('/') 
		tag = all_lines[i][j][index + 1:]
		word = all_lines[i][j][:index]
		if word_count[word] < 2: 
			all_lines[i][j] = 'UNK/' + tag

for arr in all_lines:
		# arr = l.split(' ')
		for i in range(len(arr)): 
			idx = arr[i].find('/')
			word = arr[i][:idx]
			tag = arr[i][idx+1:]

			if word not in word_index: 
				word_index[word] = cnt
				cnt += 1
			emission_counts[tag2num[tag]][word_index[word]] += 1
			if i == 0:
				tag_counts[tag2num['Start']] += 1
				trans_counts[tag2num['Start']][tag2num[tag]] += 1
			else:
				if i == len(arr) - 1: 
					tag_counts[tag2num['End']] += 1
					trans_counts[tag2num[tag]][tag2num['End']] += 1
				prev_word = arr[i - 1]
				prev_tag = prev_word[prev_word.find('/') + 1:]
				trans_counts[tag2num[prev_tag]][tag2num[tag]] += 1
			tag_counts[tag2num[tag]] += 1

alpha = 100
beta = 0.1
for row in range(1, 13): 
	if row == 12: 
		continue
	for col in range(1, 13): 
		if col == 11: 
			continue
		trans_prob[row][col] = (trans_counts[row][col] + alpha) / (tag_counts[row] + alpha * (10 + 1))

# print(np.sum(trans_counts, axis=1))
# print(tag_counts)

for row in range(1, 11): 
	for col in range(1, cnt): 
		emission_prob[row][col] = (emission_counts[row][col] + beta) / (tag_counts[row] + cnt * beta)


print('check transition probabilities')
print(np.sum(trans_prob, axis = 1))

print('check emission probabilities')
print(np.sum(emission_prob, axis = 1))

#Implementation of Viberti Algorithm
def Viberti(obs, states, trans_prob, emit_prob): 
	V = [{}]
	p = {}
	ret = []
	# print(obs)
	# print(states)
	for y in range(1, len(states) + 1): 
		V[0][y] = trans_prob[11][y] + emit_prob[y][int(obs[0])]
		p[y] = [y]

	for m in range(1, len(obs)): 
		V.append({})
		np = {}
		for y in states: 
			(prob, state) = max([(V[m-1][y0] + trans_prob[y0][y] + emit_prob[y][int(obs[m])], y0) for y0 in states])
			V[m][y] = prob
			np[y] = p[state] + [y]
		p = np
	prob, state = max([(V[len(obs) - 1][y0] + trans_prob[y0][12], y0) for y0 in states])
	return prob, p[state]

#Dev accuracy
count_correct, count_total = 0, 0
dev_file = 'dev.pos'
states = [i for i in range(1, 10 + 1)]
words_dev = []
correct_tags_dev = []
with open(dev_file) as dv: 
	l = dv.readline().strip()
	while l:
		arr = l.split(' ')
		correct_tags = np.zeros(len(arr))
		obs = np.zeros(len(arr))
		for i in range(len(arr)):
			index = arr[i].find('/')
			correct_tags[i] = tag2num[arr[i][index+1:]]
			if arr[i][:index] not in word_index: 
				obs[i] = word_index['UNK']
			else:
				obs[i] = word_index[arr[i][:index]]
		words_dev.append(obs)
		correct_tags_dev.append(correct_tags)
		prob, state = Viberti(obs, states, np.log(trans_prob), np.log(emission_prob))
		for i in range(len(arr)): 
			if state[i] == correct_tags[i]: 
				count_correct += 1
		count_total += len(arr)
		l = dv.readline().strip()
dv.close()

print('Dev Accuracy %5f\n' % (count_correct / count_total))


print('Chooseing alpha to be 100 and beta to be 0.1\n')

count_total = 0
count_correct = 0
tst_file = 'tst.pos'
with open(tst_file) as tst: 
	l = tst.readline().strip()
	while l:
		arr = l.split(' ')
		correct_tags = np.zeros(len(arr))
		obs = np.zeros(len(arr))
		for i in range(len(arr)):
			index = arr[i].find('/')
			correct_tags[i] = tag2num[arr[i][index+1:]]
			if arr[i][:index] not in word_index: 
				obs[i] = word_index['UNK']
			else:
				obs[i] = word_index[arr[i][:index]]
		# words_dev.append(obs)
		# correct_tags_dev.append(correct_tags)
		prob, state = Viberti(obs, states, np.log(trans_prob), np.log(emission_prob))
		for i in range(len(arr)): 
			if state[i] == correct_tags[i]: 
				count_correct += 1
		count_total += len(arr)
		l = tst.readline().strip()
tst.close()

print('Test Accuracy %5f' % (count_correct / count_total))









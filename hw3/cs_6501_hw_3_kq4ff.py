import numpy as np
# B: list of words, A: map, D: 2D bool array 
def parsing(B, A, D):
	S = []
	actions = []
	if len(B) <= 1: 
		return 'NON-PROJECTIVE'
	else:
		S.append(B.pop(0))
		S.append(B.pop(0))
		actions.append('SHIFT')
		actions.append('SHIFT')		
	flag = False
	while flag == False: 
		flag = True
		if len(S) > 1:
			right, left = S[-1], S[-2]
			# Update dependencies
			for d in D:
				if d[right] == 1: 
					d[right] = 0
				if d[left] == 1: 
					d[left] = 0
			if A[right] == left and np.sum(D[right]) == 0: 
				S.pop(-1); S.pop(-1)
				S.append(left)
				actions.append('RIGHTARC')
				flag = False
			elif A[left] == right and np.sum(D[left]) == 0:
				S.pop(-1); S.pop(-1)
				S.append(right)
				actions.append('LEFTARC')
				flag = False
		if len(B) > 0 and flag: 
			S.append(B.pop(0))
			actions.append('SHIFT')
			flag = False
	if len(S) == 1: 
		return actions
	else: 
		return 'NON-PROJECTIVE'

def read_data():
	f = open('data.conll', 'r')
	idxs, arcs, dpcys = [], [], []
	while True: 
		line = f.readline()
		if not line: break
		line = line.split(' ')
		if line[1] == 'sent_id': 
			continue
		elif line[1] == 'text': 
			l = len(line) - 3
			idx = []
			arc = {}
			cnt = {}
			dpcy = np.zeros((l + 1, l + 1))
			# To comsume tbe error annotation line
			line = f.readline()
			for i in range(l): 
				line = f.readline().split('\t')
				child, parent = int(line[0]), int(line[6])
				idx.append(child)
				arc[child] = parent
				dpcy[parent][child] = 1
			line = f.readline()
			idxs.append(idx); arcs.append(arc); dpcys.append(dpcy)
	return idxs, arcs, dpcys


if __name__ == '__main__': 
	idxs, arcs, dpcys = read_data()
	f = open('kq4ff-parsing-actions.txt', 'w')
	for i, (idx, arc, dpcy) in enumerate(zip(idxs, arcs, dpcys)):
		actions = parsing(idx, arc, dpcy)
		if type(actions) is str: 
			f.write(actions)
		else:
			for k in range(len(actions)):
				f.write(actions[k])
				if k != len(actions) - 1: 
					f.write(' ') 
		f.write('\n')

import matlab.engine as mateng
from time import time
import IPython
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble, lda, random_projection, preprocessing)
from scipy import linalg
import argparse
import glob

all_segments = {'1': [(176, 248), (496, 566), (739, 780), (977, 1075), (1176,1236), (1352,1405)], '2': [(585, 640), (796, 830), (1267, 1284)],
'3': [(695, 720), (857, 889), (1126, 1156), (1315, 1333), (1414, 1450)]}

color_map = {}
index_map = {}

def parse(PATH_TO_DATA):
	eng = mateng.start_matlab()

	[s, X] = eng.read_binary_blob(PATH_TO_DATA + '1.conv5b', nargout = 2)
	i = 17;
	while i <= 1489:
		[s, data] = eng.read_binary_blob(PATH_TO_DATA + str(i) + '.conv5b', nargout = 2)
		data = np.array(data)
		X = np.concatenate((X, data), axis = 0)
		i += 16
	return X

def parse_annotations():
	eng = mateng.start_matlab()

	[s, X] = eng.read_binary_blob(PATH_TO_DATA + '176.conv5b', nargout = 2)
	color = 'b'
	i = 0
	for category in all_segments:
		print "Parsing category " + category 
		segments = all_segments[category]
		if category == '2':
			color = 'r'
		elif category == '3':
			color = 'g'
		else:
			color = 'b'
		for seg in segments:
			j = seg[0]
			while j <= seg[1]:
				if j == 176:
					color_map[i] = color
					index_map[i] = j
					j += 16
					i += 1
					continue
				[s, data] = eng.read_binary_blob(PATH_TO_DATA + str(j) + '.conv5b', nargout = 2)
				data = np.array(data)
				X = np.concatenate((X, data), axis = 0)
				color_map[i] = color
				index_map[i] = j
				j += 16
				i += 1
	return X

#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, figure_name, title=None):
	x_min, x_max = np.min(X, 0), np.max(X, 0)
	X = (X - x_min) / (x_max - x_min)
	plt.figure()
	for i in range(X.shape[0]):
		plt.text(X[i, 0], X[i, 1], str(i*16 + 1), color=plt.cm.Set1((i*16 + 1)/ 100), fontdict={'weight': 'bold', 'size': 9})
	plt.xticks([]), plt.yticks([])
	if title is not None:
		plt.title(title)
	plt.savefig('plots/'+figure_name)


# def get_color(int x):
# 	if ( 176 <= x <= 365) or ( 464 <= x <= 566) or ( 739 <= x <= 780) or (977 <= x <= 1075) or (1176 <= x <= 1236) or (1352 <= x <= 1405):
# 		return 'r'
# 	else if ( 585 <= x <= 640) or (780 - 830) or (1267 <= x <= 1284):
# 		return 'b'
# 	else:
# 		return 'g'
# 	return 'y'
#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_annotated_embedding(X, figure_name, title=None):
	x_min, x_max = np.min(X, 0), np.max(X, 0)
	X = (X - x_min) / (x_max - x_min)
	plt.figure()
	r_x = r_y = []
	g_x = g_y = []
	b_x =  b_y = []
	for i in range(X.shape[0]):
		frm_num = index_map[i]
		color = color_map[i]
		if color == 'r':
			plt.text(X[i, 0], X[i, 1], 'o', color=color, fontdict={'weight': 'bold', 'size': 9})
		elif color == 'g':
			plt.text(X[i, 0], X[i, 1], 'x', color=color, fontdict={'weight': 'bold', 'size': 9})
		else:
			plt.text(X[i, 0], X[i, 1], '*', color=color, fontdict={'weight': 'bold', 'size': 13})
	# 	if color == 'r':
	# 		r_x.append(X[i, 0])
	# 		r_y.append(X[i, 1])
	# 	elif color == 'g':
	# 		g_x.append(X[i, 0])
	# 		g_y.append(X[i, 1])
	# 	else:
	# 		b_x.append(X[i, 0])
	# 		b_y.append(X[i, 1])
	# plt.plot(r_x, r_y, 'rs')
	# plt.plot(g_x, g_y, 'go')
	# plt.plot(b_x, b_y, 'b^')
	plt.xticks([]), plt.yticks([])
	if title is not None:
		plt.title(title)
	plt.savefig('plots/'+figure_name)

def pca(X):
	print("Computing PCA embedding")
	scaler = preprocessing.StandardScaler().fit(X)
	X_centered = scaler.transform(X)
	X_pca = decomposition.TruncatedSVD(n_components=40).fit_transform(X_centered)
	tsne = manifold.TSNE(init = 'pca', learning_rate = 100)
	X_tsne = tsne.fit_transform(X_pca)
	IPython.embed()
	return X_tsne

def tsne(figure_name, X):
	#----------------------------------------------------------------------
	# t-SNE embedding of the digits dataset
	print("Computing t-SNE embedding")
	tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
	X_tsne = tsne.fit_transform(X)
	return X_tsne

def plot_all():
	tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
	#----------------------------------------------------------------------
	# Pre-processing
	print "t-SNE Scaling"
	X_scaled = preprocessing.scale(X) #zero mean, unit variance
	IPython.embed()
	X_tsne_scaled = tsne.fit_transform(X_scaled)

	#normalize the data (scaling individual samples to have unit norm)
	print "t-SNE L2 Norm"
	X_normalized = preprocessing.normalize(X, norm='l2')
	X_tsne_norm = tsne.fit_transform(X_normalized)


	#whiten the data 
	print "t-SNE Whitening"
	scaler = preprocessing.StandardScaler(with_std=False).fit(X)
	X_centered = scaler.transform(X)

	IPython.embed()

	# U, s, Vh = linalg.svd(X_centered)
	shapeX = X_centered.shape
	IPython.embed()
	sig = (1/shapeX[0])*np.dot(X_centered, X_centered.T)
	U, s, Vh = linalg.svd(sig, full_matrices=False)
	eps = 1e-5
	invS = np.diag (np.reciprocal(np.sqrt(s+eps)))

	#PCA_whiten
	X_pca = np.dot(invS, np.dot(U.T, X_centered))
	X_tsne_pca = tsne.fit_transform(X_pca)

	#whiten the data (ZCA)
	X_zca = np.dot(U, X_pca)
	X_tsne_zca = tsne.fit_transform(X_zca)

	return X_tsne_scaled, X_tsne_norm, X_tsne_pca, X_tsne_zca

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("PATH_TO_DATA", help = "Please specify the path to data")
	args = parser.parse_args()
	X = parse_annotations(args.PATH_TO_DATA)
	# parser.add_argument("plot", help = "Choose from pca, tsne, etc.")
	# parser.add_argument("figure_name", help = "Figure name to be saved")
	# args = parser.parse_args()
	# plotter_func = locals()[args.plot]
	# plotter_func(args.figure_name, X)

	X_pca = pca(X)
	plot_annotated_embedding(X_pca, figure_name = 'pca_tsne_test.png', title = 't-SNE (PCA) conv5b')
	# X_tsne_scaled, X_tsne_norm, X_tsne_pca, X_tsne_zca  = plot_all()
	# plotNames = ["X_tsne_scaled", "X_tsne_norm", "X_tsne_pca", "X_tsne_zca"]

	# for plotName in plotNames:
	# 	plot_annotated_embedding (X=eval(plotName), figure_name='8fps_cropped_conv5b_'+plotName, title=plotName+'_conv5b')

# Methods to match MS1 peaks in a ms2lda object with rows in an mzmatch output file
import pandas as pd
import numpy as np
import pylab as plt
import math
import sys
import networkx as nx

class MS2MZM(object):
	def __init__(self):
		self.mzmatch = None
		self.mzmatch_file = None
		self.ms2lda = None
		self.mass_tol = 5.0
		self.rt_tol = 30.0

	def attach_mzfile(self,infile):
		self.mzmatch = pd.read_csv(infile,sep=',')
		# self.mzmatch.rename(columns={'Unnamed: 0':'mz','RT':'rt'},inplace=True)
		self.mzmatch.rename(columns={'Mass':'mz','RT (s)':'rt'},inplace=True)

		# print self.mzmatch

		self.mzmatch_file = infile

	def attach_ms2lda(self,ms2lda):
		self.ms2lda = ms2lda

	def find_links(self):
		# Loop over the ms1 peaks in the ms2lda object looking for peaks in the mzmatch

		# index
		# peakID
		# MSnParentPeakID
		# msLevel
		# rt
		# mz
		# intensity
		# Sample
		# GroupPeakMSn
		# CollisionEnergy
		# annotation
		proton_mass = 1.00727645199076
		self.matches = {}
		for row in self.ms2lda.ms1.itertuples(index=True):
			peakid = row[1]
			mz = row[5]
			rt = row[4]


			# # the following line is hacky for pos mode data
			# mz -= proton_mass
			mass_delta = mz*self.mass_tol*1e-6
			mass_start = mz-mass_delta
			mass_end = mz+mass_delta
			rt_start = rt-self.rt_tol
			rt_end = rt+self.rt_tol

			match_mass = (self.mzmatch.mz>mass_start) & (self.mzmatch.mz<mass_end)
			match_rt = (self.mzmatch.rt>rt_start) & (self.mzmatch.rt<rt_end)
			match = match_mass & match_rt & (self.mzmatch.Polarity == 'positive')

			res = self.mzmatch[match]
			if len(res) == 1:
				self.matches[row[0]] = res.index[0]
			elif len(res)>1:
				closest = None
				min_dist = sys.maxint
				for match_res in res.itertuples(index=True):
					match_rt = match_res[2]
					match_mz = match_res[1]
					dist = math.sqrt((match_rt-rt)**2 + (match_mz-mz)**2)
					if dist < min_dist:
						min_dist = dist
						closest = match_res
				self.matches[row[0]] = closest[0]

		# for match in self.matches:
		# 	print self.ms2lda.ms1.mz[match],self.mzmatch.mz[self.matches[match]],self.ms2lda.ms1.rt[match],self.mzmatch.rt[self.matches[match]]

	def create_special_nodes_ms1(self,case = 'M11',control = 'NDC6',special_nodes = []):
		case_cols = []
		control_cols = []
		for i in self.mzmatch:
			if i.startswith(case):
				case_cols.append(i)
			if i.startswith(control):
				control_cols.append(i)
		# Following lines define the colour range
		fold_min = -3
		fold_max = 3
		for i in self.ms2lda.ms1.index:
			if i in self.matches:
				mzindex = self.matches[i]
				v1 = self.mzmatch.loc[mzindex,case_cols]
				v2 = self.mzmatch.loc[mzindex,control_cols]

				SMALL = 1e-6

				for j,v in enumerate(v1):
					if np.isnan(v):
						v1[j] = SMALL
				for j,v in enumerate(v2):
					if np.isnan(v):
						v2[j] = SMALL

				# fold = self.mzmatch.loc[mzindex,case_cols].mean()/self.mzmatch.loc[mzindex,control_cols].mean()
				fold = v1.mean()/v2.mean()
				fold = np.log(fold)
				# if fold < 0:
				# 	sc = 1 - (fold - fold_min)/(np.abs(fold_min))
				# 	sc = sc * 256
				# 	if not math.isnan(sc):
				# 		h = hex(np.round(sc))
				# 		h = h[2:len(h)-1]
				# 		if len(h) == 1:
				# 			h = '0' + h
				# 		special_nodes.append(("doc_{}".format(i),"#00{}00".format(h)))
				# else:
				# 	sc = (fold)/((fold_max))
				# 	sc = sc * 255
				# 	if not math.isnan(sc):
				# 		h = hex(np.floor(sc))
				# 		h = h[2:len(h)-1]
				# 		if len(h) == 1:
				# 			h = '0' + h
				# 		special_nodes.append(("doc_{}".format(i),"#{}0000".format(h)))
				if fold < -1:
					special_nodes.append(("doc_{}".format(i), "#00FF00"))
				elif fold > 1:
					special_nodes.append(("doc_{}".format(i), "#FF0000"))
				else:
					special_nodes.append(("doc_{}".format(i), "#000000"))

		return special_nodes

	def get_annotation(self, pn, peak_annot):
		peakid = int(pn.split('_')[3])
		if peakid in peak_annot:
			annot = peak_annot[peakid][0]
			return annot
		else:
			return None

	def get_confidence(self, pn, peak_annot):
		peakid = int(pn.split('_')[3])
		if peakid in peak_annot:
			conf = peak_annot[peakid][1]
			return conf
		else:
			return None

	def compute_topic_t(self, case='M11', control='NDC6', special_nodes=[], t_thresh=4, t_annot={},
		G=None, selected=None, peak_annot={}, filename_label={}):

		n = 1
		topic_scores = []
		peak_scores = {}

		# select the case & control columns from the dataframe
		case_cols = []
		control_cols = []
		for i in self.mzmatch:
			if i.startswith(case):
				case_cols.append(i)
			if i.startswith(control):
				control_cols.append(i)

		# Create the graph if not provided
		if G is None:
			G, _ = self.ms2lda.get_network_graph(to_highlight=None, degree_filter=0)

		topic_nodes = []
		nodes = G.nodes(data=True)
		for node in nodes:
			node_data = node[1]
			if node_data['group'] == 2:
				topic_nodes.append(node)

		print "{} topic/cluster nodes loaded".format(len(topic_nodes))
		print

		topic_neighbours = {}
		for topic in topic_nodes:

			topic_data = topic[1]
			if selected is not None and 'motif' in topic_data['name']:
				topic_number = topic_data['name'].split('_')[1]
				if int(topic_number) not in selected:
					continue

			node_id = topic[0]
			node_data = topic[1]
			neighbours = G.neighbors(node_id)
			peaks = []
			mzindex = []
			peak_names = []
			for nb in neighbours:
				name = G.node[nb]['name']
				peakid = int(G.node[nb]['peakid'])
				peaks.append(peakid)
				if peakid in self.matches:
					mzindex.append(self.matches[peakid])
					peak_names.append(G.node[nb]['name'] + '_' + str(peakid))

			sub_mat = np.array(self.mzmatch.loc[mzindex,control_cols + case_cols])
			if len(sub_mat)==0:
				continue

			# sub_mat = np.log(sub_mat)
			r,c = sub_mat.shape
			SMALL = 1e-6
			for a in range(r):
				for b in range(c):
					if np.isnan(sub_mat[a,b]):
						sub_mat[a,b] = SMALL
			try:
				u,s,v = np.linalg.svd(sub_mat)
			except:
				continue

			columns = control_cols + case_cols
			temp = pd.DataFrame(v[0],index = columns).transpose()
			t_val = np.abs((temp.loc[0,case_cols].mean() - temp.loc[0,control_cols].mean()))
			t_val/=np.sqrt((temp.loc[0,control_cols].var())/(1.0*len(control_cols)) + (temp.loc[0,case_cols].var())/(1.0*len(case_cols)))

			# print annotated topic names
			t_name = topic[1]['name']
			t_number = int(t_name.split('_')[1]) # extract e.g. 123 from 'motif_123'
			t_new_name = 'Mass2Motif %d' % t_number
			if t_number in t_annot:
				t_annotated_name = "%s\n%s" % (t_new_name, t_annot[t_number])
			else:
				t_annotated_name = t_new_name

			# print annotated peak names
			print "%d. %s PLAGE score=%.3f" % (n, t_annotated_name, t_val)
			if len(peak_annot) > 0:
				sorted_peak_names = sorted(peak_names)
				for pn in sorted_peak_names:
					display_name = self.get_annotation(pn, peak_annot)
					if display_name is None:
						print " - %s" % pn
					else:
						print " - %-30s %s" % (pn, display_name)
			else:
				for pn in peak_names:
					print " - %s" % pn
			print

			if len(sub_mat)>1:
				topic_scores.append(t_val)
			sub_mat = np.log(sub_mat)
			rows,cols = sub_mat.shape
			for r in range(rows):
				m = sub_mat[r].mean()
				s = sub_mat[r].std()
				sub_mat[r,] -= m
				sub_mat[r,] /= s

			if t_val > t_thresh:

				special_nodes.append((topic[1]['name'],"#00e7eb"))

				annotated_peak_names = []
				annotated_confidences = []
				artefacts_pos = []
				for my_pos in range(len(peak_names)):
					pn = peak_names[my_pos]
					display_name = self.get_annotation(pn, peak_annot)
					conf = self.get_confidence(pn, peak_annot)
					if display_name is None:
						display_name = pn
					if display_name.lower() == 'deleted' or display_name.lower() == 'na':
						artefacts_pos.append(my_pos)
					else:
						annotated_peak_names.append(display_name)
						annotated_confidences.append(conf)

				assert len(annotated_peak_names) == len(annotated_confidences)
				# annotated_peak_names = np.array(["%s, %s" % (x, y) for (x, y) in \
				# 	zip(annotated_peak_names, annotated_confidences)])
				annotated_peak_names = np.array(annotated_peak_names)
				annotated_confidences = np.array(annotated_confidences)

				original_filenames = control_cols + case_cols
				renamed = []
				for fn in original_filenames:
					if fn in filename_label:
						renamed.append(filename_label[fn])
					else:
						renamed.append(fn)

				fig = plt.figure(figsize=(15, 10))
				ax1 = fig.add_subplot(1, 1, 1)

				# delete unwanted rows
				if len(artefacts_pos) > 0:
					to_plot = np.delete(sub_mat, artefacts_pos, axis=0)
				else:
					to_plot = sub_mat

				# reorder matrix according to the first column
				temp = to_plot[:, 0:len(control_cols)+1]
				temp = np.mean(temp, axis=1)
				reorder_idx = temp.argsort()
				to_plot = to_plot[reorder_idx]
				annotated_peak_names = annotated_peak_names[reorder_idx]
				annotated_confidences = annotated_confidences[reorder_idx]

				# make teh plot
				plt.imshow(to_plot, interpolation='none', aspect='equal')
				plt.title("{}".format(t_annotated_name), fontsize=36, fontweight='bold')
				plt.xticks(range(len(renamed)),renamed,rotation='vertical', fontsize=32)
				plt.yticks(range(len(annotated_peak_names)),annotated_peak_names, fontsize=32)
				yticks = ax1.yaxis.get_major_ticks()
				for i in range(len(yticks)):
					yt = yticks[i]
					if annotated_confidences[i] == 1:
						yt.label1.set_weight('bold')
					elif annotated_confidences[i] == 2:
						yt.label1.set_style('italic')
				plt.colorbar()
				plt.tight_layout()
				outfile = 'heatmap_%d.png' % n
				plt.savefig(outfile)
				print 'Heatmap saved to %s' % outfile
				print

				plt.figure()
				plt.bar(range(len(control_cols+case_cols)),v[0])
				plt.xticks([i + 0.5 for i in range(len(control_cols+case_cols))],control_cols+case_cols,rotation='vertical')
				plt.figure()
				plt.bar(range(len(peak_names)),u[0])
				plt.xticks([i + 0.5 for i in range(len(peak_names))],peak_names,rotation='vertical')

				for pn in peak_names:
					out_str = '%.3f %s' % (t_val, t_annotated_name)
					if pn in peak_scores:
						peak_scores[pn].append(out_str)
					else:
						peak_scores[pn] = [out_str]

			n += 1

		count = len(topic_scores)
		print "Found %d topics/clusters with >1 member above threshold=%f" % (count, t_thresh)
		return special_nodes, topic_scores, peak_scores, G

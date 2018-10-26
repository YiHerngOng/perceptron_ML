# !/usr/bin/env python

import numpy as np
import os, sys
import pdb
from process_data import *
import csv
from decimal import Decimal

class Perceptron():
	def __init__(self, fn_train, fn_valid):
		self.csv_train = CSV(fn_train)
		self.csv_train.extract_XY()
		self.csv_train.categorize_Y()
		self.x_train, self.y_train = self.csv_train.convert_all_to_numbers()		

		self.csv_valid = CSV(fn_valid)
		self.csv_valid.extract_XY()
		self.csv_valid.categorize_Y()
		self.x_valid, self.y_valid = self.csv_valid.convert_all_to_numbers()		

	def online_perceptron(self):
		iteration = 15
		w = np.zeros(len(self.x_train[0]))
		w_dec = []
		for a in xrange(len(w)):
			try:
				w_dec.append(Decimal(w[a]))
			except:
				raise TypeError
		w_dec = np.array(w_dec)
		# pdb.set_trace()
		# for i in range(iteration):
		# 	print i
		self.acc = []
		with open('valid1.csv', 'wb') as csvfile:
			writer = csv.writer(csvfile, delimiter=',')
			for i in range(iteration):
				for j, k in enumerate(self.x_train):
					u = np.dot(w_dec, self.x_train[j])
					if (self.y_train[j]*u) <= 0:
						# print self.y_train[j]*u
						w_dec = w_dec + (self.y_train[j]*self.x_train[j])
				accuracy_valid = 0
				accuracy_train = 0
				# Determine training accuracy
				for a in xrange(len(self.x_train)):
					y_pred_train = np.dot(w_dec, self.x_train[a])
					if y_pred_train < 0.0:
						y_pred_train = -1
					elif y_pred_train > 0.0:
						y_pred_train = 1
					if y_pred_train == self.y_train[m]:
						accuracy_train = accuracy_train + 1
				# Determine validation accuracy
				for m, n in enumerate(self.x_valid):
					y_pred_valid = float(np.dot(w_dec,self.x_valid[m]))
					if y_pred_valid < 0.0:	# could be 5 
						y_pred = -1
					elif y_pred_valid >= 0.0: # could be 3
						y_pred = 1
					if y_pred == self.y_valid[m]:
						accuracy_valid = accuracy_valid + 1
						# writer.writerow([y_pred_valid, y_pred, self.y_valid[m]])
				accuracies_valid = float(accuracy_valid) / float(len(self.y_valid))
				accuracies_train = float(accuracy_train) / float(len(self.y_train))
				writer.writerow([accuracies])
				self.acc.append(accuracies)

	def average_perceptron(self):
		pass
if __name__ == '__main__':
	perceptron = Perceptron('pa2_train.csv', 'pa2_valid.csv')
	# print perceptron.data
	perceptron.online_perceptron()

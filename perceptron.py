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
		self.acc_valid_online = []
		self.acc_train_online = []
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
					y_pred_train = self.sign_function(w_dec, self.x_train[a])
					if y_pred_train == self.y_train[a]:
						accuracy_train = accuracy_train + 1
				accuracies_train = float(accuracy_train) / float(len(self.y_train))		

				# Determine validation accuracy
				for m, n in enumerate(self.x_valid):
					y_pred_valid = self.sign_function(w_dec, self.x_train[m])
					if y_pred == self.y_valid[m]:
						accuracy_valid = accuracy_valid + 1
						# writer.writerow([y_pred_valid, y_pred, self.y_valid[m]])
				accuracies_valid = float(accuracy_valid) / float(len(self.y_valid))
				writer.writerow([accuracies_valid, accuracies_train])
				self.acc_valid_online.append(accuracies_valid)
				self.acc_train_online.append(accuracies_train)

	def average_perceptron(self):
		iteration = 15
		w = np.zeros(len(self.x_train[0]))
		w_avg = np.zeros(len(self.x_train[0]))
		w_dec = []
		w_avg_dec = []
		for a in xrange(len(w)):
			try:
				w_dec.append(Decimal(w[a]))
				w_avg_dec.append(Decimal(w_avg[a]))
			except:
				raise TypeError
		w_dec = np.array(w_dec)
		w_avg_dec = np.array(w_avg_dec)

		c = 0
		c_s = 0
		with open('valid2.csv', 'wb') as csvfile:
			writer = csv.writer(csvfile, delimiter=',')
			for i in range(iteration):
				for j in xrange(len(self.x_train)):
						u = np.dot(w_dec, self.x_train[j])
						if u < 0.0:
							u_sign = -1
						elif u >= 0.0:
							u_sign = 1
						if (self.y_train[j]*u_sign) <= 0:
							if c_s + c > 0:
								w_avg_dec = (c_s*w_avg_dec + c*w_dec) / (c_s + c)
							c_s = c_s + c
							w_dec = w_dec + (self.y_train[j]*self.x_train[j])
							c = 0
						else:
							c = c+1
				if c > 0:
					w_avg_dec = (c_s*w_avg_dec+ c*w_dec) / (c_s + c)

				self.acc_valid_avg = []
				self.acc_train_avg = []
				accuracy_train = 0
				accuracy_valid = 0
				# Determine training accuracy
				for a in xrange(len(self.x_train)):
					y_pred_train = self.sign_function(w_avg_dec, self.x_train[a])
					if y_pred_train == self.y_train[a]:
						accuracy_train = accuracy_train + 1
				accuracies_train = float(accuracy_train) / float(len(self.y_train))

				# Determine validation accuracy
				for m, n in enumerate(self.x_valid):
					y_pred_valid = self.sign_function(w_avg_dec, self.x_valid[m])
					if y_pred_valid == self.y_valid[m]:
						accuracy_valid = accuracy_valid + 1
				accuracies_valid = float(accuracy_valid) / float(len(self.y_valid))
				self.acc_valid_avg.append(accuracies_valid)
				self.acc_train_avg.append(accuracies_train)
				writer.writerow([accuracies_valid, accuracies_train])

	def kernel_perceptron(self):
		alpha = np.zeros(len(self.x_train))
		K_train = np.zeros((len(self.x_train), len(self.x_train)))
		K_valid = np.zeros((len(self.x_valid), len(self.x_valid)))
		alpha_dec = []
		y_train_dec = []
		y_valid_dec = []

		for i in xrange(len(self.x_train)):
			alpha_dec.append(Decimal(alpha[i]))
			y_train_dec.append(Decimal(self.y_train[i]))
		for j in xrange(len(self.x_valid)):
			y_valid_dec.append(Decimal(self.y_valid[j]))

		alpha_dec = np.array(alpha_dec)
		y_train_dec = np.array(y_train_dec)
		y_valid_dec = np.array(y_valid_dec)

		p_arr = [1,2,3,7,15]
		for p in p_arr:
			for b in xrange(len(self.x_train)):
				for c in xrange(len(self.x_train)):
					# print 'K_train'
					K_train[b, c] = self.kernel_function(self.x_train[b], self.x_train[c], p)

			for m in xrange(len(self.x_valid)):
				for n in xrange(len(self.x_valid)):
					# print 'K_valid'
					K_valid[m,n] = self.kernel_function(self.x_valid[m], self.x_train[n], p)

			iteration = 15
			# u = 0
			for g in range(iteration):
				accuracy_valid = 0
				accuracy_train = 0
				print g
				for a in xrange(len(self.x_train)):
					u = np.sign(np.sum(K_train[:,a]*alpha_dec*np.transpose(y_train_dec)))
					if y_train_dec[a]*u <= 0:
						alpha_dec[a] += Decimal(1)

				for c in xrange(len(self.x_train)):
					y_pred_train = np.sign(np.sum(K_train[:,c]*alpha_dec*np.transpose(y_train_dec)))
					if y_pred_train == y_train_dec[c]:
						accuracy_train = accuracy_train + 1
				accuracies_train = float(accuracy_train) / float(len(y_train_dec))

				for e in xrange(len(self.x_valid)):
					y_pred_valid = np.sign(np.sum(K_valid[:,e]*alpha_dec*np.transpose(y_train_dec))) 
					if y_pred_valid == y_valid_dec[e]:
						accuracy_valid = accuracy_valid + 1
				accuracies_valid = float(accuracy_valid) / float(len(y_valid_dec))
				with open('valid_{}.csv'.format(p), 'wb') as csvfile:
					writer = csv.writer(csvfile, delimiter=',')
					writer.writerow([accuracies_valid, accuracies_train])



	# determine whether it's 1 (3) or -1 (5)
	# prediction function
	def sign_function(self, w, x):
		if float(np.dot(w, x)) < 0.0:
			y = -1
		elif float(np.dot(w, x)) >= 0.0:
			y = 1
		return y

	def sign_function_2(self, s):
		if s < 0.0:
			return -1
		elif s >= 0.0:
			return 1

	def kernel_function(self, x_i, x_j, p):
		return (1 + np.dot(x_i, x_j))**p


if __name__ == '__main__':
	perceptron = Perceptron('pa2_train.csv', 'pa2_valid.csv')
	# print perceptron.data
	# perceptron.online_perceptron()
	# perceptron.average_perceptron()
	perceptron.kernel_perceptron()

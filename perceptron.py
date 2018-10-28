# !/usr/bin/env python

import numpy as np
import os, sys
import pdb
from process_data import *
import csv
from decimal import Decimal

class Perceptron():
	def __init__(self, fn_train, fn_valid, fn_test):
		self.csv_train = CSV(fn_train)
		self.csv_train.extract_XY()
		self.csv_train.categorize_Y()
		self.x_train, self.y_train = self.csv_train.convert_all_to_numbers()		

		self.csv_valid = CSV(fn_valid)
		self.csv_valid.extract_XY()
		self.csv_valid.categorize_Y()
		self.x_valid, self.y_valid = self.csv_valid.convert_all_to_numbers()		

		self.csv_test = CSV(fn_test)
		self.csv_test.extract_XY_test()
		self.csv_test.categorize_Y()
		self.x_test, self.y_test = self.csv_test.convert_all_to_numbers()

	# online iter is 14
	def online_perceptron(self):
		iteration = 14
		w = np.zeros(len(self.x_train[0]))
		arr = []
		# pdb.set_trace()
		# with open('oplabel.csv', 'wb') as csvfile:

		for i in range(iteration):
			print i
			for j, k in enumerate(self.x_train):
				u = np.dot(w, self.x_train[j])
				if (self.y_train[j]*u) <= 0:
					# print self.y_train[j]*u
					w = w + (self.y_train[j]*self.x_train[j])
			accuracy_valid = 0
			accuracy_train = 0

			# Determine training accuracy
			for a in xrange(len(self.x_train)):
				y_pred_train = self.sign_function(w, self.x_train[a])
				if y_pred_train == self.y_train[a]:
					accuracy_train = accuracy_train + 1
			accuracies_train = float(accuracy_train) / float(len(self.y_train))		

			# Determine validation accuracy
			# for m, n in enumerate(self.x_valid):
			# 	y_pred_valid = self.sign_function(w, self.x_valid[m])
			# 	if y_pred_valid == self.y_valid[m]:
			# 		accuracy_valid = accuracy_valid + 1
			for m in xrange(len(self.x_test)):
				y_pred_test = self.sign_function(w, self.x_test[m])
				if i == 13:
					# print 'in here'
					# pdb.set_trace()
					arr.append(y_pred_test)
				# if y_pred_test == self.y_test[m]:
				# 	accuracy_test += 1
					# writer.writerow([y_pred_valid, y_pred, self.y_valid[m]])
			# accuracies_valid = float(accuracy_valid) / float(len(self.y_valid))
			# accuracies_test = float(accuracy_test) / float(len(self.y_test))

			# writer.writerow([accuracies_valid, accuracies_train])
			# self.acc_valid_online.append(accuracies_valid)
			# self.acc_train_online.append(accuracies_train)
		fn = open("oplabel_1.csv", 'wb')		
		for p in xrange(len(self.x_test)):
			fn.write(str(arr[p]))
			fn.write('\n')
		fn.close()

	def average_perceptron(self):
		iteration = 15
		w = np.zeros(len(self.x_train[0]))
		w_avg = np.zeros(len(self.x_train[0]))

		c = 0
		c_s = 0
		with open('valid_avg1.csv', 'wb') as csvfile:
			writer = csv.writer(csvfile, delimiter=',')
			for i in range(iteration):
				for j in xrange(len(self.x_train)):
						u = np.dot(w, self.x_train[j])
						if u < 0.0:
							u_sign = -1
						elif u >= 0.0:
							u_sign = 1
						if (self.y_train[j]*u_sign) <= 0:
							if c_s + c > 0:
								w_avg = (c_s*w_avg + c*w) / (c_s + c)
							c_s = c_s + c
							w = w + (self.y_train[j]*self.x_train[j])
							c = 0
						else:
							c = c+1
				if c > 0:
					w_avg = (c_s*w_avg+ c*w) / (c_s + c)

				# self.acc_valid_avg = []
				# self.acc_train_avg = []
				accuracy_train = 0
				accuracy_valid = 0
				# Determine training accuracy
				for a in xrange(len(self.x_train)):
					y_pred_train = self.sign_function(w_avg, self.x_train[a])
					if y_pred_train == self.y_train[a]:
						accuracy_train = accuracy_train + 1
				accuracies_train = float(accuracy_train) / float(len(self.y_train))

				# Determine validation accuracy
				for m, n in enumerate(self.x_valid):
					y_pred_valid = self.sign_function(w_avg, self.x_valid[m])
					if y_pred_valid == self.y_valid[m]:
						accuracy_valid = accuracy_valid + 1
				accuracies_valid = float(accuracy_valid) / float(len(self.y_valid))
				# self.acc_valid_avg.append(accuracies_valid)
				# self.acc_train_avg.append(accuracies_train)
				writer.writerow([accuracies_valid, accuracies_train])

	def kernel_perceptron(self):
		alpha = np.zeros(len(self.x_train))
		K_train = np.zeros((len(self.x_train), len(self.x_train)))
		# K_valid = np.zeros((len(self.x_valid), len(self.x_valid)))

		# p_arr = [1,2,3,7,15]
		# for p in p_arr:
		p = 3		
		alpha = np.zeros(len(self.x_train))
		print 'p is {}'.format(p)
		for b in xrange(len(self.x_train)):
			for c in xrange(len(self.x_train)):
				K_train[b, c] = (1 + np.dot(self.x_train[b],self.x_train[c]))**p



		iteration = 4
		# u = 0
		arr = []
		# with open('test_p_{}.csv'.format(p), 'wb') as csvfile:
		for g in range(iteration):
			accuracy_valid = 0
			accuracy_train = 0
			print g
			for a in xrange(len(self.x_train)):					
				u = np.sign(np.sum(K_train[:,a]*alpha*np.transpose(self.y_train)))
				if self.y_train[a]*u <= 0:
					alpha[a] += 1.0

			for i in xrange(len(self.x_train)):				
				y_pred_train = np.sign(np.sum(K_train[:,i]*alpha*np.transpose(self.y_train)))
				if y_pred_train == self.y_train[i]:
					accuracy_train = accuracy_train + 1.0
			accuracies_train = float(accuracy_train) / float(len(self.y_train))

			# for e in xrange(len(self.x_valid)):
			# 	y_pred_valid = 0
			# 	for j in xrange(len(self.x_train)):
			# 		if alpha[j] != 0.0:
			# 			y_pred_valid += self.kernel_function(self.x_valid[e], self.x_train[j],p)*alpha[j]*self.y_train[j]
			# 	y_pred_valid_sign = np.sign(y_pred_valid)
			# 	if y_pred_valid_sign == self.y_valid[e]:
			# 		accuracy_valid = accuracy_valid + 1.0
			# accuracies_valid = float(accuracy_valid) / float(len(self.y_valid))

			for k in xrange(len(self.x_test)):
				y_pred_test = 0
				for m in xrange(len(self.x_train)):
					if alpha[m] != 0.0:
						y_pred_test += self.kernel_function(self.x_test[k], self.x_train[m],p)*alpha[m]*self.y_train[m]
				y_pred_test_sign = np.sign(y_pred_test)
				if g == 3:
					print y_pred_test_sign
					arr.append(y_pred_test_sign)
				# writer = csv.writer(csvfile, delimiter=',')
				# writer.writerow([accuracies_valid, accuracies_train])
		# pdb.set_trace()
		with open('kplabel.csv', 'wb') as csvfile:
			for z in xrange(len(self.x_test)):
				writer = csv.writer(csvfile)
				writer.writerow([arr[z]])


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
		return (1 + np.dot(x_i,x_j))**p

	# def prediction(self):


if __name__ == '__main__':
	perceptron = Perceptron('pa2_train.csv', 'pa2_valid.csv', 'pa2_test_no_label.csv')
	# print perceptron.data
	perceptron.online_perceptron()
	# perceptron.average_perceptron()
	# perceptron.kernel_perceptron()

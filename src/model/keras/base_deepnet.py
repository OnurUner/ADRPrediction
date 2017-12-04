#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np

class BaseDeepNet:
	
	def __init__(self):
		self.model = None
		self.weights_path = ''
	
	def predict(self, x_test, batch_size=30):
		return self.model.predict(x_test, batch_size=batch_size)
		
	def evaluate(self, x_test, y_test, batch_size=20):
		scores = self.model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
		return self.model.metrics_names, scores
		
	def save_model(self):
		np.save(self.weights_path, self.model.get_weights())
		print("Saved model to disk")

	def load_model(self):
		loaded_weights = np.load(self.weights_path)
		self.model.set_weights(loaded_weights)
		print("Loaded model from disk")

	
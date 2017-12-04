import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


class AdrNet(nn.Module):
	"""Adverse Drug Reaction Network"""
	D_h1 = 9000
	D_h2 = 9000
	D_h3 = 9000
	D_h4 = 9000
	D_h5 = 9000
	
	def __init__(self, D_in, D_out):
		super(AdrNet, self).__init__()
		self.fc1 = nn.Linear(D_in, self.D_h1)
		self.fc2 = nn.Linear(self.D_h1, self.D_h2)
		self.fc3 = nn.Linear(self.D_h2, self.D_h3)
		self.fc4 = nn.Linear(self.D_h3, self.D_h4)
		self.fc5 = nn.Linear(self.D_h4, self.D_h5)
		self.fc_out = nn.Linear(self.D_h5, D_out)
		
		self.batch_time = AverageMeter()
		self.data_time = AverageMeter()
		self.losses = AverageMeter()
		self.accuracies = AverageMeter()
		
		self.loss_graph = []
		self.acc_graph = []
		self.val_loss_graph = []
		self.val_acc_graph = []
		
	def forward(self, X):
		h1 = F.relu(self.fc1(X))
		h1 = F.dropout(h1, training=self.training)

		h2 = F.relu(self.fc2(h1))
		h2 = F.dropout(h2, training=self.training)

		h3 = F.relu(self.fc3(h2))
		h3 = F.dropout(h3, training=self.training)

		h4 = F.relu(self.fc4(h3))
		h4 = F.dropout(h4, training=self.training)

		h5 = F.relu(self.fc5(h4))
		h5 = F.dropout(h5, training=self.training)

		output = F.sigmoid(self.fc_out(h5))
		return output
	
	def calc_accuracy(self, y_prob, y_true):
		y_pred = y_prob.data.clone()
		y_pred[y_pred >= 0.5] = 1
		y_pred[y_pred < 0.5] = 0
		return np.mean(y_pred.eq(y_true.data).cpu().numpy())
		
	def reset_average_meter(self):
		self.batch_time.reset()
		self.data_time.reset()
		self.losses.reset()
		self.accuracies.reset()
	
	def print_iter(self, epoch, batch_idx, num_batch, save=False):
		print('Epoch: [{0}][{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
				epoch, batch_idx, num_batch, batch_time=self.batch_time,
				data_time=self.data_time, loss=self.losses, acc=self.accuracies))
		if save:
			self.loss_graph.append(self.losses.avg)
			self.acc_graph.append(self.accuracies.avg)

	def print_validation(self, save=False):
		print('Val Loss: {val_loss.avg:.4f}\t'
			  'Val Accuracy: {val_acc.avg:.3f}'.format(val_loss=self.losses, val_acc=self.accuracies))
		print "---------------------"
		self.val_loss_graph.append(self.losses.avg)
		self.val_acc_graph.append(self.accuracies.avg)

	@staticmethod
	def weighted_binary_cross_entropy(output, target, weights=None):
		if weights is not None:
			assert len(weights) == 2
	
			loss = torch.mul(weights[1].unsqueeze(0), (target * torch.log(output))) + torch.mul(weights[0].unsqueeze(0), ((1 - target) * torch.log(1 - output)))
		else:
			loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)
	
		return torch.neg(torch.mean(loss))
	
	def fit(self, data_loader, epochs, criterion, optimizer, validation_loader=None, verbose=0):
		self.cuda()
		self.train()
		
		for epoch in range(epochs):
			end = time.time()
			self.reset_average_meter()
			loss_values = []
			for batch_idx, (data, target) in enumerate(data_loader):
				self.data_time.update(time.time() - end)
				data = Variable(data.cuda(), requires_grad=False)
				target = Variable(target.cuda(async=True), requires_grad=False)
				
				# compute output
				output = self.forward(data)
				loss = criterion(output, target)
				
				# measure accuracy and record loss
				acc = self.calc_accuracy(output, target)
				self.losses.update(loss.data[0], data.size(0))
				self.accuracies.update(acc, data.size(0))
				loss_values.append(loss.data[0])
				
				# compute gradient and do SGD step
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				
				# measure elapsed time
				self.batch_time.update(time.time() - end)
				end = time.time()

				# print iteration info
				if verbose == 1:
					self.print_iter(epoch, batch_idx, len(data_loader))

			# print epoch info
			if verbose == 2:
				self.print_iter(epoch, len(data_loader), len(data_loader), save=True)

			# validation
			if validation_loader is not None:
				self.validate(validation_loader, criterion)
				self.print_validation(save=True)
				self.train()
	
	def validate(self, data_loader, criterion, verbose=0):
		self.reset_average_meter()
		self.cuda()
		self.eval()
		
		end = time.time()
		for batch_idx, (data, target) in enumerate(data_loader):
			self.data_time.update(time.time() - end)
			data = Variable(data.cuda(), volatile=True)
			target = Variable(target.cuda(async=True), volatile=True)
			
			# compute output
			output = self.forward(data)
			loss = criterion(output, target)
			
			# measure accuracy and record loss
			acc = self.calc_accuracy(output, target)
			self.losses.update(loss.data[0], data.size(0))
			self.accuracies.update(acc, data.size(0))
		
			# measure elapsed time
			self.batch_time.update(time.time() - end)
			end = time.time()
			
			if verbose == 1:
				self.print_iter(1, batch_idx, len(data_loader))

		if verbose == 2:
			self.print_iter(1, len(data_loader), len(data_loader))

	def test(self, data_loader):
		""" works for only one sample at a time"""
		""" batch size must be 1"""
		# TODO: make batch size independent
		self.cuda()
		self.eval()
		y_prob = dict()
		y_true = dict()
		for batch_idx, (data, target) in enumerate(data_loader):
			data = Variable(data.cuda(), volatile=True)
			target = Variable(target.cuda(async=True), volatile=True)
			output = self.forward(data)
			output = output.data.cpu().numpy()
			drug_id = data_loader.dataset.get_drug_id(batch_idx)
			if drug_id not in y_prob:
				y_prob[drug_id] = []
			y_prob[drug_id].append(output)
			y_true[drug_id] = target.data.cpu().numpy()
		return y_prob, y_true

	def plot_loss_graph(self, title):	
		plt.figure()
		ax = plt.subplot(111)
		plt.title(title)
		t = np.arange(0, len(self.loss_graph), 1)
		plt.plot(t, self.loss_graph, label="Training Loss")
		plt.plot(t, self.val_loss_graph, label = "Validation Loss")
		plt.xlabel("Epoch")
		plt.ylabel("Loss")
		ax.legend()
		plt.show()
		
	def plot_acc_graph(self, title):
		plt.figure()
		ax = plt.subplot(111)
		plt.title(title)
		t = np.arange(0, len(self.acc_graph), 1)
		plt.plot(t, self.acc_graph, label="Training Accuracy")
		plt.plot(t, self.val_acc_graph, label="Validation Accuracy")
		plt.xlabel("Epoch")
		plt.ylabel("Accuracy")
		ax.legend()
		plt.show()
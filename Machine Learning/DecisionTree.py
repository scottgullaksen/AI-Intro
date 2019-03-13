from math import log2
from collections import Counter
import os

class Tree(object):
	def __init__(self, rootTest):
		self.rootTest = rootTest
		self.branches = {} # Branches of this subtree to its subtrees
	
	def addBranch(self, label, subtree):
		self.branches[label] = subtree

	def test(self, example):
		return self.branches[example[0][self.rootTest]]

class DecisionTree(object):
	def __init__(self, examples):
		self.examples = examples  # Must be a list of tuples (X, y)
		self.attributes = self.mapAttrToValues(examples)  # attributes[i] = (possible values...)
		self.tree = None
	
	def mapAttrToValues(self, examples):  # Dics makes deletion and indexes easier
		attributes = {}
		for i in range(len(examples[0][0])):
			attributes[i] = set([ex[0][i] for ex in examples])
		return attributes

	# Constructs the tree. Analogoue to Fig. 18.5
	def learn(self, examples, attributes, parentExamples = None):
		if not examples: return self.pluralityValue(parentExamples)  # No observed example of this combination
		elif self.sameClass(examples): return examples[0][-1] # the class, i.e we are done
		elif not attributes: return self.pluralityValue(examples)  # Noise, error
		else:
			A = max(attributes, key=lambda a: self.importance(a, examples))  # Choose spit attribute
			tree = Tree(A)  # New tree with test attribute A
			for value in attributes[A]:
				exs = [ex for ex in examples if ex[0][A] == value]
				subtree = self.learn(exs, {a:attributes[a] for a in attributes if a!=A}, examples)
				tree.addBranch(value, subtree)
			self.tree = tree
		
	def sameClass(self, examples):  # Does all the examples have the same class?
		return len(self.classDistrubution(examples)) == 1

	def pluralityValue(self, examples):  # Returns classification with most instances
		split = self.classDistrubution(examples)
		return max(split, key= split.get)

	def entropy(self, V): return -sum([prob*log2(prob) for prob in V])

	def remainder(self, A, examples):
		remSum = 0
		for value in self.attributes[A]:  
			subset = [ex for ex in examples if ex[0][A] == value]
			V = self.computeProbDistrubution(subset)
			remSum += self.entropy(V) * (len(subset) / len(examples))
		return remSum

	# Returns dictionary with classes and corresponding number of instances
	def classDistrubution(self, examples): return Counter([ex[1] for ex in examples]) 

	def computeProbDistrubution(self, examples):  # Computes probability distrubution for goal attribute
		split = self.classDistrubution(examples)
		total = sum(split.values())
		return [x/total for x in split.values()]

	def importance(self, a, examples):
		V = self.computeProbDistrubution(examples)
		return self.entropy(V) - self.remainder(a, examples)  # Gain(A)

	def classify(self, example):
		currentNode = self.tree
		while type(currentNode) == Tree:
			currentNode = currentNode.test(example)
		return currentNode

	def printTree(self):  #TODO
		frontier = [('Root', self.tree)]
		curr_leaves = 1
		while frontier:
			node = frontier.pop(0)
			curr_leaves -= 1
			end = '\n' if curr_leaves == 0 else '  '
			if type(node[1]) is not Tree:
				print(node, end= end)
			else:
				print({node[0]: node[1].rootTest}, end= end)
				curr_leaves += len(node[1].branches)
				for branch in node[1].branches:
					frontier.append((branch, node[1].branches[branch]))




def loadExamples(filename):
	examples = []
	f = open(os.getcwd() + '/' + filename, 'r')
	for line in f:
		line = line.split()
		inputAttributes = tuple(line[0:-2])  # Vector
		output = line[-1]
		examples.append((inputAttributes, output))  # example format: (X, y)
	f.close()
	return examples

examples = loadExamples('Assignment_4_Data/training.txt')
bdt = DecisionTree(examples)
bdt.learn(bdt.examples, bdt.attributes)

print(bdt.tree.branches)
bdt.printTree()
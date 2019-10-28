from math import log2
from collections import Counter
import os
from copy import deepcopy
import random

class Tree(object):
	def __init__(self, rootTest):
		self.rootTest = rootTest
		self.branches = {} # Branches of this subtree to its subtrees
	
	def addBranch(self, label, subtree):
		self.branches[label] = subtree

	def test(self, example):
		return self.branches[example[0][self.rootTest]]

class DecisionTree(object):
	def __init__(self, examples, randomImportance):
		self.examples = examples  # Must be a list of tuples (X, y)
		self.attributes = self.mapAttrToValues(examples)  # attributes[i] = (possible values...)
		self.tree = None
		self.randomImportance = randomImportance
	
	def mapAttrToValues(self, examples):  # Dics makes deletion and indexes easier
		attributes = {}
		for i in range(len(examples[0][0])):
			attributes[i] = set([ex[0][i] for ex in examples])
		return attributes
	
	def learnFromExamples(self):
		self.tree = self.createTree(self.examples, self.attributes)

	# Constructs the tree. Analogoue to Fig. 18.5
	def createTree(self, examples, attributes, parentExamples = None):
		if not examples: return self.pluralityValue(parentExamples)  # No observed example of this combination
		elif self.sameClass(examples): return examples[0][-1] # the class, i.e we are done
		elif not attributes: return self.pluralityValue(examples)  # Noise, error
		else:
			A = max(attributes, key=lambda a: self.importance(a, examples))  # Choose spit attribute
			tree = Tree(A)  # New tree with test attribute A
			for value in attributes[A]:
				exs = [ex for ex in examples if ex[0][A] == value]
				subtree = self.createTree(exs, {a:attributes[a] for a in attributes if a!=A}, examples)
				tree.addBranch(value, subtree)
			return tree
		
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

	# Computes probability distrubution for goal attribute. Used by entropy calculation. 
	def computeProbDistrubution(self, examples):  
		split = self.classDistrubution(examples)  
		total = sum(split.values())
		return [x/total for x in split.values()]

	def importance(self, a, examples):  # ranks attributes for splitting
		if self.randomImportance: return random.random()
		V = self.computeProbDistrubution(examples)
		return self.entropy(V) - self.remainder(a, examples)  # Gain(A)

	def classify(self, testExample):
		currentNode = self.tree
		while type(currentNode) == Tree:
			currentNode = currentNode.test(testExample)
		return currentNode
	
	def testAccuracy(self, testSet): 
		correct = 0
		for ex in testSet:
			correct += 1 if self.classify(ex) == ex[1] else 0
		return correct / len(testSet)

	
	def printTree(self):
		currentLevel = [('root', self.tree.rootTest, self.tree)]
		nextLevel = []
		while currentLevel:
			for node in currentLevel:
				print(node[:3], end= '  ')
				tree = node[-1]
				if type(tree) is str:
					continue
				for b in tree.branches:
					if type(tree.branches[b]) is str:
						nextLevel.append((tree.rootTest, b, tree.branches[b]))
					else: 
						nextLevel.append((tree.rootTest, b, tree.branches[b].rootTest, tree.branches[b]))
			currentLevel = deepcopy(nextLevel)
			nextLevel.clear()
			print(end='\n')


def loadExamples(filename):  # loads examples from cwd
	examples = []
	f = open(os.getcwd() + '/' + filename, 'r')
	for line in f:
		line = line.split()
		inputAttributes = tuple(line[0:-2])  # Vector
		output = line[-1]
		examples.append((inputAttributes, output))  # example format: (X, y)
	f.close()
	return examples

learningExamples = loadExamples('Machine Learning/Data/training.txt')
bdt = DecisionTree(learningExamples, False)
bdt.learnFromExamples()
bdt.printTree()
testSet = loadExamples('Machine Learning/Data/test.txt')
print(bdt.testAccuracy(testSet))


randomTree = DecisionTree(learningExamples, True)
randomTree.learnFromExamples()
print(randomTree.testAccuracy(testSet))

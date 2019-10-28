class Model_Based_Reflex_Agent(object):
	def __init__(self):
		self.state = None
		self.model = None
		self.rules = {}
		self.action = None
	
	def Update_State(self, percept):  # Use action, model, state and percept to update state
		# Do something with state
		self.state = None

	def Rule_Match(self):  # Match state with rule which contains action to perform
		rule = self.rules[self.state]
		self.action = rule["action"]

class simple_Reflex_Agent(object):
	def __init__(self):
		self.rules = {}

	def interprit_Input(self, percept):
		state = None
		return state

	def produce_Action(self, percept):
		state = self.interprit_Input(percept)
		rule = self.rules[state]
		action = rule["action"]
		return action
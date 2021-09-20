import numpy as np
import cv2
import random
import datetime
import psutil
import ray

num_cpus = psutil.cpu_count(logical=False)
ray.init(num_cpus=num_cpus, num_gpus=1)

class ThreadToneGeneticAlgo():

	#Processing source image into a grayscaled circular image with given radius
	def processing(self, source, radius):
		#Get image
		image = cv2.imread(source)

		#Grayscale
		imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		#equalize
		imgContrasted = cv2.convertScaleAbs(imgGray, alpha=1, beta=10)

		#Cropping image into a square
		height, width = imgContrasted.shape[0:2]
		minEdge= min(height, width)
		topEdge = int((height - minEdge)/2)
		leftEdge = int((width - minEdge)/2)
		imgCropped = imgContrasted[topEdge:topEdge+minEdge, leftEdge:leftEdge+minEdge]

		#Resizing image with the given radius
		imgSized = cv2.resize(imgCropped, (2*radius + 1, 2*radius + 1))

		#Applying circular mask
		y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
		mask = x**2 + y**2 > radius**2
		imgSized[mask] = 255
		self.image = imgSized

	def calculatePins(self, radius, nPins):
		#Calculate all angles with n pins
		angles = np.linspace(0, 2*np.pi, nPins + 1)

		#List of coords of all pins
		self.pinsCoords = []
		#For each angle of pin
		for angle in angles[:-1]:
			#Calculate position of new pin
			x = int(radius + 1 + radius*np.cos(angle))
			y = int(radius + 1 + radius*np.sin(angle))

			#Add coords of new pin
			self.pinsCoords.append((x, y))

	def getPossibleThreads(self):
		self.possibleThreads = []
		for i in range(len(self.pinsCoords)):
			for j in range(i+1, len(self.pinsCoords)):
				self.possibleThreads.append([self.pinsCoords[i], self.pinsCoords[j]])

	def __init__(self, source, radius, nPins, populationSize, generations, lineWidth, cellWidth, initialLineProbability, mutationChance):
		#Process source image
		self.processing(source, radius)
		#Calculate coords of pins
		self.calculatePins(radius, nPins)
		self.radius = radius
		self.getPossibleThreads()
		self.cellWidth = cellWidth
		self.lineWidth = lineWidth
		self.initialLineProbability = initialLineProbability
		self.population = []
		self.populationSize = populationSize
		self.generations = generations
		self.currentGeneration = 1
		self.mutationChance = mutationChance
		self.getRGBSum()

	@ray.remote
	def randomInstance(self):
		new = instance(self)
		new.random()
		return new

	def randomPopulation(self):
		timestamp = datetime.datetime.now()
		for i in range(self.populationSize):
			self.population.append(self.randomInstance.remote(self))
		self.population = ray.get(self.population)

		total = 0
		for i in range(len(self.population)):
			total += self.population[i].score
		print("Generation:", self.currentGeneration, "Score:", total/self.populationSize)
		self.sortPopulation()
		self.saveBest()
		print("Time taken generation:", (datetime.datetime.now() - timestamp).seconds * 1000000 + (datetime.datetime.now() - timestamp).microseconds)


	def nextGeneration(self):
		timestamp = datetime.datetime.now()
		combinations = set()

		while(len(combinations) < self.populationSize):
			first = np.random.choice(np.arange(0, self.populationSize//2))
			second = np.random.choice(np.arange(0, self.populationSize//2))
			if (first != second and (first, second) not in combinations and (second, first) not in combinations):
				combinations.add((first, second))

		new = []
		for combination in combinations:
			new.append(self.population[combination[0]].crossover.remote(self.population[combination[0]], self.population[combination[1]]))
		self.population += ray.get(new)
		del new
		self.currentGeneration += 1
		total = 0
		for i in range(self.populationSize, 2*self.populationSize):
			total += self.population[i].score
		print("Generation:", self.currentGeneration, "Average score:", total/self.populationSize)
		self.sortPopulation()
		self.saveBest()
		print("Time taken generation:", (datetime.datetime.now() - timestamp).seconds * 1000000 + (datetime.datetime.now() - timestamp).microseconds)


	def allGenerations(self):
		while (self.currentGeneration <= self.generations):
			self.nextGeneration()

	def sortPopulation(self):
		self.population.sort(key=lambda x: x.score)
		for i in range(self.populationSize, len(self.population)):
			del self.population[self.populationSize]
		
	def saveBest(self):
		print("Best score:", self.population[0].score)
		self.population[0].save()

	def getRGBSum(self):
		self.RGBsum = []
		for i in range(len(self.image)):
			row = []
			for j in range(len(self.image)):
				imageSum = 0
				for k in range(i-self.cellWidth//2, i+self.cellWidth//2):
					for l in range(j-self.cellWidth//2, j+self.cellWidth//2):
						if ((k-self.radius) ** 2 + (l-self.radius) ** 2 <= self.radius**2):
							if (k >= 0 and k < len(self.image) and l >= 0 and l < len(self.image[0])):
								imageSum += self.image[k][l]
				row.append(imageSum)
			self.RGBsum.append(row)

class instance(object):
	def __init__(self, parent):
		self.parent = parent
		self.binary = []
		self.canvas = np.ones((len(self.parent.image),len(self.parent.image[0]))) * 255
		for i in range(len(self.parent.possibleThreads)):
			self.binary.append(False)

	def random(self):
		for i in range(len(self.parent.possibleThreads)):
			self.binary[i] = np.random.choice([True, False], p=[self.parent.initialLineProbability, 1 - self.parent.initialLineProbability])
			if (self.binary[i]):
				self.drawLine(self.parent.possibleThreads[i])
		self.calculateScore()

	@ray.remote
	def crossover(self, other):
		newInstance = instance(self.parent)
		for i in range(len(self.parent.possibleThreads)):
			if np.random.choice([True, False], p = [self.parent.mutationChance, 1 - self.parent.mutationChance]):
				newInstance.binary[i] = np.random.choice([True, False], p=[self.parent.initialLineProbability, 1 - self.parent.initialLineProbability])
			else:
				newInstance.binary[i] = np.random.choice([other.binary[i], other.binary[i]])
			if (newInstance.binary[i]):
				newInstance.drawLine(newInstance.parent.possibleThreads[i])
		newInstance.calculateScore()
		return newInstance

	# Compute a line mask
	def drawLine(self, pins):
		self.canvas = cv2.line(self.canvas, pins[0], pins[1], (0, 0, 0), self.parent.lineWidth)

	def calculateScore(self):
		self.score = 0
		for i in range(self.parent.cellWidth//2, len(self.canvas), self.parent.cellWidth):
			for j in range(self.parent.cellWidth//2, len(self.canvas[0]), self.parent.cellWidth):
				canvasSum = 0
				for k in range(i-self.parent.cellWidth//2, i+self.parent.cellWidth//2):
					for l in range(j-self.parent.cellWidth//2, j+self.parent.cellWidth//2):
						if ((k-self.parent.radius) ** 2 + (l-self.parent.radius) ** 2 <= self.parent.radius**2):
							if (k >= 0 and k < len(self.canvas) and l >= 0 and l < len(self.canvas[0])):
								canvasSum += self.canvas[k][l]
				self.score += abs(canvasSum - self.parent.RGBsum[i][j])
		print(self.score)

	def save(self):
		cv2.imwrite('assets/' + str(self.score) + '.png', self.canvas) 

def main():
	GA = ThreadToneGeneticAlgo("assets/obama.jpg", 200, 150, 1000, 50, 1, 5, 0.03, 0.01)
	GA.randomPopulation()
	GA.allGenerations()

if __name__ == "__main__":
	main()

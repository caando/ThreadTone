import numpy as np
import cv2
import random
import datetime
import psutil
import ray

num_cpus = psutil.cpu_count(logical=False)
ray.init(num_cpus=num_cpus)

class ThreadToneParticleSwarmOptimisation(object):

	#Processing source image into a grayscaled circular image with given radius
	def processing(self, source, radius):
		#Get image
		image = cv2.imread(source)

		#Grayscale
		imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		#equalize
		imgContrasted = cv2.equalizeHist(imgGray)
		imgContrasted = cv2.convertScaleAbs(imgContrasted, alpha=1.5, beta=8)

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

	def __init__(self, source, radius, nPins, lineWidth, populationSize, iterations, inertia, globalCoefficient, personalCoefficient):
		#Process source image
		self.processing(source, radius)
		#Calculate coords of pins
		self.calculatePins(radius, nPins)
		self.radius = radius
		self.getPossibleThreads()
		self.lineWidth = lineWidth
		self.population = []
		self.populationSize = populationSize
		self.iterations = iterations
		self.current = 1
		self.globalBestValues = []
		self.globalBestScore = 2**63
		for i in range(len(self.possibleThreads)):
			self.globalBestValues.append(0)
		self.inertia = inertia
		self.globalCoefficient = globalCoefficient
		self.personalCoefficient = personalCoefficient

	def getGlobalBestValues(self):
		bestIndex = -1
		for i in range(len(self.population)):
			if self.population[i].score < self.globalBestScore:
				self.globalBestScore = self.population[i].score
				bestIndex = i
		if bestIndex != -1:
			for i in range(len(self.possibleThreads)):
				self.globalBestValues[i] = self.population[bestIndex].values[i]
			self.population[bestIndex].save()
		print("Iteration:", self.current)
		print("Best score:", self.globalBestScore)
		self.current += 1

	def generateRandomPopulation(self):
		population = []
		for i in range(self.populationSize):
			population.append(self.generateRandomParticle.remote(self.image, self.possibleThreads, self.lineWidth))
		self.population += ray.get(population)
		del population
		self.getGlobalBestValues()

	def update(self):
		population = []
		for i in range(len(self.population)):
			population.append(self.updateParticle.remote(self.population[i], self.globalBestValues, self.inertia(self.current, self.iterations), self.personalCoefficient(self.current, self.iterations), self.globalCoefficient(self.current, self.iterations), self.image, self.possibleThreads, self.lineWidth))
		self.population = ray.get(population)
		del population
		self.getGlobalBestValues()

	@ray.remote
	def generateRandomParticle(image, possibleThreads, lineWidth):
		new = particle(image, possibleThreads)
		new.randomGenerate(image, possibleThreads, lineWidth)
		return new

	@ray.remote
	def updateParticle(particle, globalBestValues, inertia, personalCoefficient, globalCoefficient, image, possibleThreads, lineWidth):
		return particle.update(globalBestValues, inertia, personalCoefficient, globalCoefficient, image, possibleThreads, lineWidth)

	def run(self):
		while self.current < self.iterations:
			self.update()

class particle(object):
	def __init__(self, image, possibleThreads):
		self.values = []
		self.velocity = []
		self.personalBestScore = 2**31
		self.personalBestValues = []
		self.score = 2**31
		self.canvas = np.ones((len(image),len(image[0]))) * 255
		for i in range(len(possibleThreads)):
			self.values.append(0)
			self.velocity.append(0)

	def randomGenerate(self, image, possibleThreads, lineWidth):
		for i in range(len(self.values)):
			self.values[i] = np.random.uniform(0, 1)
		add = 0
		for i in range(len(self.velocity)):
			self.velocity[i] = np.random.uniform(-0.2, 0.2)
			add += self.velocity[i]
		self.personalBestValues = self.values.copy()
		self.personalBestScore = self.score
		self.drawLines(image, possibleThreads, lineWidth)
		self.calculateScore(image)

	def drawLines(self, image, possibleThreads, lineWidth):
		self.canvas = np.ones((len(image),len(image[0]))) * 255
		value_to_coords = []
		total = 0
		for i in range(len(possibleThreads)):
			value_to_coords.append([self.values[i], possibleThreads[i]])
			total += self.values[i]
		value_to_coords.sort(reverse=True)
		for i in range(len(value_to_coords)):
			self.canvas = cv2.line(self.canvas, value_to_coords[i][1][0] , value_to_coords[i][1][1], (value_to_coords[i][0]* 255, ) * 3, lineWidth)
		del value_to_coords

	def calculateScore(self, image):
		self.score = 0
		for i in range(len(self.canvas)):
			for j in range(len(self.canvas[i])):
				self.score += abs(self.canvas[i][j] - image[i][j])
		print(self.score)
		if (self.score < self.personalBestScore):
			self.personalBestScore = self.score
			for i in range(len(self.personalBestValues)):
				self.personalBestValues[i] = self.values[i]

	def update(self, globalBestValues, inertia, personalCoefficient, globalCoefficient, image, possibleThreads, lineWidth):
		for i in range(len(self.values)):
			self.velocity[i] = self.velocity[i] * inertia + personalCoefficient * np.random.uniform(0, 2) * (self.personalBestValues[i] - self.values[i]) + globalCoefficient * np.random.uniform(0, 2) * (globalBestValues[i] - self.values[i])
		for i in range(len(self.values)):
			self.values[i] += self.velocity[i]
			if (self.values[i] < 0):
				self.values[i] = 0
			elif (self.values[i] > 1):
				self.values[i] = 1
		self.drawLines(image, possibleThreads, lineWidth)
		self.calculateScore(image)
		return self

	def save(self):
		cv2.imwrite('assets/' + str(self.score) + '.png', self.canvas)

def main():
	GA = ThreadToneParticleSwarmOptimisation("assets/poetin.png", 300, 300, 1, 300, 70, lambda t, n : 0.4 * (t - n) / (n ** 2) + 0.4, lambda t, n : 0.15*(-3 * t / n + 3.5), lambda t, n : 0.15*(3 * t / n + 3.5))
	GA.generateRandomPopulation()
	GA.run()

if __name__ == "__main__":
	main()

import time
from collections import defaultdict
import numpy as np

from MFSTSP.parseCSV import *
from MFSTSP.parseCSVstring import *

from gurobipy import *
import os
import os.path

from MFSTSP.solve_mfstsp_heuristic import *

from MFSTSP import distance_functions

# =============================================================
startTime 		= time.time()

METERS_PER_MILE = 1609.34

problemTypeString = {1: 'mFSTSP IP', 2: 'mFSTSP Heuristic'}


NODE_TYPE_DEPOT	= 0
NODE_TYPE_CUST	= 1

TYPE_TRUCK 		= 1
TYPE_UAV 		= 2

MODE_CAR 		= 1
MODE_BIKE 		= 2
MODE_WALK 		= 3
MODE_FLY 		= 4

ACT_TRAVEL 			= 0
ACT_SERVE_CUSTOMER	= 1
ACT_DONE			= 2

# =============================================================

def make_dict():
	return defaultdict(make_dict)

	# Usage:
	# tau = defaultdict(make_dict)
	# v = 17
	# i = 3
	# j = 12
	# tau[v][i][j] = 44

class make_node:
	def __init__(self, nodeType, latDeg, lonDeg, altMeters, parcelWtLbs, serviceTimeTruck, serviceTimeUAV, address):
		# Set node[nodeID]
		self.nodeType 			= nodeType
		self.latDeg 			= latDeg
		self.lonDeg				= lonDeg
		self.altMeters			= altMeters
		self.parcelWtLbs 		= parcelWtLbs
		self.serviceTimeTruck	= serviceTimeTruck	# [seconds]
		self.serviceTimeUAV 	= serviceTimeUAV	# [seconds]
		self.address 			= address			# Might be None

class make_vehicle:
	def __init__(self, vehicleType, takeoffSpeed, cruiseSpeed, landingSpeed, yawRateDeg, cruiseAlt, capacityLbs, launchTime, recoveryTime, serviceTime, batteryPower, flightRange):
		# Set vehicle[vehicleID]
		self.vehicleType	= vehicleType
		self.takeoffSpeed	= takeoffSpeed
		self.cruiseSpeed	= cruiseSpeed
		self.landingSpeed	= landingSpeed
		self.yawRateDeg		= yawRateDeg
		self.cruiseAlt		= cruiseAlt
		self.capacityLbs	= capacityLbs
		self.launchTime		= launchTime	# [seconds].
		self.recoveryTime	= recoveryTime	# [seconds].
		self.serviceTime	= serviceTime
		self.batteryPower	= batteryPower	# [joules].
		self.flightRange	= flightRange	# 'high' or 'low'

class make_travel:
	def __init__(self, takeoffTime, flyTime, landTime, totalTime, takeoffDistance, flyDistance, landDistance, totalDistance):
		# Set travel[vehicleID][fromID][toID]
		self.takeoffTime 	 = takeoffTime
		self.flyTime 		 = flyTime
		self.landTime 		 = landTime
		self.totalTime 		 = totalTime
		self.takeoffDistance = takeoffDistance
		self.flyDistance	 = flyDistance
		self.landDistance	 = landDistance
		self.totalDistance	 = totalDistance


class MFSTSPUpperSolver():
	def __init__(self, numUAVs, uavs, uav_infos, locations, truck_distances, truck_velocity, masks):

		# timestamp = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
		problemName 		= "MFSTSP"
		cutoffTime 			= float(10)
		problemType 		= int(2)
		numUAVs				= int(numUAVs)
		requireTruckAtDepot = bool(1)
		requireDriver 		= bool(1)
		Etype				= int(5)
		ITER 				= int(1)

		# Define data structures
		self.node = {}
		self.vehicle = {}
		self.travel = defaultdict(make_dict)

		# Read data for node locations, vehicle properties, and travel time matrix of truck:
		self.make_mfstsp( numUAVs, uavs, uav_infos, locations, truck_distances, truck_velocity, masks )

		# Calculate travel times of UAVs (travel times of truck has already been read when we called the readData function)
		# NOTE:  For each vehicle we're going to get a matrix of travel times from i to j,
		#		 where i is in [0, # of customers] and j is in [0, # of customers].
		#		 However, tau and tauPrime let node c+1 represent a copy of the depot.
		for vehicleID in self.vehicle:
			if (self.vehicle[vehicleID].vehicleType == TYPE_UAV):
				# We have a UAV (Note:  In some problems we only have a truck)
				for i in self.node:
					for j in self.node:
						if (j == i):
							self.travel[vehicleID][i][j] = make_travel(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
						else:
							[takeoffTime, flyTime, landTime, totalTime, takeoffDistance, flyDistance, landDistance, totalDistance] = distance_functions.calcMultirotorTravelTime(self.vehicle[vehicleID].takeoffSpeed, self.vehicle[vehicleID].cruiseSpeed, self.vehicle[vehicleID].landingSpeed, self.vehicle[vehicleID].yawRateDeg, self.node[i].altMeters, self.vehicle[vehicleID].cruiseAlt, self.node[j].altMeters, self.node[i].latDeg, self.node[i].lonDeg, self.node[j].latDeg, self.node[j].lonDeg, -361, -361)
							self.travel[vehicleID][i][j] = make_travel(takeoffTime, flyTime, landTime, totalTime, takeoffDistance, flyDistance, landDistance, totalDistance)


		# print(self.node, "\n", self.vehicle, "\n", self.travel, "\n", cutoffTime, problemName, problemType, requireTruckAtDepot, requireDriver, Etype, ITER)
		[objVal, assignments, packages, _, _] = solve_mfstsp_heuristic(self.node, self.vehicle, self.travel, cutoffTime, problemName, problemType, requireTruckAtDepot, requireDriver, Etype, ITER, True)
		# print(objVal, waitingTruck, waitingUAV, sep="\n")
		# print('The mFSTSP Heuristic is Done.  It returned something')
		
		self.truck_tour = []
		self.assignments_uav = []
		# launchs = []
		# retrieves = []
		for v in assignments:
			for statusID in assignments[v]:
				for statusIndex in assignments[v][statusID]:
					if (assignments[v][statusID][statusIndex].vehicleType == TYPE_TRUCK):
						vehicleType = 'Truck'
						if (assignments[v][statusID][statusIndex].ganttStatus == GANTT_TRAVEL):
							self.truck_tour.append((assignments[v][statusID][statusIndex].startNodeID, assignments[v][statusID][statusIndex].endNodeID))
					else:
						vehicleType = 'UAV'
						if (assignments[v][statusID][statusIndex].ganttStatus == GANTT_TRAVEL):
							if (statusID == TRAVEL_UAV_PACKAGE):
								self.assignments_uav.append((v - 2, assignments[v][statusID][statusIndex].startNodeID, assignments[v][statusID][statusIndex].endNodeID))
							# elif (statusID == TRAVEL_UAV_EMPTY):
							# 	retrieves.append((assignments[v][statusID][statusIndex].startNodeID, assignments[v][statusID][statusIndex].endNodeID))
					
					# [v, vehicleType, assignments[v][statusID][statusIndex].startTime, assignments[v][statusID][statusIndex].startNodeID, assignments[v][statusID][statusIndex].endTime, assignments[v][statusID][statusIndex].endNodeID, assignments[v][statusID][statusIndex].description]	
		
		# for v, start_node, customer in launchs:
		# 	for cust, end_node in retrieves: 
		# 		if customer == cust:
		# 			self.assignments_uav.append((v, start_node, customer, end_node))
		# print(self.truck_tour, self.assignments_uav)
		# print("Done.")


	def make_mfstsp(self, numUAVs, uavs, uav_infos, locations, truck_distances, truck_velocity, masks):
		# a)  vehicles
		self.vehicle[1] = make_vehicle(TYPE_TRUCK, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1)
		for i in range(numUAVs):
			vehicleID 			= int(i + 2)
			vehicleType			= TYPE_UAV
			takeoffSpeed		= 10
			cruiseSpeed			= float(uav_infos[uavs[i]]['velocity'])
			landingSpeed		= 10
			yawRateDeg			= 360
			cruiseAlt			= 0
			capacityLbs			= float(uav_infos[uavs[i]]['capacity'])
			launchTime			= 0	# [seconds].
			recoveryTime		= 0	# [seconds].
			serviceTime			= 0	# [seconds].
			batteryPower		= float(1e10)	# [joules]. (unlimited)
			flightRange			= str(uav_infos[uavs[i]]['range'])	# 'high' or 'low'
			
			self.vehicle[vehicleID] = make_vehicle(vehicleType, takeoffSpeed, cruiseSpeed, landingSpeed, yawRateDeg, cruiseAlt, capacityLbs, launchTime, recoveryTime, serviceTime, batteryPower, flightRange)

		# b)  locations
		for i in range(locations.shape[0]):
			nodeID 				= int(i)
			nodeType			= int(bool(i))
			latDeg				= float(locations[i][0])		# IN DEGREES
			lonDeg				= float(locations[i][1])		# IN DEGREES
			altMeters			= float(0)
			if nodeID == 0:
				parcelWtLbs		= float(-1)
			elif masks[0][i] == 0:
				parcelWtLbs		= float(20)
			elif masks[1][i] == 0:
				parcelWtLbs		= float(5)
			else:
				parcelWtLbs		= float(2)
			address 			= '' # or None?

			serviceTimeTruck	= self.vehicle[1].serviceTime
			if numUAVs > 0:
				serviceTimeUAV	= self.vehicle[2].serviceTime
			else:
				serviceTimeUAV = 0

			self.node[nodeID] = make_node(nodeType, latDeg, lonDeg, altMeters, parcelWtLbs, serviceTimeTruck, serviceTimeUAV, address)

		# c) truck_travel_time
		# Travel matrix file exists
		num_nodes = locations.shape[0]
		for i in range(num_nodes):
			for j in range(num_nodes):
				tmpDist	= float(truck_distances[i][j])
				tmpTime	= tmpDist / truck_velocity

				for vehicleID in self.vehicle:
					if (self.vehicle[vehicleID].vehicleType == TYPE_TRUCK):
						self.travel[vehicleID][i][j] = make_travel(0.0, tmpTime, 0.0, tmpTime, 0.0, tmpDist, 0.0, tmpDist)


	def get_solution(self):
		return self.truck_tour, self.assignments_uav


if __name__ == '__main__':
	try:
		# parameters initialization
		num_customer = 20
		num_customer_truck = int(num_customer * 0.2)
		num_flexible_location = num_customer * 1
		
		numUAVs = 4
		uavs = [0, 0, 1, 1] # uav_no -> uav_type
		uav_velocity = [12, 12] # uav_type -> uav_v
		uav_range = [1000, 1000] # uav_type -> uav_r
		uav_capacity = [10, 3.6] # uav_type -> uav_c
		uav_infos = [
			{
				'velocity': 12, 
				'range': 'low', 
				'capacity': 10, 
			}, 
			{
				'velocity': 18, 
				'range': 'high', 
				'capacity': 3.6, 
			}
		]
		
		SIZE = 5_000
		truck_velocity = 8
		locations = np.array([(SIZE / 2,SIZE / 2)] + [(np.random.randint(0, SIZE), np.random.randint(0, SIZE)) for _ in range(num_customer)])
		manhattan_dist_matrix = np.abs(locations[:, np.newaxis] - locations).sum(axis=2)
		masks = np.ones([len(uav_infos), num_customer + 1])
		masks[:, :num_customer_truck + 1] = 0
		MFSTSPUpperSolver(numUAVs, uavs, uav_infos, locations, manhattan_dist_matrix, truck_velocity, masks)
	except:
		print("There was a problem.  Sorry things didn't work out.  Bye.")
		raise
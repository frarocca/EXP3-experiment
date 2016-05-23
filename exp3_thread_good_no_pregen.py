from probability import distr,draw
#from exp3 import exp3, exp3_queued
from writeFileCsv import writeFileCsv

from graph import regretWeightsGraph2
import threading
import math
import random
import Queue
import time
import datetime
import numpy as np

#number of program iteration
numExperiments = 1
#number of arms
numActions = 5
#horizon
numRounds = 30000
#keep reward vector as a global variable to calculate weak regret
rewardVector = []
#best action in hindsight
bestAction = 0
#number of EXP3 agents
numAgents = 1
#convergence lower bound
convergenceBound = 0.75
#reward counter per choice
rewardPerChoice = []


class Exp3Stats:
   def __init__(self, _numActions):
      self.rewardPerChoice = [0 for _ in range(_numActions)]
      self.numActions = _numActions
      self.relativeRound = 0
   def Reset(self):
      del self.rewardPerChoice[:]
      self.rewardPerChoice = [0 for _ in range(self.numActions)]
      self.relativeRound = 0
   def IncRelativeRound(self):
      self.relativeRound += 1
      
class Exp3Queued(threading.Thread):
   agent_index = 0

   def __init__(self, name, gamma):
      threading.Thread.__init__(self)
      self.gamma = gamma
      self.name = name
      self.choice_queue = Queue.Queue()
      self.reward_queue = Queue.Queue()
      self.weights = [1.0] * numActions
      self.choiceFreq = [0]* numActions
      #Distanza dall'ultima scelta
      self.times = [-1]*numActions
      self.D = 1
      
   def distanceLastchoice(self,t,ch):
       self.D = t-self.times[ch]
       self.times[ch] = t
       return self.D   
   
   def exp3(self,numActions, gamma, rewardMin = 0, rewardMax = 1):
      #weights = [1.0] * numActions
   
      t = 0
      while True:
         probabilityDistribution = distr(self.weights, gamma)
         choice = draw(probabilityDistribution)
         self.choiceFreq[choice]+=1
         #put choice in the queue
         self.choice_queue.put(choice)
         self.choice_queue.join()
         #get reward from queue
         theReward = self.reward_queue.get()
         self.reward_queue.task_done()
         #theReward = reward(choice, t)
         scaledReward = (theReward - rewardMin) / (rewardMax - rewardMin) # rewards scaled to 0,1
         probChoice=float(self.choiceFreq[choice] +1)/(t+1)
         #estimatedReward = 1.0 * scaledReward / probabilityDistribution[choice]
         #estimatedReward = (1.0 * scaledReward) / probChoice
         estimatedReward = 1.0 * scaledReward * self.distanceLastchoice(t,choice)
         self.weights[choice] *= math.exp(estimatedReward * gamma / numActions) # important that we use estimated reward here!
         
      
         yield choice, theReward, estimatedReward, self.weights
         t = t + 1
      
   def exp3Thread(self):
      cumulativeReward = 0
      bestActionCumulativeReward = 0
      weakRegret = 0
      t = 0
      sumV=np.array([0.00]*numActions,float)
      #fo = open("results/%s.txt" % (self.name), "wb")
      csvfile = writeFileCsv("results/%s.csv" % (self.name),False)
      for (choice, reward, est, weights) in self.exp3(numActions, self.gamma):
         cumulativeReward += reward
         
         sumV=sumUpdate(sumV, t)
         best= np.argmax(sumV)
         
         bestActionCumulativeReward += rewardVector[t][bestAction]
       
         weakRegret = (bestActionCumulativeReward - cumulativeReward)
         #regretBound = (math.e - 1) * self.gamma * bestActionCumulativeReward + (numActions * math.log(numActions)) / self.gamma
         regretBound = (math.e - 1) * self.gamma * bestActionCumulativeReward + ((numActions * math.log(numActions)) / self.gamma)
         #regretBound =2* float(math.sqrt((math.e - 1)*numActions * math.log(numActions) * t))
         #print("agent: %s\tregret: %d\tmaxRegret: %.2f\tweights: (%s)" % (self.name,weakRegret, regretBound, ', '.join(["%.3f" % weight for weight in distr(weights)])))
         if t % 100==0:
             csvfile.writefile(weakRegret, regretBound, weights)
            #fo.write("regret: %d\tmaxRegret: %.2f\tweights: (%s)\n" % (weakRegret, regretBound, ', '.join(["%.3f" % weight for weight in distr(weights)])));
         line = "%i\n" % (choice)
       
         t += 1
         if t >= numRounds:
            #fo.close()
            csvfile.closefile()
            break
       

   def run(self):
      self.exp3Thread()
   def popChoice(self):
      choice = self.choice_queue.get()
      self.choice_queue.task_done()
      return choice
   def pushReward(self, reward):
      self.reward_queue.put(reward)
      self.reward_queue.join()
   def getWeights(self):
      return self.weights

def sumUpdate(sumV,t):
    global numActions,rewardVector
    for i in range(numActions):
        #printVector(rewardVector[t])
        sumV[i]+=rewardVector[t][i]
    return sumV   

def printVector(vector):
    for x in vector:
        print x
        
        
def genRew(mu,sigma):
    #mean = mu-sigma
    tmp = int(sigma*10000.00)
    return mu+(random.randint(-tmp,tmp)/10000.00)

# Test Exp3 using stochastic payoffs.
def exp3threadTest(simulation_index = 0):

   print datetime.datetime.now()
   start_time = time.time()
   # generate bias vector
   #biases = [1.0 / k for k in range(2,12)]

   #print rewardVector[0][0]
   # generate reward vector
   #rewardVector = [[1 if random.random() < bias else 0 for bias in biases] for _ in range(numRounds)]
   #rewardVector = []
   #for _ in range(numRounds):
    #  rewardVector.append(biases)

   rewards = lambda choice, t: rewardVector[t][choice]
   
   #calculate the best action in hindsight: aka the sum of all rewards over the horizon
   #bestAction = max(range(numActions), key=lambda action: sum([rewardVector[t][action] for t in range(numRounds)]))
   
   #calculate 'g' as an upper bound on rewards of the best action
   #in this case it's the action with the highest probability: 1/2
   #so the upper bound on reward should be T/2
   bestUpperBoundEstimate = 2 * numRounds / 3
   #set gamma using the corollary 
   gamma = min(1,math.sqrt(numActions * math.log(numActions) / ((math.e - 1) * numRounds)))
   
   print "gamma:%3f"%gamma
   #gamma = 0.7

   agentsChoiceFrequencies = [[0 for _ in range(numActions)] for _ in range(numAgents)]
   agentsConvergenceCheck = [True for _ in range(numAgents)]

   # Create and start EXP3 agents
   exp3_threads = []
   for agent_index in range(numAgents):
      thread_name = "EXP3_Thread-" + str(agent_index)
      exp3_thread = Exp3Queued(thread_name, gamma)
      exp3_thread.agent_index = agent_index
      exp3_threads.append(exp3_thread)
      # start EXP3 agent
      exp3_thread.start()

   stats = Exp3Stats(numActions)
   #f = open("EXP3-collisions", 'w')
   fc = open("Simulation-%i_EXP3-choices" % (simulation_index,), 'w')
   for t in range(numRounds):
       
      tmp = [ genRew(1.0/(i+2),1.0/(i+2)/5.0) for i in range(numActions)]
      rewardVector.append(tmp)
          
        
      if (t % 5000) == 0:
         print("Round %i\n" % (t))
      #exp3_choices = [numAgents for _ in range(numActions)]
      exp3_choices = [0 for _ in range(numActions)]
      agent_choices = []
      #collect EXP3 agents choices
      for exp3_agent in exp3_threads:
         #get EXP3 choice on thread queue
         choice = exp3_agent.popChoice()
         #save choice for this agent
         agent_choices.append(choice)
            #save agent index for this choice
            #exp3_choices[choice] = agent_index
            #inc num agents that made this choice
         exp3_choices[choice] += 1;
      
      #save agent's choices on file
      choices_line = '\t'.join(map(str, agent_choices))
      fc.write(choices_line + "\n")
      
      #distribute rewards to agents
      for exp3_agent in exp3_threads:
         choice = agent_choices.pop(0)
         #feed agent with reward
         reward = rewardVector[t][choice]
         #print("[MAIN] reward to agent %s : %i\n" % (exp3_agent.getName(), reward))
         exp3_agent.pushReward(reward)
         
         #update reward statistic for selected action
         global rewardPerChoice
         rewardPerChoice[choice] += reward
         
         #update agent's choice statistics and 
         #check agent's choice for convergence to the best action
         agentsChoiceFrequencies[exp3_agent.agent_index][choice] += 1
         if (t > 5000) and (agentsConvergenceCheck[exp3_agent.agent_index]):
            freqBound = float(t) * convergenceBound
            if agentsChoiceFrequencies[exp3_agent.agent_index][choice] > freqBound:
               print("Round %i Agent %i converged to choice %i\n" % (t,exp3_agent.agent_index,choice))
               weights_string = map(str, exp3_agent.getWeights())
               print("Agent %i weights: [%s]\n" % (exp3_agent.agent_index, ",".join(weights_string)))
               print("Effective reward per choice: [%s]\n" % (", ".join(map(str, rewardPerChoice))))
               #clear reward per choice counter
               del rewardPerChoice[:]
               rewardPerChoice = [0 for _ in range(numActions)]
               agentsConvergenceCheck[exp3_agent.agent_index] = False
      t += 1
      if t >= numRounds:
         break
   #f.close()
   fc.close()
   
   print("--------------------------")
   

   print datetime.datetime.now()
   print("--- Elapsed %s seconds ---" % (time.time() - start_time))
   print("Done.\n")
   for agent_index in range(numAgents):
        regretWeightsGraph2("results/EXP3_Thread-%d.csv" %(agent_index),"EXP3_Thread-%d" %(agent_index))


if __name__ == "__main__":
   for experiment_iter in range(numExperiments):
      rewardPerChoice = [0 for _ in range(numActions)]
      print("----- (BEGIN) Experiment nr. %i -----" % (experiment_iter,))
      exp3threadTest(experiment_iter)
      print("----- (END) Experiment nr. %i -----" % (experiment_iter,))
      #clear reward vector
      del rewardVector[:]
      #clear reward counter for experiment
      del rewardPerChoice[:]
import math
import numpy as np
import random

class MaxEntUnsupervised:

    def __init__(self, observations, states, iternum):

        try:
            checknp = np.__name__ == type(observations).__module__
            if not checknp:
                raise TypeError('Error: Require numpy array input')
        except TypeError:
            print('Error: Require numpy array input')
            raise

        try:
            self.numobservations = observations.shape[0]
            self.numbinfeat = observations.shape[1]
        except IndexError:
            print("Error: The input data does not have the same length")
            raise

        try:
            checkval=np.all(np.logical_or(observations == 0, observations == 1))
            if not checkval:
                raise ValueError('Error: Input data needs to have value 0 or 1')
        except ValueError:
            print('Error: Input data needs to have value 0 or 1')
            raise

        self.observations = observations
        self.states = np.arange(states)
        self.features = [[i,j] for i in range(0,self.numbinfeat) for j in self.states]
        self.lambdalist = np.zeros(shape = (self.numbinfeat,states))
        self.iternum = iternum

    def randlambda(self):
        for i in range(0,self.numbinfeat):
            self.lambdalist[i] = np.array(np.random.uniform(0,1, size = len(self.states)))

    def featureeval(self, xvalue, feature):
        if xvalue[1] == feature[1]:
            return xvalue[0][feature[0]]
        else:
            return 0

    def computenormalizerdp(self, lambdas = None, states = None):

        if lambdas is None:
            lambdas = self.lambdalist
        if states is None:
            states = self.states

        normalizer = 0
        for state in states:
            currentsum = np.zeros(len(states))
            currentsum[0] = 1
            for i in range(0,self.numbinfeat):
                prevsum = np.sum(currentsum)
                currentsum[0] = math.exp(0)*prevsum
                currentsum[1] = math.exp(lambdas[i][state])*prevsum
            normalizer += np.sum(currentsum)
        return normalizer

    def computemarginal(self,  observation, lambdas = None, states = None):

        if lambdas is None:
            lambdas = self.lambdalist
        if states is None:
            states = self.states

        marginalprobs = np.zeros(len(states))
        for state in states:
            marginalprobs[state] = math.exp(np.dot(observation,lambdas[:,state]))
        return marginalprobs/ np.sum(marginalprobs)

    def computedataeta(self, features = None, lambdas = None, observations = None, states = None):

        if features is None:
            features = self.features
        if lambdas is None:
            lambdas = self.lambdalist
        if observations is None:
            observations = self.observations
        if states is None:
            states = self.states

        times = np.zeros(2)

        etafeat = np.zeros(len(features))
        for i in range(0,len(features)):
            numerator = np.zeros(len(states))
            numsum = 0
            for obs in observations:
                t0 = time.time()
                for state in states:
                    xvalue = [obs, state]
                    numerator[state] = self.featureeval(xvalue, features[i])
                t1 = time.time()
                times[0] += t1 - t0
                numsum += np.dot(numerator, self.computemarginal(obs, lambdas, states))
                t2 = time.time()
                times[1] += t2 - t1
            etafeat[i] = numsum
        print(times/len(features)/len(observations))
        return etafeat/len(observations)

    def computefeatavg(self, features = None, lambdas = None, states = None):

        if features is None:
            features = self.features
        if lambdas is None:
            lambdas = self.lambdalist
        if states is None:
            states = self.states

        denom = self.computenormalizerdp(lambdas, states)
        featavg = np.zeros(len(features))
        for i in range(0, len(features)):
            featsum = 0
            curfeatavg = 0
            curbin = features[i][0]
            curstate = features[i][1]
            currentsum = np.zeros(len(states))
            currentsum[0] = 1
            for j in range(0, self.numbinfeat):
                prevsum = np.sum(currentsum)
                if j == curbin:
                    currentsum[0] = math.exp(lambdas[j][curstate])*prevsum/len(states)
                    currentsum[1] = math.exp(lambdas[j][curstate])*prevsum/len(states)
                else:
                    currentsum[0] = math.exp(0)*prevsum
                    currentsum[1] = math.exp(lambdas[j][curstate])*prevsum
            featsum += np.sum(currentsum)
            featavg[i] = featsum
        return featavg/denom

    def computeentropy(self, denominator, eta, lambdas = None):
        if lambdas is None:
            lambdas = self.lambdalist
        return math.log(denominator) - np.dot(eta, np.resize(lambdas, (self.numbinfeat*len(self.states),1)))

    def computelambda(self, features = None, lambdas = None, observations = None, states = None, iternum = None):

        if features is None:
            features = self.features
        if lambdas is None:
            lambdas = self.lambdalist
        if observations is None:
            observations = self.observations
        if states is None:
            states = self.states
        if iternum is None:
            iternum = self.iternum

        iterlambdas = lambdas
        times = np.zeros(4)
        for i in range(0, iternum):
            t0 = time.time()
            cureta = self.computedataeta(features, iterlambdas, observations, states)
            t1 = time.time()
            times[0] += t1 - t0
            curavg = self.computefeatavg(features, iterlambdas, states)
            t2 = time.time()
            times[1] += t2 - t1
            itratio = np.divide(cureta, curavg)
            itratiore = np.resize(itratio, (self.numbinfeat,len(states)))
            t3 = time.time()
            times[2] += t3 - t2
            iterlambdanew = np.add(iterlambdas,1/self.numbinfeat*np.log(itratiore))
            iterlambdas = iterlambdanew
            t4 = time.time()
            times[3] += t4 - t3
        print("times per loop", times/iternum)
        print("time for one computation", times)
        print("total time", np.sum(times))
        return iterlambdas

    def computeobsstate(self):
        maxentropy = 0
        maxentlambdas = np.zeros(shape = (self.numbinfeat,len(self.states)))
        resultstate = np.zeros(self.numobservations)
        for iteration in range(0,100):
            self.randlambda()
            self.lambdalist = self.computelambda()
            denom = self.computenormalizerdp()
            cureta = self.computedataeta()
            curavg = self.computefeatavg()
            entropy = self.computeentropy(denom, cureta)
            for i in range(0, self.numobservations):
                resultstate[i] = self.computemarginal(self.observations[i]).argmax()
            if entropy > maxentropy:
                maxentropy = entropy
                maxentlambdas = np.copy(self.lambdalist)
        self.lambdalist = np.copy(maxentlambdas)
        resultstate = np.zeros(self.numobservations)
        for i in range(0, self.numobservations):
            resultstate[i] = self.computemarginal(self.observations[i]).argmax()
        return resultstate

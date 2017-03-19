import math
import numpy as np
import random
import skmonaco

class MaxEntCont:

    def __init__(self, observations, states, iternum, featfuncs):

        try:
            checknp = np.__name__ == type(observations).__module__
            if not checknp:
                raise TypeError('Error: Require numpy array input')
        except TypeError:
            print('Error: Require numpy array input')
            raise

        try:
            self.numobservations = observations.shape[0]
            self.numvar = observations.shape[1]
            self.numfeat = len(featfuncs)
        except IndexError:
            print("Error: The input data does not have the same length")
            raise

        observedfeats = np.zeros(shape = (len(observations), len(featfuncs)))
        for i in range(0, len(observations)):
            tempobsfts = np.zeros(len(featfuncs))
            for j in range(0, len(featfuncs)):
                tempobsfts[j] = featfuncs[j](observations[i])
            observedfeats[i] = tempobsfts

        self.observations = observations
        self.observedfeats = observedfeats
        self.states = np.arange(states)
        self.features = [[i,j] for i in range(0,self.numfeat) for j in self.states]
        self.lambdalist = np.zeros(shape = (self.numfeat,states))
        self.iternum = iternum
        self.featfuncs = featfuncs

    def randlambda(self):
        for i in range(0,self.numfeat):
            self.lambdalist[i] = np.array(np.random.uniform(-1,1, size = len(self.states)))

    def featureeval(self, featfunc, xvalue, feature):
        if xvalue[1] == feature[1]:
            return featfunc(xvalue[0])
        else:
            return 0

    def p(self, inputs, state):
        featvector = np.zeros(self.numfeat)
        for i in range(0, self.numfeat):
            featvector[i] = self.featfuncs[i](inputs)
        return np.exp(np.dot(self.lambdalist[:,state], featvector))

    def fp(self, featvector, a,b):
        return self.p(featvector,b)*featvector[a]

    def fep(self, inputs, a,b, curgamma):
        featvector = np.zeros(self.numfeat)
        for i in range(0, self.numfeat):
            featvector[i] = self.featfuncs[i](inputs)
        fsum = np.sum(featvector)
        return self.p(inputs, b)*featvector[a]*math.exp(fsum*curgamma)

    def ffep(self, inputs, a,b, curgamma):
        featvector = np.zeros(self.numfeat)
        for i in range(0, self.numfeat):
            featvector[i] = self.featfuncs[i](inputs)
        fsum = np.sum(featvector)
        return self.p(inputs, b)*featvector[a]*math.exp(fsum*curgamma)*fsum

    def computenormalizerdp(self, lambdas = None, states = None):

        if lambdas is None:
            lambdas = self.lambdalist
        if states is None:
            states = self.states

        normalizer = 0
        integrand = 0
        for state in states:
            integrand += skmonaco.mcquad(self.p, xl = np.zeros(self.numvar), xu = np.ones(self.numvar), args = ([state]), npoints = 100000)[0]
        return integrand

    def computemarginal(self,  observation, lambdas = None, states = None):

        if lambdas is None:
            lambdas = self.lambdalist
        if states is None:
            states = self.states

        observedfeats = np.zeros(self.numfeat)
        for i in range(0, self.numfeat):
            observedfeats[i] = self.featfuncs[i](observation)

        marginalprobs = np.zeros(len(states))
        for state in states:
            marginalprobs[state] = math.exp(np.dot(observedfeats,lambdas[:,state]))
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

        etafeat = np.zeros(len(features))
        for i in range(0,len(features)):
            numerator = np.zeros(len(states))
            numsum = 0
            for obs in observations:
                for state in states:
                    xvalue = [obs, state]
                    numerator[state] = self.featureeval(self.featfuncs[features[i][0]], xvalue, features[i])
                numsum += np.dot(numerator, self.computemarginal(obs, lambdas, states))
            etafeat[i] = numsum
        return etafeat/len(observations)

    def computeMstep(self, etalist, features = None, lambdas = None, states = None):

        if features is None:
            features = self.features
        if lambdas is None:
            lambdas = self.lambdalist
        if states is None:
            states = self.states

        denom = self.computenormalizerdp(lambdas, states)

        gammastep = np.zeros(len(features))

        for iteration in range(0,2):
            for i in range(0, len(features)):
                a = features[i][0]
                b = features[i][1]
                gammastep[i] += -1*((skmonaco.mcquad(self.fep, xl = np.zeros(self.numvar),
                                    xu = np.ones(self.numvar), args = ([a,b,gammastep[i]]),
                                    npoints = 100000)[0] - denom*etalist[i])/
                                    (skmonaco.mcquad(self.ffep, xl = np.zeros(self.numvar),
                                    xu = np.ones(self.numvar), args = ([a,b,gammastep[i]]),
                                    npoints = 100000)[0]))
            print(gammastep)

        return gammastep


#    def computeentropy(self, denominator, eta, lambdas = None):
#        if lambdas is None:
#            lambdas = self.lambdalist
#        return math.log(denominator) - np.dot(eta, np.resize(lambdas, (self.numbinfeat*len(self.states),1)))

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
        for i in range(0, iternum):
            cureta = self.computedataeta(features, iterlambdas, observations, states)
            print("eta is:", cureta)
            curgamma = self.computeMstep(cureta, features, iterlambdas, states)
            print("gamma is:", curgamma)
            curgammare = np.resize(curgamma, (self.numfeat,len(states)))
            iterlambdanew = np.add(iterlambdas,curgammare)
            iterlambdas = iterlambdanew
            print(iterlambdas)
            self.lambdalist = iterlambdas
        return iterlambdas

    def computeobsstate(self):
        maxentropy = 0
        maxentlambdas = np.zeros(shape = (self.numfeat,len(self.states)))
        resultstate = np.zeros(self.numobservations)
        for iteration in range(0,1):
            self.randlambda()
            print(self.lambdalist)
            self.lambdalist = self.computelambda()
        resultstate = np.zeros(self.numobservations)
        for i in range(0, self.numobservations):
            resultstate[i] = self.computemarginal(self.observations[i]).argmax()
        return resultstate

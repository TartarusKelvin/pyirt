import numpy as np

class PCM:
    def __init__(self,max_mark = 1,discrimination = None,boundaries = []):
        self.max_mark = int(max_mark)
        self.alpha = discrimination
        self.boundaries = boundaries
        if len(self.boundaries) != self.max_mark and self.boundaries != []:
            raise ValueError(f"Boundaries ({len(self.boundaries)}) must be of the same size as max_mark ({self.max_mark})")
        if self.boundaries == []:
            self.boundaries = [None] * self.max_mark

    def predict(self, theta, mark=None):
        if mark != None and mark > self.max_mark:
            raise ValueError(f"Mark must not exceed {self.max_mark}")
        probabilties = []

        theta_bound_diff = [theta - bound for bound in self.boundaries]
        bottom_part = sum([np.exp(self.alpha * sum(theta_bound_diff[:r])) for r in range(self.max_mark+1)])

        for m in range(self.max_mark+1):
            top_part = np.exp(self.alpha*sum(theta_bound_diff[:m]))
            probabilties.append(top_part/bottom_part)
        if mark == None:
            return probabilties
        else:
            return probabilties[mark]

    @classmethod
    def train(cls,data,max_mark=None,max_step_count=5000):
        """ Data comes in the Form:
              R1  R2  R3  .. Rn
        Item: 0   3   1   .. 2
        Theta:1   0.3 0.4 .. -3
        """
        ## assume the max mark achieved was the max available
        if max_mark == None:
            max_mark = max(data[0])
        elif max(data[0]) > max_mark:
            raise ValueError("There is a response which has a higher mark than what has been set as achievable")

        responses = np.zeros((max_mark+1, len(data[0])))
        for r in range(len(data[0])):
            responses[data[0][r],r] = 1

        ## Start our guesses for params randomly along the noraml dist
        g_alpha      = np.random.normal(0, 1, 1)[0]
        g_boundaries = np.random.normal(0, 1, max_mark)

        learn_rate = 0.01
        dx = 0.001
        
        def calc_cost(alpha,bounds):
            current_model = cls(max_mark=max_mark,discrimination=alpha,boundaries=list(bounds))
            cost = 0
            for i in range(len(data[1])):
                cost += np.sum(np.square(responses[...,i] - current_model.predict(data[1][i])))/len(data[1])
            return np.log(cost)

        def nabla():
            current = calc_cost(g_alpha,g_boundaries)
            d_alpha = (calc_cost(g_alpha + dx,g_boundaries) - current)/dx
            d_boundaries = []
            for i in range(len(g_boundaries)):
                dx_array = np.zeros(len(g_boundaries))
                dx_array[i] = dx
                d_boundaries.append((calc_cost(g_alpha,g_boundaries + dx_array) - current)/dx)
            return d_alpha, d_boundaries


        for i in range(max_step_count):
            if i%1000 == 0:
                print (f"{calc_cost(g_alpha,g_boundaries)} ({learn_rate})")
            learn_rate = 1.0/np.sqrt(1+i)
            d_alpha , d_boundaries = nabla()
            g_alpha -= learn_rate * d_alpha
            g_boundaries -= learn_rate * np.array(d_boundaries)
            

        return cls(max_mark=max_mark,discrimination=g_alpha,boundaries=list(g_boundaries))




    def show(self):
        import matplotlib.pyplot as plt
        Theta = np.linspace(-8,8,600)
        ys = [[self.predict(theta,mark=m) for theta in Theta] for m in range(self.max_mark+1)]
        count = 0
        for y in ys:
            plt.plot(Theta,y,label=count)
            count +=1
        plt.legend()
        plt.show()

    def expected(self,theta):
        return sum([(mark * self.predict(theta,mark=mark)) for mark in range(self.max_mark +1)])

    def __str__(self):
        return f"General Partial Credit Model With Parameters:\n- Alpha: {self.alpha}\n- Boundaries: {self.boundaries}"


print("Running")
Model = PCM.train([[1,4,0,2],[-1,2,0,0.6]],max_mark=4)
print(Model)
Model.show()
print("Done")
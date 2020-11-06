import numpy as np

class TPM:
    def __init__(self,alpha,beta,gamma):
        # Alpha: discrimination
        # Beta: Bias
        # Gamma: Guessing Coef

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def predict(self,theta):
        return self.gamma +(1-self.gamma)*np.exp(self.alpha*(theta - self.beta))/(1+np.exp(self.alpha*(theta - self.beta)))

    @classmethod
    def fit(cls,responses,thetas,max_step_count=9000):
        g_alpha = np.random.normal(0, 1, 1)[0]
        g_beta  = np.random.normal(0, 1, 1)[0]
        g_gamma = np.random.normal(0, 1, 1)[0]
        
        learn_rate = 0.01
        dx = 0.001

        def likelihood (alpha,beta,gamma):
            current_model = cls(alpha,beta,gamma)
            p = 1
            for i in range(len(thetas)):
                if responses[i] == 0:
                    p*= 1 -current_model.predict(thetas[i])
                else:
                    p*= current_model.predict(thetas[i])   
            return np.log(p)
        
        def nabla():
            current = likelihood(g_alpha,g_beta,g_gamma)
            d_alpha = (likelihood(g_alpha+dx,g_beta,g_gamma)-current)/dx 
            d_beta  = (likelihood(g_alpha,g_beta+dx,g_gamma)-current)/dx
            d_gamma = (likelihood(g_alpha,g_beta,g_gamma+dx)-current)/dx
            return d_alpha, d_beta, d_gamma

        for i in range(max_step_count):
            if i%1000 == 0:
                print(f"{likelihood(g_alpha,g_beta,g_gamma)} ({learn_rate})")
            learn_rate = 1.0/np.log(3+i)
            d_alpha, d_beta, d_gamma = nabla()
            g_alpha += learn_rate*d_alpha
            g_beta  += learn_rate*d_beta
            g_gamma += learn_rate*d_gamma

        return cls(g_alpha,g_beta,g_gamma)


    def show(self):
        import matplotlib.pyplot as plt
        Theta = np.linspace(-8,8,600)
        ys = [[self.predict(theta) for theta in Theta]]
        count = 0
        for y in ys:
            plt.plot(Theta,y,label=count)
            count +=1
        plt.show()

    def __str__(self):
        return f"Three Parameter IRT Model With Parameters:\n- Alpha: {self.alpha}\n- Beta: {self.beta}\n- Gamma: {self.gamma}"

TPM.fit([0,1,1,0,0,1,1,0],np.random.normal(0, 1, 8)).show()
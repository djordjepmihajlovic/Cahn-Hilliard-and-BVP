from CH import Update
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Visual(object): #Set up class to animate by calling CH
    
    def __init__(self, phi, N, M, a, k, dx, dt):
        
        self.model = Update(phi, N, M, a, k, dx, dt)
        
        self.fig, self.ax = plt.subplots()        
        self.implot = self.ax.imshow(self.model.order_param, cmap = "cool")
        self.ani = None # For storing animation object
        
    def run(self):

        self.ani = animation.FuncAnimation(self.fig, self.animate,
                                           interval=1, blit=True)
        plt.show()

        
    def animate(self, frame): #This determines each frame of animation
        
        for i in range(0, 1000):
            self.model.C_P()
            self.model.O_P()
            

        self.implot.set_data(self.model.order_param)

        return self.implot,
    
    
def main():
    print("Cahn Hilliard: Initial value problem: Water-oil emulsion simulator. ")
    
    dim = int(input("Dimension of simulation = "))
    phi = float(input("Initial phi value = "))
    M = float(input("Parameter value M = "))
    a = float(input("Parameter value a = "))
    k = float(input("Parameter value k = "))
    
    
    B = Visual(phi, dim, M, a, k, 1, 1)
    B.run()
    
main()
# MCMCS Coursework 1
# Luc Berthouze 2018-10-23

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from random import seed
import random
# seed random number generator
seed(1)

# Definition of Complex landscape
def ComplexLandscape(x, y):
    return 4*(1-x)**2*np.exp(-(x**2)-(y+1)**2) -15*(x/5 - x**3 - y**5)*np.exp(-x**2-y**2) -(1./3)*np.exp(-(x+1)**2 - y**2)-1*(2*(x-3)**7 -0.3*(y-4)**5+(y-3)**9)*np.exp(-(x-3)**2-(y-3)**2)

# Definition of gradient of Complex landscape
def ComplexLandscapeGrad(x, y):
    g = np.zeros(2)
    g[0]=-8* np.exp(-(x**2)-(y+1)**2)*((1-x)+x*(1-x)**2)-15* np.exp(-x**2-y**2)*((0.2-3*x**2) -2*x*(x/5 - x**3 - y**5)) +(2./3)*(x+1)* np.exp(-(x+1)**2 - y**2)-1* np.exp(-(x-3)**2-(y-3)**2)*(14*(x-3)**6-2*(x-3)*(2*(x-3)**7-0.3*(y-4)**5+(y-3)**9))
    g[1]=-8*(y+1)*(1-x)**2* np.exp(-(x**2)-(y+1)**2) -15* np.exp(-x**2-y**2)*(-5*y**4 -2*y*(x/5 - x**3 - y**5)) +(2./3)*y* np.exp(-(x+1)**2 - y**2)-1* np.exp(-(x-3)**2-(y-3)**2)*((-1.5*(y-4)**4+9*(y-3)**8)-2*(y-3)*(2*(x-3)**7-0.3*(y-4)**5+(y-3)**9))
    return g
    
# Definition of Simple landscape
def SimpleLandscape(x, y):
    return np.where(1-np.abs(2*x)>0,1-np.abs(2*x)+x+y,x+y)
    
# Definition of gradient of Simple landscape
def SimpleLandscapeGrad(x, y):
    g = np.zeros(2)
    if 1 - np.abs(2 * x) > 0:
        if x < 0:
            g[0] = 3
        elif x == 0:
            g[0] = 0
        else:
            g[0] = -1
    else:
        g[0] = 1
    g[1] = 1
    return g

# Function to draw a surface (equivalent to ezmesh in Matlab)
# See argument cmap of plot_surface instruction to adjust color map (if so desired)
def DrawSurface(fig, varxrange, varyrange, function):
    """Function to draw a surface given x,y ranges and a function."""
    ax = fig.gca(projection='3d')
    xx, yy = np.meshgrid(varxrange, varyrange, sparse=False)
    z = function(xx, yy)
    ax.plot_surface(xx, yy, z, cmap='RdBu') # color map can be adjusted, or removed! 
    fig.canvas.draw()
    return ax

i=0
# Function implementing gradient ascent
def GradAscent(StartPt,NumSteps,LRate):
    PauseFlag=1
    for i in range(NumSteps):
        # TO DO: Calculate the 'height' at StartPt using SimpleLandscape or ComplexLandscape
        height = ComplexLandscape(StartPt[0], StartPt[1])
        # TO DO: Plot point on the landscape 
        plt.scatter(StartPt[0],StartPt[1],color='k',s=10)
        # Use a markersize that you can see well enough (e.g., * in size 10)
        plt.xlabel("X")
        plt.xlabel("Y")
        plt.title('Task2: Graph of X and Y for GradAscent using ComplexLandscape ')
        plt.show()
        # TO DO: Calculate the gradient at StartPt using SimpleLandscapeGrad or ComplexLandscapeGrad
        gradient=ComplexLandscapeGrad(StartPt[0], StartPt[1])
        # TO DO: Calculate the new point and update StartPt
        StartPt[0] = StartPt[0] + LRate * gradient
        StartPt[1] = StartPt[1] + LRate * gradient
        # Ensure StartPt is within the specified bounds (un/comment relevant lines)
        #StartPt = np.maximum(StartPt,[-2,-2])
        #StartPt = np.minimum(StartPt,[2,2])
        StartPt = np.maximum(StartPt,[-3,-3])
        StartPt = np.minimum(StartPt,[7,7])
        print("Epoch")
        print(i)
        i= i+1
        
        # Pause to view output
        if PauseFlag:
            x=plt.waitforbuttonpress()



# TO DO: Mutation function
# Returns a mutated point given the old point and the range of mutation
def Mutate(OldPt,MaxMutate):
    #TO DO: Select a random distance MutDist to mutate in the range (-MaxMutate,MaxMutate)
    MutDist = random.uniform(-MaxMutate,MaxMutate)
    #TO DO: Randomly choose which element of OldPt to mutate and mutate by MutDist
    MutatedPt = random.randint(0,len(OldPt)-1)
    #OldPt[MutatedPt] = MutDist
    return MutatedPt
    

# Function implementing hill climbing
def HillClimb(StartPt,NumSteps,MaxMutate):
    PauseFlag=1
    for i in range(NumSteps):
        # TO DO: Calculate the 'height' at StartPt using SimpleLandscape or ComplexLandscape
        height = ComplexLandscape(StartPt[0], StartPt[1])
        # TO DO: Plot point on the landscape 
        plt.scatter(StartPt[0],height,color='k',s=10)
        # Use a markersize that you can see well enough (e.g., * in size 10)
        plt.xlabel("X")
        plt.xlabel("Y")
        plt.title('Task2: Graph of X and Y for HillClimb using ComplexLandscape')
        plt.show()
        # Use a markersize that you can see well enough (e.g., * in size 10)
        
        # Mutate StartPt into NewPt
        NewPt = Mutate(np.copy(StartPt),MaxMutate) # Use copy because Python passes variables by references (see Mutate function)
        
        # Ensure NewPt is within the specified bounds (un/comment relevant lines)
        NewPt = np.maximum(NewPt,[-2,-2])
        NewPt = np.minimum(NewPt,[2,2])
        #NewPt = np.maximum(NewPt,[-3,-3])
        #NewPt = np.minimum(NewPt,[7,7])
               
        # TO DO: Calculate the height of the new point
        height = ComplexLandscape(StartPt[0], StartPt[1])        
        # TO DO: Decide whether to update StartPt or not
        if NewPt[1] > StartPt[1]:
        	StartPt[0] = NewPt[0]
         
        # Pause to view output
        if PauseFlag:
            x=plt.waitforbuttonpress()
        

# Template 
# Plot the landscape (un/comment relevant line)
plt.ion()
fig = plt.figure()
#ax = DrawSurface(fig, np.arange(-2,2.025,0.025), np.arange(-2,2.025,0.025), SimpleLandscape)
ax = DrawSurface(fig, np.arange(-3,7.025,0.025), np.arange(-3,7.025,0.025), ComplexLandscape)

# Enter maximum number of iterations of the algorithm, learning rate and mutation range
NumSteps=50
LRate=0.1
MaxMutate=1

# TO DO: choose a random starting point with x and y in the range (-2, 2) for simple landscape, (-3,7) for complex landscape
#StartPt = [-2,2]
#For Complex landscape
StartPt = [-3,7]
# Find maximum (un/comment relevant line)
GradAscent(StartPt,NumSteps,LRate)
#HillClimb(StartPt,NumSteps,MaxMutate)













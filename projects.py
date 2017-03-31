from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt, zeros, ones, array, linspace, logspace
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
filename='abcd.txt'
data=np.loadtxt(filename)

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
w1=data[:,0]
w2=data[:,1]
w3=data[:,2]
ax.scatter(w1,w2,w3, marker='o', c='b')

ax.set_xlabel(' skills 1')
ax.set_ylabel('skills 2')
ax.set_zlabel('probability of Placement')
plt.show()

row,col=data.shape

def normalize_data(x):
    x_norm=x;
    for i in np.arange(col):
        m=np.mean(x[:,i])
        s=np.std(x[:,i])
        if(s==0):
            s=1
        x_norm[:,i]=(x_norm[:,i]-m)/s
    
    return x_norm
def compute_cost(theta,x,y):
    m=y.size
    prediction=x.dot(theta)
    
    sq_error=(prediction-y)**2
    
    error_sum=np.sum(sq_error)
    
    J=(1.0/(2*m))*error_sum
    
    return J

def gradient_descent(theta,x,y,alpha,total_itr):

    m=y.size
    J_history=np.zeros((total_itr,1))
    n=theta.size
    row,col=x.shape
    

    
    for i in np.arange(total_itr):
        prediction=x.dot(theta)
        error_prediction=(prediction-y).flatten();
        
        for j in np.arange(n):
            sq_error=error_prediction.dot(x[:,j])
            theta[j][0]=theta[j][0]-((alpha*(1.0/m)*np.sum(sq_error)))
        
        J_history[i][0]=compute_cost(theta,x,y);
    
    return theta,J_history

def main():
    
    
    total_itr=150
    z=normalize_data(data);
    x=data[:,:col-1]
    y=data[:,col-1:col]
    it=np.ones((row,col))
    
    for i in np.arange(row):
        for j in np.arange(col-1):
            it[i][j+1]=x[i][j]
    
    alpha=0.001
    theta=np.zeros((col,1))
    
    m=y.size
    
    theta,J_theta=gradient_descent(theta,it,y,alpha,total_itr);
           
    
    

        
 
 
 

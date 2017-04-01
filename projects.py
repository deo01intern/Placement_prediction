from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt, zeros, ones, array, linspace, logspace
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
filename='Book11.txt'
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
    mean_x=[]
    std_x=[]
    n=x.shape[1]
    for i in np.arange(n):
        m=np.mean(x[:,i])
        s=np.std(x[:,i])
        mean_x.append(m)
        std_x.append(s)
        x_norm[:,i]=(x_norm[:,i]-m)/s
    
    return mean_x,std_x,x_norm
def compute_cost(theta,x,y):
    m=y.size
    prediction=x.dot(theta)
    
    sq_error=(prediction-y)**2
    
    
    J=(1.0/(2*m))*sq_error.sum()
    
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


    
    
total_itr=600

x=data[:,:col-1]
y=data[:,col-1:col]
mean_r,std_r,z=normalize_data(x);
    
it=np.ones((row,col))
    
for i in np.arange(row):
    for j in np.arange(col-1):
        it[i][j+1]=data[i][j]
    
alpha=0.01

theta=np.zeros((col,1))
    
m=y.size
    
theta,J_theta=gradient_descent(theta,it,y,alpha,total_itr);
#plot cost vs iteration
plot(np.arange(total_itr), J_theta)
xlabel('Iterations')
ylabel('Cost Function')
show()

    
#Predict probaliity of placemet

print 'Rate your skills out of 100'
print '1: codin skills', '\n','2:Aptitude','\n','3: Technical','\n','4:Communication','\n','5:Core Knowledge','\n','6: Presentation skills'
print  '7:Academic','\n','8: Puzzle solving','\n','9:English prociency','\n','10: programming skills','\n','11:Management','\n','12:projects'
print '13:Internship','\n','14:Training','\n','15:backlog'
res=np.ones((1,16));
while(True):
    for i in np.arange(15):
        xx=input()
        res[0][i]=((xx-mean_r[i])/std_r[i])
    placed = res.dot(theta)
    print 'probability of getting Placement is : %f' % (placed)
    print 'Press 1 to Continue or 0 to exit'
    yy=input()
    if(yy==0):
        break
    

    

        
 
 
 

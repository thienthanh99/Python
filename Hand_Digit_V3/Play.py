import numpy as np
from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib



mnist = fetch_mldata('mnist-original', data_home='./')
N, d= mnist.data.shape
x_all= mnist.data
y_all= mnist.target
print(N)
print(d)
#show some number in data set

plt.imshow(x_all.T[:,3000].reshape(28,28))
plt.axis("off")
plt.show()
# luc lai chi con 2 chu so 0 va 1
X0=x_all[np.where(y_all==0)[0]]
X1=x_all[np.where(y_all==1)[0]]
y0=np.zeros(X0.shape[0])
y1=np.ones(X1.shape[0])

#gop 0 va 1 lai thanh 1 dataset
X=np.concatenate((X0, X1),axis=0)
Y=np.concatenate((y0, y1))
one =np.ones((X.shape[0],1))
X = np.concatenate((X,one), axis=1)

def gradent_decent(X,y, theta_inist, eta=0.05):
    theta_old = theta_inist
    theta_epoch = theta_inist
    N = X.shape[0]
    for it in range(10000):
        mrx_id = np.random.permutation(N)
        for i in mrx_id:
            xi = X[i,:]
            yi = y[i]
            hi = 1.0/ (1+np.exp(-np.dot(xi, theta_old.T)))
            gi = (yi - hi)* xi
            theta_new = theta_old + eta*gi
            theta_old= theta_new

        if np.linalg.norm(theta_epoch-theta_old)< 1e-1:
            break

        theta_epoch = theta_old
        return theta_epoch, it

theta_init= np.random.randn(1, X.shape[1])[0]
# print (theta_init)
theta, it= gradent_decent(X,Y,theta_init)
print("nghiem:", theta)
print("it:", it)

np.savetxt('theta.txt', theta)

import numpy as np
import math
import array as arr
x = np.array([[0.50,0.75,1.00,1.25,1.50,1.75,2.00,2.25,2.50,2.75,3.00,3.25,3.50,3.75,4.00,4.25,4.50,4.75,5.00,5.50]]).T
y = np.array([0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1]).T
one = np.ones((x.shape[0],1))
x= np.concatenate((x,one),axis=1)
# print(x)

def gradient_descent(x, y, theta_init, eta = 0.05):
    theta_old = theta_init
    theta_epoch = theta_init
    N = x.shape[0]
    for it in range(10000):
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = x[i,:]
            yi = y[i]
            hi = 1.0/(1.0 +np.exp(-1.0 * np.dot(xi,theta_old.T)))
            gi = (yi-hi)*xi
            theta_new = theta_old + eta*gi
            theta_old = theta_new
        if np.linalg.norm(theta_epoch-theta_old) < 1e-4:
            break
        theta_epoch = theta_old
    return theta_epoch, it 


#theta = arr.array('i',gradient_descent(x,y,))
theta_init = np.random.randn(1,x.shape[1])[0]
theta, it = gradient_descent(x, y, theta_init)

print ("nghiem:",theta)


hours = float(input('nhap vao hours:'))
predict = 1.0/(1.0 +np.exp(-1.0 * (theta[0]*hours +theta[1])))

print("dau rot la: ",predict)
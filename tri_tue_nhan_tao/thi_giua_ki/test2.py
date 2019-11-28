from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import mahotas
import cv2
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from sklearn.preprocessing import MinMaxScaler

features = np.loadtxt("feature-datas.txt")
# print(features)
labels = np.loadtxt("label-datas.txt")
# print(labels, 'y')
## Train Mo hinh
# V1
# model = SVC(gamma='auto', random_state=9)
# model.fit(features, labels)

bins = 8
# End V1

# Begin V2

X = features
# y = labels
y = labels
# y = np.concatenate((-1*np.ones((1, 80)), np.ones((1, 82))), axis = 1)

# V = X.T*y
V = X.T*y

K = matrix(V.T.dot(V))
K = matrix(K, K.size, 'd')

N = K.size[1]

p = matrix(-np.ones((N, 1)))
G = matrix(-np.eye(N))
h = matrix(np.zeros((N, 1)))
A = matrix(np.array(y))
A = A.T
# print(A.size)

b = matrix(np.zeros((1, 1)))
# print(b.size)
# print('=================')
# print(K, 'K')
# print(p, 'P')
# print(G, 'G')
# print(h, 'h')
# print(A, 'A')
# print(b, 'b')
# print('=================')
solvers.options['show_progress'] = False
sol = solvers.qp(K, p, G, h, A, b)

l = np.array(sol['x'])
# print(l.size, 'l size')
# print('lambda = ')

epsilon = 1e-6
S = np.where(l > epsilon)[0]
# print(S.shape, 'S shape')
# print(X, 'X')
# print(X.size, 'X size')
# print(V.shape, 'V shape')
VS = V.T[:, S]
# print(VS.shape, 'VS shape')
XS = X[:, S]
# print(XS.shape, 'XS shape')

y = np.matrix(y)
yS = y[:, S]
# print(yS, 'yS')

lS = l[S]
# print(lS.shape, 'lS shape')

w = VS.dot(lS)
print(w, 'w')
print(w.shape, 'w shape')
# print(w.T.dot(XS), 'w.dot(XS)')
b = np.mean(yS.T - w.T.dot(XS))

print('b = ', b)

# End V2

# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

fixed_size = tuple((500, 500))

image = cv2.imread('image_0001.jpg')

# resize the image
image = cv2.resize(image, fixed_size)

    ####################################
    # Global Feature extraction
    ####################################
fv_hu_moments = fd_hu_moments(image)
fv_haralick   = fd_haralick(image)
fv_histogram  = fd_histogram(image)

    ###################################
    # Concatenate global features
    ###################################
global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

# scale features in the range (0-1)
scaler = MinMaxScaler(feature_range=(-1, 1))
rescaled_feature = scaler.fit_transform(global_feature.reshape(-1, 1))

# predict label of test image
# V1
# prediction = model.predict(rescaled_feature.reshape(1,-1))[0]
# print('raw global feature')
# print(global_feature)
# End V1
# print(prediction)

# Begin V2

x = global_feature.reshape(-1,1)
x = x[x != 0.0]
print(x.shape, 'x size')
# print(matrix(x[1], axis = 1), 'x 1')

sumWX = 0.0
sumW = 0.0
inrange = w.size
if x.size < w.size:
    inrange = x.size

for i in range(0, inrange - 1):
    sumWX += w[i] * x[i]
    sumW += w[i] * w[i]
print(float(sumWX))

print(np.sqrt(abs(int(sumWX))), 'sqrt sumwx')

mypredict = (float(sumWX) + b) / np.sqrt(sumW)

print('mypredict = ' , mypredict)
# End V2

label = 'Unknown'

if mypredict<=-1.0:
    label ='Hoa Trang'
elif mypredict>=1.0:
    label ='Hoa Vang'

print (label)
 
# show predicted label on image
cv2.putText(image, str(label), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

# display the output image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
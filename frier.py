import imutils
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import pylab

znccl = []
angles = []

picrott = cv2.imread('stereo-parallel-smaller1.tif')
picroott = cv2.imread('stereo-parallel-smaller1.tif')
pic = picrott#[80:400, 90:410]
picg = picroott#[80:400, 90:410]
picg[:,:,2]=0
#picg[:,:,1]=0

pic[:,:,0]=0
pic[:,:,2]=0
pic1 = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)

gray1 = cv2.GaussianBlur(pic1, (41, 41), 0)
(minVal1, maxVal1, minLoc1, maxLoc1) = cv2.minMaxLoc(gray1)
piic1 = pic1#[maxLoc1[1]-150:maxLoc1[1]+150, maxLoc1[0]-150:maxLoc1[0]+150]
pix1 = picg#[maxLoc1[1]-150:maxLoc1[1]+150, maxLoc1[0]-150:maxLoc1[0]+150]
img = cv2.GaussianBlur(piic1, (15, 15), 0)

img1 = piic1 - img
ret,th = cv2.threshold(img1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

median = cv2.medianBlur(th,5)
#median1 = cv2.medianBlur(median,5)
#median2 = cv2.medianBlur(median1,5)
#median3 = cv2.medianBlur(median2,5)
#median4 = cv2.medianBlur(median3,5)
f = np.fft.fft2(median)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
#median1 = cv2.medianBlur(median,5)

plt.subplot(121),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

ppic1 = cv2.imread('stereo-parallel-smaller2.tif')
ppiclol1 = cv2.imread('stereo-parallel-smaller2.tif')
pix2 = ppic1
pixrot = ppiclol1
pixrot[:,:,0]=0
pixrot[:,:,1]=0

pix2[:,:,0]=0
pix2[:,:,2]=0
pixx2 = cv2.cvtColor(pix2, cv2.COLOR_BGR2GRAY)

ppic1[:,:,0]=0
ppic1[:,:,2]=0
ppic11 = cv2.cvtColor(ppic1, cv2.COLOR_BGR2GRAY)

for angle in range(0, 360, 1):
  rotated = imutils.rotate_bound(ppic11, angle=angle)
  angles.append(angle)
  rotated = rotated#[80:400, 90:410]
  gray1 = cv2.GaussianBlur(rotated, (41, 41), 0)
  (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray1)
  ppicc11 = rotated[maxLoc[1]-150:maxLoc[1]+150, maxLoc[0]-150:maxLoc[0]+150]
  
  iimg1 = cv2.GaussianBlur(ppicc11, (15, 15), 0)

  iimg11 = ppicc11 - iimg1
  ret1,th1 = cv2.threshold(iimg11,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

  mmedian1 = cv2.medianBlur(th1,5)
  #mmedian11 = cv2.medianBlur(mmedian1,5)
  #mmedian12 = cv2.medianBlur(mmedian11,5)
  #mmedian13 = cv2.medianBlur(mmedian12,5)
  #mmedian14 = cv2.medianBlur(mmedian13,5)
  f1 = np.fft.fft2(mmedian1)
  fshift1 = np.fft.fftshift(f1)
  magnitude_spectrum1 = 20*np.log(np.abs(fshift1))
  #median1 = cv2.medianBlur(median,5)

  #plt.subplot(121),plt.imshow(magnitude_spectrum1, cmap = 'gray')
  #plt.title('Magnitude Spectrum1'), plt.xticks([]), plt.yticks([])
  #plt.show()

  imgg1 = magnitude_spectrum
  imgg2 = magnitude_spectrum1

  def getAverage(imgg, u, v, n):
    """img as a square matrix of numbers"""
    s = 0
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            s += imgg[u+i][v+j]
    return float(s)/(2*n+1)**2

  def getStandardDeviation(imgg, u, v, n):
    s = 0
    avgg = getAverage(imgg, u, v, n)
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            s += (imgg[u+i][v+j] - avgg)**2
    return (s**0.5)/(2*n+1)

  def zncc(imgg1, imgg2, u1, v1, u2, v2, n):
    stdDeviation1 = getStandardDeviation(imgg1, u1, v1, n)
    stdDeviation2 = getStandardDeviation(imgg2, u2, v2, n)
    avgg1 = getAverage(imgg1, u1, v1, n)
    avgg2 = getAverage(imgg2, u2, v2, n)

    s = 0
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            s += (imgg1[u1+i][v1+j] - avgg1)*(imgg2[u2+i][v2+j] - avgg2)
    return float(s)/((2*n+1)**2 * stdDeviation1 * stdDeviation2)

  if __name__ == "__main__":
   
    print(zncc(imgg1, imgg2, 1,1,1,1, 1))
    znccl.append((zncc(imgg1, imgg2, 1,1,1,1, 1)))

plt.plot(angles,znccl)
plt.show()


znccpt = int(znccl.index(max(znccl)))
print(angles[znccpt])

for angle in range(0, 360 - angles[znccpt] , 1):
  rotatedd = imutils.rotate(pixx2, angle=angle)
  rrotatedd = imutils.rotate(pixrot, angle=angle)
  rrotatedd = rrotatedd#[80:400, 90:410]
  rotatedd = rotatedd#[80:400, 90:410]
  grayy1 = cv2.GaussianBlur(rotatedd, (41, 41), 0)
  (minVall1, maxVall1, minLocc1, maxLocc1) = cv2.minMaxLoc(grayy1)
  rotatedd = rotatedd[maxLocc1[1]-150:maxLocc1[1]+150, maxLocc1[0]-150:maxLocc1[0]+150]
  rrotatedd = rrotatedd[maxLocc1[1]-150:maxLocc1[1]+150, maxLocc1[0]-150:maxLocc1[0]+150]

for angle in range(0, 360 - angles[znccpt] , 1):
  rottatedd = imutils.rotate(pix1, angle=angle)
  

pic3d = rottatedd + rrotatedd
picx3d = cv2.resize(pic3d,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
for angle in range(0, 360 - angles[znccpt] , 1):
  rotatedd3d = imutils.rotate(pic3d, angle=angle)
  picc3d = cv2.resize(rotatedd3d,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
for angle in range(0, 90, 1):
  rotatedd3d1 = imutils.rotate(pic3d, angle=angle)
  picc3d1 = cv2.resize(rotatedd3d1,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
for angle in range(0, 90, 1):
  rotatedd3d2 = imutils.rotate(rotatedd3d, angle=angle)
  picc3d2 = cv2.resize(rotatedd3d2,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
for angle in range(0, 180, 1):
  rotatedd3dx1 = imutils.rotate(pic3d, angle=angle)
  picc3dx1 = cv2.resize(rotatedd3dx1,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
for angle in range(0, 180, 1):
  rotatedd3dy2 = imutils.rotate(rotatedd3d, angle=angle)
  picc3dy2 = cv2.resize(rotatedd3dy2,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
cv2.imshow('3d1',picc3d)
cv2.imshow('3d11',picc3d1)
cv2.imshow('3d12',picc3d2)
cv2.imshow('3d1x1',picc3dx1)
cv2.imshow('3d1y2',picc3dy2)
cv2.imshow('3d2',picx3d)
cv2.imshow('im1',rottatedd)
cv2.imshow('im2',rrotatedd)
cv2.imshow('1st',piic1)
cv2.imshow('2nd',rotatedd)
cv2.imshow('sub',median)
cv2.imshow('sub1',mmedian1)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
img = cv2.imread('image.jpg')

greyimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
rbgimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )
hslimg = cv2.cvtColor(img, cv2.COLOR_BGR2HLS )
cv2.imshow(" ",greyimg)
cv2.waitKey(0)

cv2.imshow(" ",rbgimg)
cv2.waitKey(0)

cv2.imshow(" ",hslimg)
cv2.waitKey(0)

imgblur = cv2.blur(img,(5,5))
cv2.imshow(" ",imgblur)
cv2.waitKey(0)

imgblur = cv2.GaussianBlur(img,(5,5),0,0)
cv2.imshow(" ",imgblur)
cv2.waitKey(0)

imgblur = cv2.medianBlur(img, 9)
cv2.imshow(" ",imgblur)
cv2.waitKey(0)

imgblur = cv2.medianBlur(img, 75, 75)
cv2.imshow(" ",imgblur)
cv2.waitKey(0)

gx = cv2.Sobel(greyimg, cv2.CV_32F, 1,0,ksize=5)
gx - cv2.convertScaleAbs(gx)
cv2.imshow(" ",gx)
cv2.waitKey(0)

gy = cv2.Sobel(greyimg, cv2.CV_32F, 0,1,ksize=5)
gy - cv2.convertScaleAbs(gx)
cv2.imshow(" ",gy)
cv2.waitKey(0)

combin = cv2.addWeighted(gx , 0.5 , gy , 0.5, 0)
cv2.imshow(" ",combin)
cv2.waitKey(0)

import cv2
vidcap = cv2.VideoCapture('../../extra/sun1.mp4')
success,image = vidcap.read()

count = 1
success = True

subnum = 0

while success:
  success,image = vidcap.read()
  cv2.imwrite("../../extraresults/%d/sun1_%d.jpg" % subnum % count, image)
  print("saved image %d.jpg in %d folder." % count % subnum)
  
  if cv2.waitKey(10) == 27:                    
      break
  count += 1

  if count%16 == 0:
     subnum +=1
     
     
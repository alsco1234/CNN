#import pandas as pd
import csv

f = open('/home/alsco1234/s3d/S3D/kinetics_softmax_S3D.csv', 'r')
file = open('kinetics_s3d_top10_test.csv', 'w')

reader = csv.reader(f, delimiter=",")
line = []
pred5 = []
N = 10
linecount = 0
for line in reader:
    if linecount > 10:
        break
    """
    if line[0] == 'dir':
        continue
    del line[0]
    """
    writer = csv.writer(file)
    #line = f.readline()
    # line = line.split(',')

    if linecount > 0 :
        res = sorted(line, reverse=True)
        #print("readling line\n", line, end='')

        # 두번째칸부터 각 줄에서 가장 큰 값이면 1 아니면 0
        for idx, num in enumerate(line):
            if idx==0:
                continue
            float(line[idx])
            if line[idx] > res[N-1]:
                line[idx] = 1
            elif idx > 0 :
                line[idx] = 0
        
    linecount = linecount + 1

    writer.writerow(line)

print("line count is ", linecount)
f.close()
file.close()
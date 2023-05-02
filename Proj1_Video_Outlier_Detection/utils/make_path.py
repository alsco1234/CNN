# writedata.py
f = open("CCD_crash.txt", 'w')

for i in range(1, 1501):
    data = "/projects/vode/data/CCD/Crash-1500/" + f'{i:06d}' + ".mp4\n"
    f.write(data)

f.close()
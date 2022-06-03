import dateutil.parser

def read_series(filename):
    infile=open(filename,'r')
    times=[]
    values=[]
    for line in infile:
        #print(">%s<%d"%(line,len(line)))
        if line.startswith("#") or len(line)<=1:
            continue
        parts=line.split()
        times.append(dateutil.parser.parse(parts[0]))
        values.append(float(parts[1]))
    infile.close()
    return (times,values)

def shifted_data_variance(data):
    if len(data) < 2:
        return 0.0
    K = data[0]
    n = Ex = Ex2 = 0.0
    for x in data:
        n = n + 1
        Ex += x - K
        Ex2 += (x - K) * (x - K)
    variance = (Ex2 - (Ex * Ex) / n) / (n - 1)
    # use n instead of (n-1) if want to compute the exact variance of the given data
    # use (n-1) if data are samples of a larger population
    return variance

list = ['tide_hansweert.txt', 'waterlevel_hansweert.txt',
        'tide_terneuzen.txt', 'waterlevel_terneuzen.txt',
        'tide_vlissingen.txt', 'waterlevel_vlissingen.txt',
        'tide_bath.txt', 'waterlevel_bath.txt']

variance_list = []
for i in range(4):
    time1, value1 = read_series(list[2*i])
    time2, value2 = read_series(list[2*i + 1])
    for j in range(len(value1)):
        value1[j] = value1[j] - value2[j]
    mean = 0
    for j in range(len(value1)):
        mean += value1[j]
    mean = mean / len(value1)
    for j in range(len(value1)):
        value1[j] = value1[j] - mean
    variance_list.append(shifted_data_variance(value1))
averaged_variance = sum(variance_list) / len(variance_list)
print("Average variance: %f" % averaged_variance)

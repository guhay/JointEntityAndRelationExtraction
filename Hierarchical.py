import numpy as np
import datetime
def HaverSin(theta):
    v=np.sin(theta/2)
    return np.multiply(v,v)
def convertDegreesToRadians(degrees):
    return np.pi*degrees/180
def Distance(lat1,lon1,lat2,lon2):
    EARTH_RADIUS=6371.0
    batch_size=lat1.shape[1]
    seq_size=lat2.shape[1]
    orientation = np.matrix(np.ones(seq_size))
    portrait=np.matrix(np.ones(batch_size).reshape((batch_size,1)))
    lat1=convertDegreesToRadians(lat1).reshape((batch_size,1))#[[x1],[x2]]
    lat1=lat1*orientation
    lon1=convertDegreesToRadians(lon1).reshape((batch_size,1))#[[y1],[y2]]
    lon1=lon1*orientation
    lat2=convertDegreesToRadians(lat2)#[n1,n2,n3,n4,n5]
    lat2=portrait*lat2
    lon2=convertDegreesToRadians(lon2)#[m1,m2,m3,m4,m5]
    lon2=portrait*lon2

    vLon=np.abs(lon1-lon2)
    vLat=np.abs(lat1-lat2)
    h=HaverSin(vLat)+np.multiply(np.multiply(np.cos(lat1),np.cos(lat2)),HaverSin(vLon))
    distance=2*EARTH_RADIUS*np.arcsin(np.sqrt(h))
    return distance
    #
    # distance=2*EARTH_RADIUS*np.arcsin(np.sqrt(h))
    # return distance
def GetList(path):
    d={}
    with open(path,encoding='utf-8') as f:
        for line in f.readlines():
            lineArr=line.strip().split(',')
            temp=(float(lineArr[1]),float(lineArr[2]))
            d[int(lineArr[0])]=temp
    return d

def init_maxtrix_multi(l,batch_size):
    d1={}
    n=0
    num=len(l)//batch_size
    for i in range(num):
        startTime=datetime.datetime.now()
        startIndex=i*batch_size
        endIndex=(i+1)*batch_size
        temp=l[startIndex:endIndex]
        a=np.matrix([n[0] for n in temp])
        b=np.matrix([n[1] for n in temp])
        c=np.matrix([n[0] for n in l])
        d=np.matrix([n[1] for n in l])
        distance=Distance(a,b,c,d)
        l=get_min_loc(distance)
        print(l)
        endTime=datetime.datetime.now()
        print(i+1," ",(endTime-startTime).seconds)
def get_min_loc(mat):
    l=[]
    arr=np.array(mat)
    for i in range(len(arr)):
        min_num=np.min(arr[i])
        loc_col=np.where(arr[i]==min_num)
        l.append({'loc':loc_col[0][0],"num":min_num})
    return l
def load_mat(path):
    d={}
    with open(path,encoding='utf-8') as f:
        for line in f.readlines():
            lineArr=line.strip().split('\t')
            num=int(lineArr[0])
            minnum=int(lineArr[1])
            mindis=float(lineArr[2])
            d[num]={'loc':minnum,'dis':mindis}
    return d
# d1=GetList(r'D:\Python3.5\trail\res.txt')
# l=[temp for temp in d1.values()]
# init_maxtrix_multi(l,200)
d=load_mat(r'D:\Python3.5\trail\.txt')
print(d)

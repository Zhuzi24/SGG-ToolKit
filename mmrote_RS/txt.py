import sys
sys.getdefaultencoding()
import pickle
import numpy as np
np.set_printoptions(threshold=1000000000000000)
path = '/media/dell/DATA/WLL/RSSGG/mmrotate/out/oriented_rcnn/outxin.pkl'
file = open(path,'rb')
inf = pickle.load(file,encoding='iso-8859-1')       #duqu pkl
print(len(inf))
#fr.close()
inf=str(inf)
obj_path = '/media/dell/DATA/WLL/RSSGG/mmrotate/out/oriented_rcnn/outxin.txt'
ft = open(obj_path, 'w')
ft.write(inf)

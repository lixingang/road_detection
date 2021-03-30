import numpy as np
import cv2
import os,sys
from tqdm import tqdm
import argparse
from PIL import Image
import matplotlib.pyplot as plt
os.chdir(sys.path[0])
plt.rcParams['figure.dpi'] = 300
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', action='store', dest="mode",
                        help="d:direction;e:edge;r:region")
parser.add_argument('-s', '--source',action='store', dest="source",
                    help="input image dictionary")
parser.add_argument('-t', '--target',action='store', dest="target",
                    help="save dictionary")

args = parser.parse_args()

# * * * * * * * * * *
# 方向注意力标签生成
# * * * * * * * * * *
def direction_process(imgpath, savedir, dir_num=8, ):
    
    img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE )
    img = np.where(img>0,1,0)
    shp = img.shape
    
    img_pad = np.zeros([shp[0]+2, shp[0]+2])
    img_pad[1:-1,1:-1] = img
    dir_array = np.zeros([shp[0],shp[1]])
    
    for i in range(shp[0]):
        for j in range(shp[1]):
            if img[i,j]==0:
                continue
            
            dir_array[i,j] =  img_pad[i,j]   + img_pad[i+1,j]    + img_pad[i+2,j]+ \
                              img_pad[i,j+1] +                     img_pad[i+2,j+1]+ \
                              img_pad[i,j+2] + img_pad[i+1,j+2]+ img_pad[i+2,j+2]
                    
    final = dir_array
    savepath = os.path.join(savedir, imgpath.split('/')[-1])
    cv2.imwrite(savepath, final)
def batch_process(imgdir, savedir):
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    for i in tqdm(os.listdir(imgdir)):
        if i.split('.')[-1] != 'png':
            print('continue..')
            continue
        direction_process(os.path.join(imgdir, i), savedir, 8)
        
# direction_process('vegas/labels_pixel_png/RGB-PanSharpen_AOI_2_Vegas_img1003_mask.png', 'vegas/labels_direction')
# batch_process('/data/lxg/road/roadtracer/labels_pixel', '/data/lxg/road/roadtracer/labels_direction')

# * * * * * * * * * *
# 边缘标签生成
# * * * * * * * * * *

def edge_extract(input, output):
    #img = cv2.imread("labels_pixel/RGB-PanSharpen_AOI_2_Vegas_img10_mask.png", 0)
    img = cv2.imread(input, 0)
    imgGau = cv2.GaussianBlur(img,(3,3),0)
    gray_lap = cv2.Laplacian(imgGau, cv2.CV_16S, ksize=3)
    dst = cv2.convertScaleAbs(gray_lap)
    dst = cv2.resize(dst,(512,512))
    dst[dst>0]=200
    #bond = np.hstack((img,dst))
#     plt.rcParams['figure.dpi'] = 200
#     plt.imshow(dst,cmap=plt.cm.gray)

    cv2.imwrite(output,dst)
    
def batch_edge_extract(in_dir,out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        
    i = 1 
    for img in tqdm(os.listdir(in_dir)):
        i+=1
        crt_img = os.path.join(in_dir, img)
        out_img = os.path.join(out_dir, img)
        if os.path.exists(out_img):
            print(out_img,'exists!\n ')
            continue
        edge_extract(crt_img, out_img)
    print(i)
def draw(imgpath):
    img = cv2.imread(imgpath) 
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
    gray = cv2.resize(gray,(512,512))
    ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)  
    im2,contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
    cv2.drawContours(im2,contours,-1,(200,200,200),1)
    cv2.imwrite(imgpath,im2)
def batch_draw(in_dir):
    for img in tqdm(os.listdir(in_dir)):
        if img.split('.')[-1]!='png':
            print('passed')
        try:
            crt_img = os.path.join(in_dir, img)
            draw(crt_img)
        except:
            print(crt_img)
# batch_edge_extract('/data/lxg/road/roadtracer/labels_pixel','/data/lxg/road/roadtracer/labels_edge')
# batch_draw('khartoum/labels_edge')

# * * * * * * * * * *
# 面域标签生成
# * * * * * * * * * *

class Grid_labels(object):
    def __init__(self, path,o_dir, grid_num=32, mini=True):
        self.grid_num = grid_num
        self.true_value = 200
        self.o_dir = o_dir
        self.path = path
        self.mini = mini
        if not os.path.exists(o_dir):
            os.mkdir(o_dir)
    def batch_seg(self):
        if self.mini==False:
            label_list = self._read_label_list(self.path)
            for i in tqdm(label_list):
                if not i.split('.')[-1]=='png':
                    continue
                if os.path.exists(os.path.join(self.o_dir, i)):
                    continue
                self._seg(i)
        else:
            label_list = self._read_label_list(self.path)
            print(len(label_list))
            for i in tqdm(label_list):
                if not i.split('.')[-1]=='png':
                    continue
                if os.path.exists(os.path.join(self.o_dir, i)):
                    continue
                self._seg_mini(i)
        
    def _read_label_list(self, path):
        # return the list of label-image-path
        ll = [os.path.join(path,i) for i in os.listdir(path)]
        return ll
    def _read_and_resize(self, i, scale=(512,512)):
        im = Image.open(i)
        im = np.array(im.resize(scale))
        width, height = im.shape
        return im,width,height
    def _seg(self, path):
        o_name = path.split('/')[-1]
        img, w, h = self._read_and_resize(path)
        assert w%self.grid_num==0 and h%self.grid_num==0  
        w_scale = w//self.grid_num
        h_scale = h//self.grid_num
        label_copy = np.zeros(shape=(w,h))
        for i in range(0,self.grid_num):
            for j in range(0,self.grid_num):
                current_block = img[i*w_scale:(i+1)*w_scale, j*w_scale:(j+1)*w_scale]
                if np.sum(current_block>10):
                    label_copy[i*h_scale:(i+1)*h_scale, j*h_scale:(j+1)*h_scale] = self.true_value
                output = Image.fromarray(label_copy).convert('RGB')
                output.save(os.path.join(self.o_dir, o_name))
    def _seg_mini(self, path):
        o_name = path.split('/')[-1]
        img, w, h = self._read_and_resize(path)
        assert w%self.grid_num==0 and h%self.grid_num==0  
        w_scale = w//self.grid_num
        h_scale = h//self.grid_num
        label_copy = np.zeros(shape=(self.grid_num, self.grid_num))
        for i in range(0,self.grid_num):
            for j in range(0,self.grid_num):
                current_block = img[i*w_scale:(i+1)*w_scale, j*w_scale:(j+1)*w_scale]
                if np.sum(current_block>10):
                    label_copy[i,j] = self.true_value
                output = Image.fromarray(label_copy).convert('RGB')
                output.save(os.path.join(self.o_dir, o_name))
                
# print('[I] Seg Start')
# g = Grid_labels('labels_pixel','labels_seg_mini', mini=True)
# g.batch_seg()
# print('[I] Seg Done')

# * * * * * * * * * *
# 生成list列表文件
# * * * * * * * * * *
def save_image_list(imagepath, output_file):
    '''
    该工具用来将文件夹里的文件列表输出文件txt
    '''
    f = open(output_file,'w')
    for i in os.listdir(imagepath):
        name = i.split('.')[0]
        f.write(name+'\n')
    f.close
def save_image_pair_list(imagepath, labelpath, output_file):
    '''
    该工具用来将文件夹里的文件列表输出 （图片文件名 标签文件名）txt文件
    '''
    f = open(output_file,'w')
    for i in os.listdir(imagepath):
        s = ['.ipynb']
        if s[0] in i:
            continue
        imgNo = find_name(i)
        name = 'RGB-PanSharpen_AOI_5_Khartoum_'+imgNo+'_mask.png'
        # valid
        path = os.path.join(labelpath, name)
        if not os.path.exists(path):
            print(path,'is not exists')
            continue
        f.write(i+' '+name+'\n')
    f.close


# * * * * * * * * * *
# 程序入口
# * * * * * * * * * *

if args.mode=="d":
#     batch_process('/data/lxg/road/roadtracer/labels_pixel', '/data/lxg/road/roadtracer/labels_direction')
    batch_process(args.source, args.target)
    print('[I] Direction label Done')
elif args.mode=="e":
#     batch_edge_extract('/data/lxg/road/roadtracer/labels_pixel','/data/lxg/road/roadtracer/labels_edge')
    batch_edge_extract(args.source, args.target)
    print('[I] Edge label Done')
elif args.mode=="r":
#     g = Grid_labels('labels_pixel','labels_seg_mini', mini=True)
    g = Grid_labels(args.source, args.target, mini=True)
    g.batch_seg()
    print('[I] Seg label Done')
  
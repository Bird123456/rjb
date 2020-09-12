import os

path =r'F:\Python\rjb_yfzd\img\out_img1'
original_images =[]
#walk direction
'''
root:path下的文件夹路径
dirs：root下的文件夹名称
filenames：root下的文件名称
'''
for root, dirs, filenames in os.walk(path):
    for filename in filenames:
        original_images.append(os.path.join(root, filename)) #将所有文件添加到一个list

print('num:',len(original_images)) #打印list长度  就是保存的数量
f = open('count_total.txt','w+')

current_dirname =os.path.dirname(original_images[0])#去掉文件名返回目录

file_num =0#初始化数量
for filename in original_images:
        dirname = os.path.dirname(filename)#去掉文件名返回目录
        #一个目录数量读取完后，需要重新初始化数量，读取下一个目录
        #当读取最后一个文件时，下面也没有跟新的目录了，所以需要将数据写入

        if dirname != current_dirname or filename == original_images[-1]:
            if dirname == current_dirname and filename == original_images[-1]:#最后一个文件时，对数量+1
                file_num += 1

            if dirname != current_dirname and filename == original_images[-1]:
                total_name = current_dirname[32:]
                f.write('%s:\t%d\n'%(total_name,file_num))#将数据写入，（接下来就是跟新操作）
                file_num = 1#初始化数量统计
                current_dirname = dirname#跟新新的目录

            total_name = current_dirname[32:]
            f.write('%s:\t%d\n'%(total_name,file_num))#将数据写入，（接下来就是跟新操作）
            current_dirname = dirname#跟新新的目录
            file_num = 1#初始化数量统计
        #同一个目录就直接数量+1
        else:
            file_num +=1
    #返回保存文档的首位
f.seek(0)
#获取文档信息
for s in f:
    print(s)
f.close()

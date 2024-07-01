import numpy as np
import os
import argparse
import cv2
import tqdm
import glob
import esim_torch
import torch
import h5py
import time
import sys
import hdf5plugin

import threading #多线程处理
import pdb #打断点用的

H = 480
W = 640
NBINS = 5

def render(x, y, pol, H, W):
    # 断言检查，确保 x, y, pol 的大小相等，并且 H 和 W 大于 0
    assert x.size == y.size == pol.size
    assert H > 0
    assert W > 0

    # 初始化一个高为 H，宽为 W 的 3 通道图像，每个像素初始化为白色 (255, 255, 255)
    img = np.full((H,W,3), fill_value=255,dtype='uint8')

    # 初始化一个高为 H，宽为 W 的掩码，初始值为 0
    mask = np.zeros((H,W),dtype='int32')

    # 将极性数组转换为整数类型
    pol = pol.astype('int')

    # 将极性为 0 的值转换为 -1（-1的话，仍然是-1）
    pol[pol==0]=-1

    # 创建一个布尔掩码，用于筛选出有效的 x, y 坐标
    mask1 = (x>=0)&(y>=0)&(W>x)&(H>y)

    # 将有效坐标的极性值映射到掩码中
    mask[y[mask1],x[mask1]]=pol[mask1]
    img[mask==0]=[255,255,255] #将掩码中值为 0 的像素设置为白色
    img[mask==-1]=[255,0,0] #将掩码中值为 -1 的像素设置为蓝色
    img[mask==1]=[0,0,255] #将掩码中值为 1 的像素设置为红色（bgr）
    return img

def to_voxel_grid_numpy(events, num_bins, width=W, height=H):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    """

    assert(events.shape[1] == 4)
    assert (num_bins > 0) and (num_bins < 1000)
    assert(width > 0) and (width < 2048)
    assert(height > 0) and (height < 2048)
    events = events.astype(np.float32)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    ts = events[:, 0]
    xs = events[:, 1]
    ys = events[:, 2]
    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(np.int64)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, (xs[valid_indices] + ys[valid_indices] * width +
              tis[valid_indices] * width * height).astype(np.int64), vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, (xs[valid_indices] + ys[valid_indices] * width +
              (tis[valid_indices] + 1) * width * height).astype(np.int64), vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    return voxel_grid

def ignore_files(directory, files):
    return [f for f in files if os.path.isfile(os.path.join(directory, f))]

def save_evs_to_h5(xs, ys, ts, ps, evs_file_path, Cneg=0.2, Cpos=0.2, refractory_period_ns=0, compr="gzip", compr_lvl=4):
    # 定义压缩配置
    compression = {
        "compression": compr,
        "compression_opts": compr_lvl,
    }
    # 打开一个新的 HDF5 文件，用于写入数据
    with h5py.File(evs_file_path, "w") as f:
        # 创建数据集并使用 Blosc 压缩（使用 zstd 压缩算法，压缩级别为 compr_lvl，启用字节重排）
        f.create_dataset("x", data=xs, **hdf5plugin.Blosc(cname='zstd', clevel=compr_lvl, shuffle=hdf5plugin.Blosc.SHUFFLE)) 
        f.create_dataset("y", data=ys, **hdf5plugin.Blosc(cname='zstd', clevel=compr_lvl, shuffle=hdf5plugin.Blosc.SHUFFLE)) 
        f.create_dataset("t", data=ts, **hdf5plugin.Blosc(cname='zstd', clevel=compr_lvl, shuffle=hdf5plugin.Blosc.SHUFFLE)) 
        f.create_dataset("p", data=ps, **hdf5plugin.Blosc(cname='zstd', clevel=compr_lvl, shuffle=hdf5plugin.Blosc.SHUFFLE))  
        # TODO: ms_to_idx

    # 输出保存信息
    # print(f"Saved {len(xs)} events to {evs_file_path}.")
    return

def to_voxel_grid(xs, ys, ts, ps, nb_of_time_bins=5, remapping_maps=None):
    """Returns voxel grid representation of event steam.

    In voxel grid representation, temporal dimension is
    discretized into "nb_of_time_bins" bins. The events fir
    polarities are interpolated between two near-by bins
    using bilinear interpolation and summed up.

    体素网格表示法中，时间维度被离散化为固定数量的时间片（nb_of_time_bins），
    并且事件的极性（polarity）通过双线性插值在相邻的两个时间片之间进行插值和累加。
    如果事件流为空，则体素网格也为空。

    If event stream is empty, voxel grid will be empty.
    """
    # 初始化一个零张量，形状为(nb_of_time_bins, H, W)，表示体素网格（nb_of_time_bins为时间的切片）
    voxel_grid = torch.zeros(nb_of_time_bins,
                          H,
                          W,
                          dtype=torch.float32,
                          device='cpu')
    # 将体素网格展平成一维张量
    voxel_grid_flat = voxel_grid.flatten()

    # 将极性ps的值转换为int8类型，并将0的极性值设为-1
    ps = ps.astype(np.int8)
    ps[ps == 0] = -1

    # Convert timestamps to [0, nb_of_time_bins] range.（将时间戳转换为[0, nb_of_time_bins]范围）
    duration = ts[-1] - ts[0] #获取时间戳的持续时间
    start_timestamp = ts[0]#获取时间戳的开始时间
    features = torch.from_numpy(np.stack([xs.astype(np.float32), ys.astype(np.float32), ts, ps], axis=1))
    x = features[:, 0]
    y = features[:, 1]
    polarity = features[:, 3].float()
    t = (features[:, 2] - start_timestamp) * (nb_of_time_bins - 1) / duration  # torch.float64
    t = t.to(torch.float64)

     # 如果提供了remapping_maps，则进行坐标重映射（类似于去除失真）
    if remapping_maps is not None:
        remapping_maps = torch.from_numpy(remapping_maps)
        x, y = remapping_maps[:,y,x]

    # 计算左边和右边的时间、x、y坐标
    left_t, right_t = t.floor(), t.floor()+1
    left_x, right_x = x.floor(), x.floor()+1
    left_y, right_y = y.floor(), y.floor()+1

    for lim_x in [left_x, right_x]:
        for lim_y in [left_y, right_y]:
            for lim_t in [left_t, right_t]:
                # 创建掩码，确保坐标在合法范围内
                mask = (0 <= lim_x) & (0 <= lim_y) & (0 <= lim_t) & (lim_x <= W-1) \
                       & (lim_y <= H-1) & (lim_t <= nb_of_time_bins-1)

                # 计算线性索引
                # we cast to long here otherwise the mask is not computed correctly
                lin_idx = lim_x.long() \
                          + lim_y.long() * W \
                          + lim_t.long() * W * H

                 # 计算插值权重
                weight = polarity * (1-(lim_x-x).abs()) * (1-(lim_y-y).abs()) * (1-(lim_t-t).abs())

                # 将加权的极性值累加到体素网格中
                voxel_grid_flat.index_add_(dim=0, index=lin_idx[mask], source=weight[mask].float())

    return voxel_grid

def save_voxels_to_h5(voxel, evs_file):
    voxel_float16 = voxel.to(torch.float16)
    with h5py.File(evs_file, "w") as f:
        f.create_dataset("voxel", data=voxel_float16, **hdf5plugin.Blosc(cname='zstd', clevel=4, shuffle=hdf5plugin.Blosc.SHUFFLE))

def convert_sequence(root, device, stereo="left"):

    assert stereo == "left" #确认选用的为左目序列
    imgdir = os.path.join(root, f"image_{stereo}")#获取原始图像的位置路径
    # print("imgdir:",imgdir);

    if not os.path.exists(imgdir):#如果原始图像的位置路径不存在
        # 如果目录中存在fps.txt文件，则说明已经转换过了，直接跳过
        if os.path.exists(os.path.join(root, "imgs")): 
            print("\033[31m no {imgdir} but have {root}/imgs/, it should be okay \033[0m");
        else:
            print("\033[31m no {imgdir} and {root}/imgs/, please check !!!!!!!!!!!!!!!!! \033[0m");
    else:#如果解压后的原始图像的位置路径存在，则将它改为imgs文件夹
        cmd = f"mv {root}/image_{stereo}/ {root}/imgs/" #移动 image_left 目录到 imgs 目录。
        os.system(f"{cmd}")#执行命令

    evs_dir = imgdir.replace(f"image_{stereo}", f"event_{stereo}")
    img_dir = os.path.join(root, "imgs")
    if os.path.exists(evs_dir):#r如果此处已经存在event文件夹，则直接跳过
        print(f"Already converted {root} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return    
    print(f"\033[0;31;42m No upsampling!!!!!! \033[0m")
    print(f"Converting {img_dir} to {evs_dir}")  
    print(f"Using GPU {device} ")  

    # pdb.set_trace()

    # 开始创建事件，确定阈值
    # create events
    C = 0.25
    dC = 0.09
    Cneg = np.random.uniform(C-dC, C+dC)
    Cpos = np.random.uniform(C-dC, C+dC)

    os.makedirs(evs_dir, exist_ok=True) #创建evs_dir文件夹
    cmd = f"touch {evs_dir}/Theshold_of_event_generation.txt"
    os.system(f"{cmd}")
    cmd = f"echo contrast_threshold_neg:{Cneg}; contrast_threshold_pos:{Cpos} > {evs_dir}/Theshold_of_event_generation.txt"
    os.system(f"{cmd}")
    refractory_period_ns = 0  # TODO: sample refractory?

    # 在指定的GPU上进行初始化
    with torch.cuda.device(device):
        # 下面进行ESIM初始化，生成事件
        simulator = esim_torch.ESIM(
            contrast_threshold_neg=Cneg,
            contrast_threshold_pos=Cpos,
            refractory_period_ns=refractory_period_ns,  
        )

        # 获取原始的图像文件
        image_files = sorted(glob.glob(os.path.join(root, "imgs/*.png")))
        # 获取原始图像的数量
        N_images = len(glob.glob(os.path.join(img_dir, "*.png")))
        assert N_images == len(image_files) 

        fps_imgs_s=25;#假定原始图像的fps为25
        # 推算出原始的每张图像的时间戳
        tss_imgs_ns = (np.arange(start=0, stop=N_images)*1e9/fps_imgs_s).astype(np.int64) # [0, N_images-1]

        #原始图像的时间戳放到GPU上
        tss_ns=torch.from_numpy(tss_imgs_ns).cuda()

        # 检查时间戳是否正确
        assert np.abs(tss_ns[0].item() - tss_imgs_ns[0]) < 10000000  # 10ms
        assert np.abs(tss_ns[-1].item() - tss_imgs_ns[-1]) < 10000000 # 10ms
        print(f"begin time: {tss_ns[0].item() }, end time: {tss_ns[-1].item() }")

        #将时间保存下来 save img_tss and img_up_tss
        # cmd = f"cp {upimgs}/timestamps.txt {evs_dir}/tss_upimgs_sec.txt"
        # os.system(f"{cmd}")
        f = open(os.path.join(evs_dir, "tss_imgs_sec.txt"), "w") #原始图像的时间戳
        for ts_img_ns in tss_imgs_ns:
            f.write(f"{ts_img_ns/1e9}\n")
        f.close()

        # 通过bar显示进度
        pbar = tqdm.tqdm(total=len(image_files)-1)
        num_events = 0

        img_right_counter = 1 # start at 1 because the first image is not useds
        xs, ys, ts, ps = [], [], [], [] #这是把事件以stream/voxel的形式保存下来，注意h5文件中，应该以raw event的形式存数据
        # 创建文件夹存放raw event(h5)以及可视化图片(viz)
        os.makedirs(os.path.join(evs_dir, "h5"), exist_ok=True)
        os.makedirs(os.path.join(evs_dir, "viz"), exist_ok=True)

        # 定义视频保存路径和视频编码器
        # event_video_path = os.path.join(root, "events_video.avi")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # 或者使用其他编码器
        # event_video_writer = cv2.VideoWriter(event_video_path, fourcc, fps_imgs_s, (W, H))  # fps_imgs_s是帧率
        # image_video_path = os.path.join(root, "image_video.avi")
        # image_video_writer = cv2.VideoWriter(image_video_path, fourcc, fps_imgs_s, (W, H),isColor=0)#图像用了灰度图 

        IE_video_path = os.path.join(root, "combined_image_event_video.avi")
        combined_video_writer = cv2.VideoWriter(IE_video_path, fourcc, fps_imgs_s, (W * 2, H))  # 合并后的视频宽度是两个图像宽度之和

        # 逐个处理每一张图像
        for image_file, ts_ns in zip(image_files, tss_ns):
            image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)#读取图像为灰度图
            # image_video_writer.write(image)#将图像写入视频
            log_image = np.log(image.astype("float32") / 255 + 1e-5)#对图像进行log处理
            log_image = torch.from_numpy(log_image).cuda()#将图像放到GPU上

            sub_events = simulator.forward(log_image, ts_ns)#生成事件

            # for the first image, no events are generated, so this needs to be skipped
            if sub_events is None: # TODO: handle case when no events are generated if img_ct_right > 1
                continue

            sub_events = {k: v.cpu() for k, v in sub_events.items()}    
            num_events += len(sub_events['t'])#计算生成的事件数量，用于更新总的事件量
    
            # do something with the events
            # np.savez(os.path.join(outdir, "%010d.npz" % counter), **sub_events)
            x = sub_events["x"].numpy().astype(np.int16) if sub_events else np.empty(0, dtype=np.int16)
            y = sub_events["y"].numpy().astype(np.int16) if sub_events else np.empty(0, dtype=np.int16)
            t = sub_events["t"].numpy().astype(np.int64) if sub_events else np.empty(0, dtype=np.int64)
            p = sub_events["p"].numpy().astype(np.int8) if sub_events else np.empty(0, dtype=np.int8)
            
            # 按image的时间，累积成voxel
            if t.max() <= tss_imgs_ns[img_right_counter] or img_right_counter == N_images - 1:
                xs.append(x)
                ys.append(y)
                ts.append(t)
                ps.append(p)
                if img_right_counter == N_images - 1:
                    assert t.max() - tss_imgs_ns[img_right_counter] < 100000 # 100us = 0.1ms
                    print(f"diff: {t.max() - tss_imgs_ns[img_right_counter]}ns")
            else:
                idx = np.where(t > tss_imgs_ns[img_right_counter])[0][0]
                xs.append(x[:idx])
                ys.append(y[:idx])
                ts.append(t[:idx])
                ps.append(p[:idx])

                # 将事件累计成voxel
                xs = np.concatenate(np.array(xs, dtype=object)).astype(np.uint16)
                ys = np.concatenate(np.array(ys, dtype=object)).astype(np.uint16)
                ts = np.concatenate(np.array(ts, dtype=object)).astype(np.int64)
                ps = np.concatenate(np.array(ps, dtype=object)).astype(np.int8)

                # 获取event的h5文件的路径
                evs_file = os.path.join(evs_dir, "h5", "%010d.h5" % img_right_counter)
                # voxel = to_voxel_grid(xs, ys, ts, ps, nb_of_time_bins=NBINS) 
                # save_voxels_to_h5(voxel, evs_file)#将事件（以voxel的格式）保存到h5文件中

                # 以h5文件的格式保存事件
                save_evs_to_h5(xs, ys, ts, ps, evs_file_path=evs_file, Cneg=Cneg, Cpos=Cpos, refractory_period_ns=refractory_period_ns)
                
                img = render(xs, ys, ps, H=H, W=W)#将事件渲染成图片（此处的xs是累计成voxel的事件）
                cv2.imwrite(os.path.join(evs_dir, "viz", "%010d.png" % img_right_counter), img)

                # 将图像写入视频
                # event_video_writer.write(img)
                # 创建一个新的图像，将image放在左侧，img放在右侧
                combined_image = np.zeros((H, W * 2,3), dtype=np.uint8)
                combined_image[:, :W] = cv2.merge([image, image, image])  # 左侧放原始图像(将灰度图复制到三个通道)
                combined_image[:, W:] = img  # 右侧放事件图像
                combined_video_writer.write(combined_image)  # 将合并后的图像写入视频

                img_right_counter += 1
                xs = [x[idx:]]
                ys = [y[idx:]]
                ts = [t[idx:]]
                ps = [p[idx:]]

            pbar.set_description(f"Num events generated: {num_events}")
            pbar.update(1)

        time.sleep(5)
        # 结束事件的生成
        # pdb.set_trace()
        # 打印一共用于生成事件的图像量以及总的图像数目
        # print(f"img_right_counter = {img_right_counter}, N_images = {N_images}")

        # 把事件保存到h5文件中，并且保存可视化的图片（此处为何再做一次？应该是把最后的没存的再存一次）
        if len(xs) > 0:
            if len(xs[0]) > 1:
                xs = np.concatenate(np.array(xs, dtype=object)).astype(np.uint16)
                ys = np.concatenate(np.array(ys, dtype=object)).astype(np.uint16)
                ts = np.concatenate(np.array(ts, dtype=object)).astype(np.int64)
                ps = np.concatenate(np.array(ps, dtype=object)).astype(np.int8)

                print(f"Saving last {len(xs)} events from {ts[0]}nsec to {ts[-1]}nsec.")
                print(f"tss_imgs_ns[-1]: {tss_imgs_ns[-1]}, tss_imgs_ns[-2]: {tss_imgs_ns[-2]}")
                # TODO: assert these are the correct events

                evs_file = os.path.join(evs_dir, "h5", "%010d.h5" % img_right_counter)
                # voxel = to_voxel_grid(xs, ys, ts, ps, nb_of_time_bins=NBINS)
                # save_voxels_to_h5(voxel, evs_file)

                # 以h5文件的格式保存事件
                save_evs_to_h5(xs, ys, ts, ps, evs_file_path=evs_file, Cneg=Cneg, Cpos=Cpos, refractory_period_ns=refractory_period_ns)
                img = render(xs, ys, ps, H=H, W=W)#将事件渲染成图片（此处的xs是累计成voxel的事件）
                cv2.imwrite(os.path.join(evs_dir, "viz", "%010d.png" % img_right_counter), img)
                img_right_counter += 1

        if img_right_counter == N_images:
            print(f"Sucees: img_right_counter = {img_right_counter}, N_images - 1 = {N_images}. ")
        else:
            print(f"\n\n\n\nError**********! ")
            print(f"img_right_counter = {img_right_counter}, N_images = {N_images}")
            print(f"Error**********!\n\n\n\n")

        simulator.reset()#注意要重置，不然下一次处理可能会受到影响
        time.sleep(5)
        # pdb.set_trace()

        pass
    print(f"Finished processing {root} on {device}")

    # # create a file to indicate that the conversion is done（ #创建及修改文件的时间属性 ）
    # cmd = f"touch {root}/converted.txt"
    # os.system(f"{cmd}")
    return


def process_individual_gpu(root, device_id):
    device = f'cuda:{device_id}' #指定使用的GPU
    print(f"\033[0;31;42m Processing {root} on {device} \033[0m")
    convert_sequence(root, device) #执行序列的转换处理

def main():
    parser = argparse.ArgumentParser(description="Raw to png images in dir")
    parser.add_argument("--dirsfile", help="Input raw dir.", default="/DIRECTORY/test.txt")#输入的文件，根据文件内的路径进行处理
    # 定义一个bool变量，名为--whetherunzip，如果输入了这个参数，则该变量为True，否则为False。
    parser.add_argument("--whetherunzip", action='store_true', help="Whether to unzip the zip files in the directory.")

    args = parser.parse_args()
    assert ".txt" in args.dirsfile #输入的需要为txt文件
    print(f"config file = {args.dirsfile}")
     
    file = open(f"{args.dirsfile}", "r") #打开文件并读取里面的内容
    # file.read()：读取文件的全部内容，返回一个包含文件内容的字符串。
    # .splitlines()：将读取的字符串按照换行符（\n）进行分割，返回一个包含每一行内容的列表。
    # ROOTS = ...：将分割后的列表赋值给变量 ROOTS。
    ROOTS = file.read().splitlines() #获取数据目录的位置
    file.close()
    print(f"convert_tartan.py: Processing {len(ROOTS)} dirs: {ROOTS}")

    if args.whetherunzip: #如果输入了--whetherunzip参数,就进行解压
        root_directory = os.path.dirname(os.path.abspath(args.dirsfile))
        # 将dirsfile所在目录下所有的zip文件解压到当前目录
        for root, dirs, files in os.walk(root_directory):#目录路径，子目录，文件列表
            zip_files = [f for f in files if ".zip" in f and "image" in f]
            for zip_name in zip_files:
                # 对所有名字中包含image的zip文件进行解压
                zip_file = os.path.join(root_directory, zip_name)
                cmd = f"unzip {zip_file} -d {root_directory}/TartanAir"
                print("\nrun cmd:",cmd)
                # os.system(f"{cmd}")

    # 下面是输出一系列系统的信息
    print('sys version', sys.version)
    print('torch version', torch.__version__)
    print('cuda', torch.cuda.is_available())
    print('cudnn', torch.backends.cudnn.enabled)
    device = torch.device('cuda')
    print('device properties:', torch.cuda.get_device_properties(device))
    print('while device is using: ', torch.tensor([1.0, 2.0]).cuda())

    # 输出nvcc的版本
    cmd = "nvcc --version"
    os.system(f"{cmd}")
    cmd = "module load cuda/11.2"
    os.system(f"{cmd}")
    cmd = "module load cudnn"
    os.system(f"{cmd}")

    # for i in range(len(ROOTS)):#循环处理每一个序列
    #     print(f"\n\nconvert_tartan.py: Start processing {ROOTS[i]}")
    #     convert_sequence(ROOTS[i], stereo="left") #执行序列的转换处理
    #     print(f"convert_tartan.py: Finished processing {ROOTS[i]}\n\n")
    
    # pdb.set_trace()
    # 改为多线程处理，每个线程用一个GPU
    threads = []
    # for i in range(len(ROOTS)):
    #     # 每个线程分配一个GPU
    #     t = threading.Thread(target=process_individual_gpu, args=(ROOTS[i], i % 4))
    #     threads.append(t)
    #     t.start()

    # for t in threads:
    #     t.join()

    num_gpus = 4
    for i in range(0, len(ROOTS), num_gpus):
        batch = ROOTS[i:i + num_gpus]  # 每次取4个序列
        for j, root in enumerate(batch):
            t = threading.Thread(target=process_individual_gpu, args=(root, j))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()
        threads = []  # 清空线程列表，开始新一轮处理(但这样每次只是以num_gpus为组处理) 



if __name__ == "__main__":
    main()
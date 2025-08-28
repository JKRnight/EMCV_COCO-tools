# Date:2019.04.10
# Author: DuncanChen
# A tool implementation on gdal and geotool API
# functions:
# 1. get mask raster with shapefile
# 2. clip raster and shapefile with grid

from PIL import Image, ImageDraw
import os
from osgeo import gdal, gdalnumeric
import numpy as np
from osgeo import ogr
import glob

from skimage.segmentation import mark_boundaries

gdal.UseExceptions()


class GeoTiff(object):
    def __init__(self, tif_path):
        """
        A tool for Remote Sensing Image
        Args:
            tif_path: tif path
        Examples::
            >>> tif = GeoTif('xx.tif')
            # if you want to clip tif with grid reserved geo reference
            >>> tif.clip_tif_with_grid(512, 'out_dir')
            # if you want to clip tif with shape file
            >>> tif.clip_tif_with_shapefile('shapefile.shp', 'save_path.tif')
            # if you want to mask tif with shape file
            >>> tif.mask_tif_with_shapefile('shapefile.shp', 'save_path.tif')
        """
        self.dataset = gdal.Open(tif_path)# 读取tif图像
        self.bands_count = self.dataset.RasterCount# 读取波段数
        # get each band
        self.bands = [self.dataset.GetRasterBand(i + 1) for i in range(self.bands_count)]
        self.col = self.dataset.RasterXSize# 获取图像列数
        self.row = self.dataset.RasterYSize# 获取图像行数
        self.geotransform = self.dataset.GetGeoTransform()#获取影像的地理变换信息，存储在 self.geotransform 中
        self.src_path = tif_path#将传入的文件路径 tif_path 存储在 self.src_path 中，便于后续使用或引用
        self.mask = None #存储mask数据
        self.mark = None #存储不同要素的属性值

    def get_left_top(self):
        return self.geotransform[3], self.geotransform[0]#获取影像左上角地理坐标

    def get_pixel_height_width(self):
        return abs(self.geotransform[5]), abs(self.geotransform[1])#获取像素的高和宽

    def __getitem__(self, *args):
        """

        Args:
            *args: range, an instance of tuple, ((start, stop, step), (start, stop, step))

        Returns:
            res: image block , array ,[bands......, height, weight]

        """
        if isinstance(args[0], tuple) and len(args[0]) == 2:#检查 args 是否长度为 2
            # get params
            start_row, end_row = args[0][0].start, args[0][0].stop#行
            start_col, end_col = args[0][1].start, args[0][1].stop#列
            start_row = 0 if start_row is None else start_row# 如果起始行或列的索引为 None，默认从 0 开始
            start_col = 0 if start_col is None else start_col
            num_row = self.row if end_row is None else (end_row - start_row)# 如果结束行或列索引为 None，默认到影像的最大行数或列数
            num_col = self.col if end_col is None else (end_col - start_col)# 计算要读取的行数和列数（num_row 和 num_col）
            # dataset read image array
            res = self.dataset.ReadAsArray(start_col, start_row, num_col, num_row)
            return res
        else:
            raise NotImplementedError('the param should be [a: b, c: d] !')

    def clip_tif_with_grid(self, clip_size, begin_id, out_dir):
        """裁剪遥感影像
        clip image with grid
        Args:
            clip_size: int
            out_dir: str

        Returns:

        """
        if not os.path.exists(out_dir):
            # check the dir
            os.makedirs(out_dir)
            print('create dir', out_dir)#检查目标目录 out_dir 是否存在,如果不存在，则创建该目录

        row_num = int(1)#根据影像的行数 self.row 和列数 self.col，计算可以裁剪的网格数量
        col_num = int(1)#row_num：影像在行方向可以分成的网格数；col_num：影像在列方向可以分成的网格数

        gtiffDriver = gdal.GetDriverByName('GTiff')#获取 GDAL 的 GeoTIFF 驱动，用于后续保存裁剪后的影像
        if gtiffDriver is None:
            raise ValueError("Can't find GeoTiff Driver")

        count = 1
        # for i in range(row_num):# i 遍历行方向的网格；j 遍历列方向的网格
        #     for j in range(col_num):
                # if begin_id+i*col_num+j in self.mark:
                #     continue
                # 行范围：i * clip_size 到 (i + 1) * clip_size
                # 列范围：j * clip_size 到 (j + 1) * clip_size
        clipped_image = np.array(self[0: self.row, 0: self.col])
        clipped_image = clipped_image.astype(np.int8)

        try:
            save_path = os.path.join(out_dir, '%d.tif' % (begin_id))#保存影像
            save_image_with_georef(clipped_image, gtiffDriver,
                                           self.dataset, self.col, self.row, save_path)
            print('clip successfully！(%d/%d)' % (count, row_num * col_num))
            count += 1
        except Exception:
            raise IOError('clip failed!%d' % count)

        return row_num * col_num# 返回图像总个数

    def clip_mask_with_grid(self, clip_size, begin_id, out_dir):
        """裁剪mask
        clip mask with grid
        Args:
            clip_size: int
            out_dir: str

        Returns:

        """
        if not os.path.exists(out_dir):
            # check the dir
            os.makedirs(out_dir)
            print('create dir', out_dir)

        row_num = int(1)#划分网格数量
        col_num = int(1)

        gtiffDriver = gdal.GetDriverByName('GTiff')#加载GDAL驱动
        if gtiffDriver is None:
            raise ValueError("Can't find GeoTiff Driver")

        # self.mark = []

        count = 1
        # for i in range(row_num):
        #     for j in range(col_num):#循环裁剪图像
        #         clipped_image = np.array(self.mask[0, i * clip_size: (i + 1) * clip_size, j * clip_size: (j + 1) * clip_size])
        clipped_image = np.array(self.mask[0, 0: self.row, 0: self.col])
        ins_list = np.unique(clipped_image)
                # if len(ins_list) <= 1:
                #     self.mark.append(begin_id+i*col_num+j)
                #     continue
        ins_list = ins_list[1:]# 对于每个类别值 ins_list[id]生成对应的二值mask
        for id in range(len(ins_list)):
            bg_img = np.zeros((clipped_image.shape)).astype(np.int8)
            if ins_list[id] > 0:
                bg_img[np.where(clipped_image == ins_list[id])] = 255
            try:
                mark_value = self.mark[id]
                save_path = os.path.join(out_dir, '%d_%d_%s.tif' % (begin_id, id, mark_value))
                save_image_with_georef(bg_img, gtiffDriver,
                                               self.dataset, self.col, self.row, save_path)
                print('clip mask successfully！(%d/%d)' % (count, row_num * col_num))#保存二值mask
                count += 1
            except IndexError as e:
            # 如果索引错误，则跳过当前循环
                print(f"IndexError: Could not find mark value for id {id}, skipping...")
                continue
            except Exception as e:
                # 捕获其他异常，输出异常信息并跳过
                print(f"Error occurred while processing id {id} (count {count}): {str(e)}")
                continue
            # except Exception:
            #     raise IOError('clip failed!%d' % count)
            #     continue

    def world2Pixel(self, x, y):
        """世界坐标转换为像素坐标
        Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
        the pixel location of a geospatial coordinate
        """
        ulY, ulX = self.get_left_top()
        distY, distX = self.get_pixel_height_width()#获取分辨率

        pixel_x = abs(int((x - ulX) / distX))#计算像素坐标
        pixel_y = abs(int((ulY - y) / distY))
        pixel_y = self.row if pixel_y > self.row else pixel_y#限制像素坐标范围
        pixel_x = self.col if pixel_x > self.col else pixel_x
        return pixel_x, pixel_y

    def mask_tif_with_shapefile(self, shapefile_path, label=255, attribute_field='type'):  # 将shp文件转换为mask
        """
        mask tif with shape file, supported point, line, polygon and multi polygons
        Args:
            shapefile_path:
            save_path:
            label:

        Returns:

        """
        driver = ogr.GetDriverByName('ESRI Shapefile')  # 读取shp文件
        dataSource = driver.Open(shapefile_path, 0)
        if dataSource is None:
            raise IOError('could not open!')
        gtiffDriver = gdal.GetDriverByName('GTiff')
        # if gtiffDriver is None:
        #     raise ValueError("Can't find GeoTiff Driver")

        layer = dataSource.GetLayer(0)  # 获取矢量图层，将地理坐标转为像素坐标
        # # Convert the layer extent to image pixel coordinates
        minX, maxX, minY, maxY = layer.GetExtent()
        ulX, ulY = self.world2Pixel(minX, maxY)

        # initialize mask drawing
        rasterPoly = Image.new("I", (self.col, self.row), 0)
        rasterize = ImageDraw.Draw(rasterPoly)  # 绘制mask
        self.mark = {}# 定义储存属性值的容器
        feature_num = layer.GetFeatureCount()  # get poly count；提取每个对象的几何类型（POLYGON, MULTIPOLYGON, LINESTRING, POINT）
        for i in range(feature_num):
            points = []  # store points
            pixels = []  # store pixels
            feature = layer.GetFeature(i)
            geom = feature.GetGeometryRef()
            if geom is None:
                print("Warning: Geometry is None for a feature in the shapefile.")
                continue  # 跳过当前 feature

            # 读取属性字段的值
            if attribute_field is not None:
                try:
                    field_index = feature.GetFieldIndex(attribute_field)  # 获取字段索引
                    if field_index == -1:
                        raise ValueError(f"Attribute field '{attribute_field}' not found in the shapefile.")
                    self.mark[i] = feature.GetField(field_index)  # 读取字段值
                except Exception as e:
                    print(f"Error in shapefile '{shapefile_path}': {e}")
                    continue  # 跳过当前feature，继续处理其他feature
            feature_type = geom.GetGeometryName()
            if feature_type == 'POLYGON' or 'MULTIPOLYGON':  # 处理多边形
                # multi polygon operation
                # 1. use label to mask the max polygon
                # 2. use -label to mask the other polygon
                for j in range(geom.GetGeometryCount()):
                    sub_polygon = geom.GetGeometryRef(j)
                    if feature_type == 'MULTIPOLYGON':
                        sub_polygon = sub_polygon.GetGeometryRef(0)
                    for p_i in range(sub_polygon.GetPointCount()):
                        px = sub_polygon.GetX(p_i)
                        py = sub_polygon.GetY(p_i)
                        points.append((px, py))

                    for p in points:  # 使用 world2Pixel 方法，将地理坐标转换为像素坐标
                        origin_pixel_x, origin_pixel_y = self.world2Pixel(p[0], p[1])
                        # the pixel in new image
                        new_pixel_x, new_pixel_y = origin_pixel_x, origin_pixel_y
                        pixels.append((new_pixel_x, new_pixel_y))

                    rasterize.polygon(pixels, i + 1)  # 使用 polygon 方法在掩膜图像上绘制多边形
                    pixels = []
                    points = []
                    if feature_type != 'MULTIPOLYGON':
                        label = -abs(label)

                # restore the label value
                label = abs(label)
            else:  # 遍历当前几何对象中的每个点，并将点的坐标添加到列表 points 中
                for j in range(geom.GetPointCount()):
                    px = geom.GetX(j)
                    py = geom.GetY(j)
                    points.append((px, py))

                for p in points:  # 地理坐标转像素坐标
                    origin_pixel_x, origin_pixel_y = self.world2Pixel(p[0], p[1])
                    # the pixel in new image
                    new_pixel_x, new_pixel_y = origin_pixel_x, origin_pixel_y
                    pixels.append((new_pixel_x, new_pixel_y))

                feature.Destroy()  # delete feature

                if feature_type == 'LINESTRING':  # 在mask上绘制点和线
                    rasterize.line(pixels, i + 1)
                if feature_type == 'POINT':
                    # pixel x, y
                    rasterize.point(pixels, i + 1)

        mask = np.array(rasterPoly)
        self.mask = mask[np.newaxis, :]  # extend an axis to three




    def clip_tif_and_shapefile(self, clip_size, begin_id, shapefile_path, out_dir):
        # 调用 mask_tif_with_shapefile 方法，将shp文件转换为mask格式
        self.mask_tif_with_shapefile(shapefile_path)
        # 调用 clip_mask_with_grid 方法，将生成的掩膜（self.mask）按照指定的网格大小裁剪成小块并且保存
        self.clip_mask_with_grid(clip_size=clip_size, begin_id=begin_id, out_dir=out_dir + '/annotations')
        # 调用 clip_tif_with_grid 方法，将原始影像按照同样的网格大小裁剪成小块
        pic_id = self.clip_tif_with_grid(clip_size=clip_size, begin_id=begin_id, out_dir=out_dir + '/emcv_2024')
        return pic_id
# C,H,W->H,W,C
def channel_first_to_last(image):
    """

    Args:
        image: 3-D numpy array of shape [channel, width, height]

    Returns:
        new_image: 3-D numpy array of shape [height, width, channel]
    """
    new_image = np.transpose(image, axes=[1, 2, 0])
    return new_image
# H,W,C->C,H,W
def channel_last_to_first(image):
    """

    Args:
        image: 3-D numpy array of shape [channel, width, height]

    Returns:
        new_image: 3-D numpy array of shape [height, width, channel]
    """
    new_image = np.transpose(image, axes=[2, 0, 1])
    return new_image
# 保存tif图像
def save_image_with_georef(image, driver, original_ds, offset_x=0, offset_y=0, save_path=None):
    """

    Args:
        save_path: str, image save path
        driver: gdal IO driver
        image: an instance of ndarray
        original_ds: a instance of data set
        offset_x: x location in data set
        offset_y: y location in data set

    Returns:

    """
    # get Geo Reference
    ds = gdalnumeric.OpenArray(image)
    gdalnumeric.CopyDatasetInfo(original_ds, ds, xoff=offset_x, yoff=offset_y)
    driver.CreateCopy(save_path, ds)
    # write by band
    clip = image.astype(np.int8)
    # write the dataset
    if len(image.shape)==3:
        for i in range(image.shape[0]):
            ds.GetRasterBand(i + 1).WriteArray(clip[i])
    else:
        ds.GetRasterBand(1).WriteArray(clip)
    del ds
# 将mask按照tif图像的格式保存
def define_ref_predict(tif_dir, mask_dir, save_dir):
    """
    define reference for raster referred to a geometric raster.
    Args:
        tif_dir: the dir to save referenced raster
        mask_dir:
        save_dir:

    Returns:

    """
    # 获取文件列表
    tif_list = glob.glob(os.path.join(tif_dir, '*.tif'))

    mask_list = glob.glob(os.path.join(mask_dir, '*.png'))
    mask_list += (glob.glob(os.path.join(mask_dir, '*.jpg')))
    mask_list += (glob.glob(os.path.join(mask_dir, '*.tif')))

    tif_list.sort()
    mask_list.sort()

    os.makedirs(save_dir, exist_ok=True)
    gtiffDriver = gdal.GetDriverByName('GTiff')
    if gtiffDriver is None:
        raise ValueError("Can't find GeoTiff Driver")
    for i in range(len(tif_list)):
        save_name = tif_list[i].split('\\')[-1]
        save_path = os.path.join(save_dir, save_name)#根据文件名生成保存路径
        tif = GeoTiff(tif_list[i])
        mask = np.array(Image.open(mask_list[i]))
        mask = channel_last_to_first(mask)# 转换mask的通道顺序
        save_image_with_georef(mask, gtiffDriver, tif.dataset, save_path=save_path)# 保存

class GeoShaplefile(object):
    def __init__(self, file_path=""):
        self.file_path = file_path# 传入shp文件路径
        self.layer = ""
        self.minX, self.maxX, self.minY, self.maxY = (0, 0, 0, 0)
        self.feature_type = ""
        self.feature_num = 0
        self.open_shapefile()
    def open_shapefile(self):# 打开shp文件
        driver = ogr.GetDriverByName('ESRI Shapefile')
        dataSource = driver.Open(self.file_path, 0)
        if dataSource is None:
            raise IOError('could not open!')
        gtiffDriver = gdal.GetDriverByName('GTiff')
        if gtiffDriver is None:
            raise ValueError("Can't find GeoTiff Driver")
        # 获取图层和范围信息
        self.layer = dataSource.GetLayer(0)
        self.minX, self.maxX, self.minY, self.maxY = self.layer.GetExtent()
        self.feature_num = self.layer.GetFeatureCount()  # get poly count
        if self.feature_num > 0:# 获取第一个要素的几何类型
            polygon = self.layer.GetFeature(0)
            geom = polygon.GetGeometryRef()
            # feature type
            self.feature_type = geom.GetGeometryName()

def clip_from_file(clip_size, root, img_path, shp_path):#裁剪图像与shp文件
    # img_list = os.listdir(root + '/' + img_path)
    img_list = os.listdir(img_path)
    n_img = len(img_list)# 获取图像列表img_list
    pic_id = 0
    for i in range(n_img):
        # tif = GeoTiff(root + '/' + img_path + '/' + img_list[i])# 为每张图像创建 GeoTiff 类
        tif = GeoTiff(img_path + '/' + img_list[i])
    # for i in range(n_img):
    #     try:
    #         # 尝试读取图像
    #         tif = GeoTiff(img_path + '/' + img_list[i])
    #     except Exception as e:  # 捕获所有异常，也可以替换成更具体的异常类型（如 IOError/RuntimeError）
    #         print(f"跳过无法读取的图像: {img_list[i]}，错误信息: {str(e)}")
    #         continue  # 跳过这张图片
        img_id = img_list[i].split('.', 1)[0]# 从图像文件名中提取 ID
        # 调用 GeoTiff 类的 clip_tif_and_shapefile 方法来裁剪图像
        # pic_num = tif.clip_tif_and_shapefile(clip_size, pic_id, root + '/' + shp_path + '/' +  img_id + '.shp', root + '/dataset')
        pic_num = tif.clip_tif_and_shapefile(clip_size, pic_id, shp_path + '/' +  img_id + '.shp', root + '/dataset')
        pic_id += pic_num

if __name__ == '__main__':
    root = r'./example_data/original_data'
    img_path = 'D:/DOWNLO/building file/sliceShp/111'
    shp_path = 'D:/DOWNLO/building file/sliceShp/111'
    clip_from_file(512, root, img_path, shp_path)

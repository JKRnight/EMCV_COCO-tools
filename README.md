

# EMCV_COCO tools

Efficient Conversion of Geospatial Data to Deep Learning Standard Format



### 📖 Project Overview

This tool automates the pipeline for converting ArGIS `shp` file vector data (e.g., building/road contours) into the `COCO` dataset format, supporting instance segmentation and object detection tasks. Ideal for remote sensing analysis, urban planning, and similar applications.
**Core Features**

• Extract geometric features from `shp` files and generate instance masks

• Set the class for the instance mask based on the information contained in the `shp` file attribute table

• Generate `JSON` markup files (with images, annotations, categories, etc.) that conform to the `COCO` standard.

• Support for multi-category tag mapping (e.g. building→ category_id = 1)
### ⚙️ Installation Guide
install GDAL（Recommended Conda）  
```sh
conda install -c conda-forge gdal  
```
install python dependencies  
```sh
pip install -r requirements.txt
```
### 🚀 Usage Instructions
**1. Configure File Structure**
```
filetree 
├── /pycococreatortools/
├── .gitignore
├── README.md
├── requirements.txt
├── slice_dataset.py
├── shape_to_coco.py
├── tif_process.py
```
**2. Run Conversion Script**
```
python shape_to_coco.py --step clip # Extract geometric features
python shape_to_coco.py --step split # Split the dataset
python shape_to_coco.py --step coco_all # Generate all annotations
python shape_to_coco.py --step coco_train # Generate train set annotations
python shape_to_coco.py --step coco_eval # Generate eval set annotations
python shape_to_coco.py --step coco_test # Generate test set annotations
```
**3. Visualization**

The results can be visualized using [cocoapi](https://github.com/cocodataset/cocoapi)

### 🔗 Dataset Download
### 🛠 Custom Configuration
**Category ID Mapping**

Change the value of `attribute_field` in `mask_tif_with_shapefile` to select the attribute table field

**Category ID**

Define the correspondence between the category name and id in `CATEGORIES`：

        'id': 1,
        'name': 'building',
        'supercategory': 'building',

        'id': 2,
        'name': 'road',
        'supercategory': 'road',

        'id': 3,
        'name': 'waterbody',
        'supercategory': 'waterbody',

        'id': 4,
        'name': 'forest',
        'supercategory': 'forest',

### 🙏 Acknowledgements
The code for this project references the project by [shp2coco](https://github.com/Dingyuan-Chen/shp2coco) <br>
Thanks to the Third Party Libs<br>
[geotool](https://github.com/Kindron/geotool)<br>
[pycococreator](https://github.com/waspinator/pycococreator)<br>

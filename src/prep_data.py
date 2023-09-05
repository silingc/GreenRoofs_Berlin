# test area: Berlin Mitte (frame .shp created)
# green roof ground truth data: convert to 4326, buffer (some polygons have self-intersect issues) -> dissolve
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
import geopandas as gpd
gr_buf_path = '/Users/silingchen/PycharmProjects/Roofpedia/Roofpedia_Geodataset/temp/GR_senat_4326_buff.geojson'
gr_buf = gpd.read_file(gr_buf_path)
gr_buf_dis = gr_buf.dissolve(by='gruen_kat')
gr_buf_dis.to_file('/Users/silingchen/PycharmProjects/Roofpedia/Roofpedia_Geodataset/temp/GR_senat_4326_buff_dis.shp')

# create the plot
fig, ax = plt.subplots(figsize = (10,6))

# plot the data
gr_buf_dis.reset_index().plot(column = 'gruen_kat', ax=ax)

# Set plot axis to equal ratio
ax.set_axis_off()
plt.axis('equal')
plt.show()

frame_path = '/Users/silingchen/PycharmProjects/Roofpedia/Roofpedia_Geodataset/temp/Berlin_Mitte_frame.shp'
frame = gpd.read_file(frame_path)
gr_clip = gpd.clip(gr_buf_dis, frame)
gr_clip.to_file('/Users/silingchen/PycharmProjects/Roofpedia/Roofpedia_Geodataset/temp/GR_senat_4326_buff_dis_clip_.shp')

# then use generate XYZ tiles on QGIS (documented in Preparing xyz tiles for Roofpedia.docx) to the clipped .shp

#%% Jan 2023
#%% define CRS of .tif (merged .ecws) as 25833 and trial of gdal2tiles
### EXECUTED IN TERMINAL!
import sys, os, os.path, string, array, math, shutil, glob, re, locale, math, traceback, time, datetime, csv
from osgeo import gdal, gdalconst, osr
from osgeo.gdalconst import *
import subprocess

driver = gdal.GetDriverByName("ECW")
#driver = gdal.GetDriverByName("GTiff")
srs = osr.SpatialReference()
srs.ImportFromEPSG(25833)
srs = osr.SpatialReference()
#sr.SetProjection ("EPSG:25833")
#sr_wkt = sr.ExportToWkt()
file # .tif to define CRS
ds = gdal.Open(file, gdal.GA_Update)
if ds:
    print('Updating projection for ' + file)
    ds.SetProjection(srs.ExportToWkt())
    ds = None # save, close
else:
    print('Could not open with GDAL: ' + file)
# exit python
gdal_translate -of VRT -ot Byte -scale /Users/silingchen/PycharmProjects/Roofpedia/results/02Images/Berlin_test/temp/West_test.tif temp.vrt
gdal2tiles.py -p mercator -z 19 -w leaflet --processes=4 --xyz -r average -a 0.0 --s_srs EPSG:25833  temp.vrt  /Users/silingchen/PycharmProjects/Roofpedia/results/02Images/Berlin_test/temp/xyz_West_25833_gdal


#%% compare converted tiles with 25833 and 4326
import os
path1 = '/Users/silingchen/PycharmProjects/Roofpedia/results/02Images/Berlin_test/temp/xyz_Mitte_25833'
path2 = '/Volumes/LaCie/Sling/Roofpedia/Berlin_test/xyz/Mitte_reproject4326'
files1 = glob.glob(path1 + '/*/*/*.png', recursive=True) ### / for mac
files2 = glob.glob(path2 + '/*/*/*.png', recursive=True) ### / for mac
len(files1), len(files2)

#%% select "edge" tile .ecw -> generate xyz
import pandas as pd
import numpy as np
import shutil, os
tile_name = list(pd.read_excel('/Users/silingchen/PycharmProjects/Roofpedia/Roofpedia_Geodataset/DOP20RGB/Blattschnitt2x2km.xlsx', sheet_name='selected').values.flatten())
take_list = [x for x in  tile_name if str(x) != 'nan']
count = 0
dest_folder = '/Users/silingchen/PycharmProjects/Roofpedia/Roofpedia_Geodataset/DOP20RGB/_edges'
for ii in take_list:
    for ort in ['Mitte','Nord','Nordost','Nordwest','Ost','Sued','Suedost','Suedwest','West']:
        ori_file = os.path.join('/Users/silingchen/PycharmProjects/Roofpedia/Roofpedia_Geodataset/DOP20RGB', ort, 'dop20_'+ii+'.ecw')
        if os.path.exists(ori_file):
            dest_file = os.path.join('/Users/silingchen/PycharmProjects/Roofpedia/Roofpedia_Geodataset/DOP20RGB/_edges', 'dop20_'+ii+'.ecw')
            if os.path.exists(dest_file):
                print('file already exists, file: ', dest_file)
            else:
                shutil.copy(ori_file, dest_folder)
                another_ori = os.path.join('/Users/silingchen/PycharmProjects/Roofpedia/Roofpedia_Geodataset/DOP20RGB', ort, 'dop20_'+ii+'.eww')
                another_dest = os.path.join('/Users/silingchen/PycharmProjects/Roofpedia/Roofpedia_Geodataset/DOP20RGB/_edges','dop20_' + ii + '.eww')
                shutil.copy(another_ori,another_dest)
                print(ii)
                count += 1
#%% FOR EACH SMALLER SEGMENT
#key = 'up'
for key in ['down','left','right']:
    tile_name = list(pd.read_excel('/Users/silingchen/PycharmProjects/Roofpedia/Roofpedia_Geodataset/DOP20RGB/Blattschnitt2x2km.xlsx', sheet_name=key).values.flatten())
    take_list = [x for x in  tile_name if str(x) != 'nan']
    count = 0
    addfolder = '/'+key
    dest_folder = '/Users/silingchen/PycharmProjects/Roofpedia/Roofpedia_Geodataset/DOP20RGB/_edges'+ addfolder
    ori_folder = '/Users/silingchen/PycharmProjects/Roofpedia/Roofpedia_Geodataset/DOP20RGB/_edges'
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    for ii in take_list:
        found = False
        ori_file = os.path.join(ori_folder, 'dop20_'+ii+'.ecw')
        if os.path.exists(ori_file):
            dest_file = os.path.join(dest_folder, 'dop20_'+ii+'.ecw')
            if os.path.exists(dest_file):
                #print('file already exists, file: ', dest_file)
                found = True
            else:
                shutil.copy(ori_file, dest_folder)
                another_ori = os.path.join(ori_folder, 'dop20_'+ii+'.eww')
                another_dest = os.path.join(dest_folder,'dop20_' + ii + '.eww')
                shutil.copy(another_ori,another_dest)
                print(ii)
                count += 1
                found = True
        if found == False:
            print('File not found: ', ii)
"""
#%% identify white/black color for each segment
from PIL import Image
import glob, os, shutil
import numpy as np
key = '' #'Nord' #'Mitte'#'down' #'left'
#source_path = '/Users/silingchen/PycharmProjects/Roofpedia/results/02Images/Berlin_test/temp/xyz_edges_%s_25833' % key
source_path = '/Users/silingchen/PycharmProjects/Roofpedia/results/02Images/Berlin_test/temp/xyz_%s_25833' % key
files_source = glob.glob(source_path + '/*/*/*.png', recursive=True) ### / for mac
target_path = '/Users/silingchen/PycharmProjects/Roofpedia/results/02Images/Berlin_test/temp/sorted'
#tile = '/Users/silingchen/PycharmProjects/Roofpedia/results/02Images/Berlin_test/temp/xyz_edges_left_25833/19/281540/171852.png'
#tile = '/Users/silingchen/PycharmProjects/Roofpedia/results/02Images/Berlin_test/temp/xyz_edges_left_25833/19/281540/171839.png'
source = list()
target = list()
incomplete = list()
incomplete_pixelcount = list()
repetitive = list()
#tile = '/Users/silingchen/PycharmProjects/Roofpedia/results/02Images/Berlin_test/temp/xyz_edges_left_25833/19/281196/171655.png'
for tile in files_source:
    img = Image.open(tile)
    img_arr = np.array(img)
    count = 0
    #countercount = 0
    for col in range(len(img_arr)): #256
        for row in range(len(img_arr[0])): #256
            # check edge, if and how many pixels are white
            if col == 0 or row == 0 or col == len(img_arr) or row == len(img_arr[0]):
                #print(sum(img_arr[col][row]))
                if sum(img_arr[col][row]) == 0 or sum(img_arr[col][row]) == 255: # white pixel OR black pixel
                    count += 1
            #else:
            #    countercount += 1
    if count == 0: # complete img
        dest = target_path + '/' + tile[-20:]
        if not os.path.exists(os.path.dirname(dest)):
            os.makedirs(os.path.dirname(dest))
        if os.path.exists(dest):
            #print('File exists: ', dest)
            repetitive.append(dest)
        else:
            source.append(tile)
            target.append(dest)
            shutil.copy(tile, dest)
    else:
        incomplete.append(tile)
        incomplete_pixelcount.append(count)

import json
with open(os.path.join('/Users/silingchen/PycharmProjects/Roofpedia/results/02Images/Berlin_test/temp/sorted','%s.txt'% key), 'w') as f:
  json.dump([{'source':source, 'target':target,'incomplete_tiles':incomplete,'incomplete_count':incomplete_pixelcount,'repetitive':repetitive}], f, ensure_ascii=False)
#count(x<100 for x in incomplete_pixelcount)
'''
for ii, i in enumerate(incomplete_pixelcount):
    if i < 100:
        print(incomplete[ii], i)
'''
print(len(repetitive))
#%% check repetitive tiles
import copy
repetitive_backup = copy.deepcopy(repetitive)
#repetitive = copy.deepcopy(repetitive_backup)
source_path = '/Users/silingchen/PycharmProjects/Roofpedia/results/02Images/Berlin_test/temp/xyz_%s_25833' % key
if len(repetitive) > 0:
    for tile in repetitive:
        path_h = tile.split('/')[-3:]
        tile2 = os.path.join(source_path, path_h[0],path_h[1],path_h[2])
        #print(tile2)
        img1 = Image.open(tile)
        img_arr1 = np.array(img1)
        img2 = Image.open(tile2)
        img_arr2 = np.array(img2)
        break
        if (img_arr1 == img_arr2).all:
            #print('ok')
            repetitive.remove(tile)
len(repetitive)
#%% screen xyz tiles
'''
# calculate file size in KB, MB, GB
def convert_bytes(size):
    # Convert bytes to KB, or MB or GB
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return "%3.1f %s" % (size, x)
        size /= 1024.0

f_size = os.path.getsize(tile)
x = convert_bytes(f_size)
print('file size is', x)
'''
target_path = '/Users/silingchen/PycharmProjects/Roofpedia/results/02Images/Berlin_test/temp/sorted'
for tile in repetitive:
    path_h = tile.split('/')[-3:]
    tile2 = os.path.join(source_path, path_h[0], path_h[1], path_h[2])
    f_size1 = os.path.getsize(tile)
    f_size2 = os.path.getsize(tile2)
    #break
    if (f_size1 - f_size2)/1024 < 1.5: # KB
        repetitive.remove(tile)
        print('remove ', tile)
    else:
        #print((f_size1 - f_size2)/1024 )
        if f_size1 - f_size2 < 0: # in source the tile is larger than the one in target path -> replace
            shutil.copy(tile2, tile1)
            print('Replace file ', tile)
            #break

len(repetitive)
#%% read from json
import json
f = open(os.path.join('/Users/silingchen/PycharmProjects/Roofpedia/results/02Images/Berlin_test/temp/sorted','%s.txt'% key))
data = json.load(f)
incomplete = data[0]['incomplete_tiles']
incomplete_pixelcount = data[0]['incomplete_count']
f.close()
#%% arrange datasets into a dataframe
# cols: y, z, path, white_count
import pandas as pd
collect = pd.DataFrame()
ys = []
zs = []
paths = []
counts = []
#y, z, path = tile.split('/')[-2], tile.split('/')[-1].split('.')[0], tile.split('/')[-4]
for ii, tile in enumerate(incomplete):
    y, z, path = tile.split('/')[-2], tile.split('/')[-1].split('.')[0], tile.split('/')[-4]
    ys.append(y)
    zs.append(z)
    paths.append(path)
    counts.append(incomplete_pixelcount[ii])

collect['y'] = ys
collect['z'] = zs
collect['path'] = paths
collect['count'] = counts
collect.head()
collect.to_excel(os.path.join('/Users/silingchen/PycharmProjects/Roofpedia/results/02Images/Berlin_test/temp/sorted','%s.xlsx'% key))

#%% merge incomplete dataframes and see if there are overlapping tiles
check_regions = ['Mitte','Nord','up','down']
for ii, key in enumerate(check_regions):
    if ii == 0:
        collect = pd.read_excel(os.path.join('/Users/silingchen/PycharmProjects/Roofpedia/results/02Images/Berlin_test/temp/sorted','%s.xlsx' % key),header=0,index_col=0)
    else:
        collect = collect.append(pd.read_excel(os.path.join('/Users/silingchen/PycharmProjects/Roofpedia/results/02Images/Berlin_test/temp/sorted','%s.xlsx' % key),header=0,index_col=0))

unique_rows = collect.groupby(['y','z']).size().reset_index().rename(columns={0:'count_rep'})
if len(collect) == len(unique_rows):
    print('No repetition between check regions: ', check_regions)
else:
    print('Repetition found: ', len(collect)-len(unique_rows))
"""
#%% sort tiles and obtain a FINAL sorted complete set of tiles
# 1) get all tile names
# 2) compare the repititive ones -> get the ones with larger size; to be sure, add them to one folder (y_z_key.png) for comparison
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd, os, shutil, glob, numpy as np
keys = ['Mitte','Nord','Nordost_Ost', 'Nordwest','Sued', 'Suedost', 'Suedwest','West','edges_down','edges_up','edges_left', 'edges_right']
yznames = []
regions = []
sizes = []
all_files = pd.DataFrame()
for key in keys:
    print(key)
    source_path = '/Users/silingchen/PycharmProjects/Roofpedia/results/02Images/Berlin_test/temp/xyz_%s_25833' % key
    files_source = glob.glob(source_path + '/*/*/*.png', recursive=True)  ### / for mac
    for file in files_source:
        yznames.append('%s_%s' % (file.split('/')[-2],file.split('/')[-1]))
        regions.append(key)
        sizes.append(os.path.getsize(file)/1024)

all_files ['yz'] = yznames
all_files ['region'] = regions
all_files['size'] = sizes
all_files.head()
#all_files.to_excel(os.path.join('/Users/silingchen/PycharmProjects/Roofpedia/results/02Images/Berlin_test/temp/sorted','%s.xlsx'% key))
print('Repetition: ', len(all_files) - len(list(set(yznames))))


def copy_tile(df_in, des_folder):
    ''' with input of a ONE-row df, copy tile to the dest folder '''
    key = df_in['region'].to_string(index=False)
    yz = df_in['yz'].to_string(index=False)
    ori_file = '/Users/silingchen/PycharmProjects/Roofpedia/results/02Images/Berlin_test/temp/xyz_%s_25833/19/%s/%s' % (
    key, yz.split('_')[0], yz.split('_')[1])
    folder = os.path.join(des_folder, yz.split('_')[0])
    if not os.path.exists(folder):
        os.makedirs(folder)
    dest_file = os.path.join(folder, yz.split('_')[-1])
    if os.path.exists(dest_file):
        print('File already exists ', dest_file)
        #break
    else:
        shutil.copy(ori_file, dest_file)
    return
uniques = list(set(yznames))
des_folder = '/Users/silingchen/PycharmProjects/Roofpedia/results/02Images/Berlin_test/temp/sorted_3001'
rep_tiles = pd.DataFrame()
for unitile in uniques:
    sel = all_files.loc[all_files['yz']== unitile]
    if len(sel) == 1: #unique tile
        # recover it's name -> copy to the final folder
        #break
        copy_tile(sel, des_folder)
    else:
        ## compare files -> take the largest to the folder, and ALL related files are exported (as .xlsx, also tiles copied to a folder for better comparison)
        rep_tiles = rep_tiles.append(sel)
        sel = sel.sort_values(by=['size'],ascending=False)
        take = pd.DataFrame(sel.iloc[0]).transpose() #sel[sel['size']==np.max(sel['size'])]
        # possible that more than 1 tiles have the largest size -> CHECK
        copy_tile(take, des_folder)

for tt in rep_tiles.index:
    tile = rep_tiles[rep_tiles.index==tt]
    des_folder = '/Users/silingchen/PycharmProjects/Roofpedia/results/02Images/Berlin_test/temp/repetitive_3001'
    key = tile['region'].to_string(index=False)
    yz = tile['yz'].to_string(index=False)
    ori_file = '/Users/silingchen/PycharmProjects/Roofpedia/results/02Images/Berlin_test/temp/xyz_%s_25833/19/%s/%s' % (
        key, yz.split('_')[0], yz.split('_')[1])
    dest_file = os.path.join(des_folder, yz.split('.')[0]+'_'+key+'.png')
    if os.path.exists(dest_file):
        print('File already exists ', dest_file)
    else:
        shutil.copy(ori_file, dest_file)

print(len(rep_tiles))
rep_tiles.reset_index(drop=True)
rep_tiles.to_excel(os.path.join('/Users/silingchen/PycharmProjects/Roofpedia/results/02Images/Berlin_test/temp/repetitive_3001','rep_tiles.xlsx'))

#%% identify white/black color for each segment <- didnt run
from PIL import Image
import glob, os, shutil
import numpy as np
source_path =  '/Users/silingchen/PycharmProjects/Roofpedia/results/02Images/Berlin_test/temp/sorted_3001'
files_source = glob.glob(source_path + '/*/*/*.png', recursive=True) ### / for mac
target_path = '/Users/silingchen/PycharmProjects/Roofpedia/results/02Images/Berlin_test/temp/suspect_3001'
tile = '/Users/silingchen/PycharmProjects/Roofpedia/results/02Images/Berlin_test/temp/sorted_3001/281850/172308.png'
tile = '/Users/silingchen/PycharmProjects/Roofpedia/results/02Images/Berlin_test/temp/sorted_3001/281848/172292.png'
source = list()
target = list()
repetitive = list()
#tile = '/Users/silingchen/PycharmProjects/Roofpedia/results/02Images/Berlin_test/temp/xyz_edges_left_25833/19/281196/171655.png'
for tile in files_source:
    img = Image.open(tile)
    img_arr = np.array(img)
    count = 0
    #countercount = 0
    for col in range(len(img_arr)): #256
        for row in range(len(img_arr[0])): #256
            # check edge, if and how many pixels are white
            if col == 0 or row == 0 or col == len(img_arr) or row == len(img_arr[0]):
                #print(sum(img_arr[col][row]))
                if sum(img_arr[col][row]) == 0 or sum(img_arr[col][row]) == 255: # white pixel OR black pixel
                    count += 1
            #else:
            #    countercount += 1
    if count != 0: # incomplete img
        dest = target_path + '/' + '%s_%s' % (tile.split('/')[-2],tile.split('/')[-1])
        if os.path.exists(dest):
            print('File exists: ', dest)
            repetitive.append(dest)
        else:
            shutil.copy(tile, dest)
#%% compare files
file1 = '/Users/silingchen/PycharmProjects/Roofpedia/Roofpedia_Geodataset/Belrin_Gruendaecher_Dacteilfl_Gebae.geojson'
file2 = '/Users/silingchen/PycharmProjects/Roofpedia/Roofpedia_Geodataset/temp/Berlin_Gründächer_Gebäude.shp'
import geojson
with open(file1) as f:
    gj = geojson.load(f)
gj

#%% check if folder has '.DS_Store'
ground_path = '/Users/silingchen/PycharmProjects/Roofpedia_2022/dataset/data_split'
import pathlib
paths = list(pathlib.Path(ground_path).rglob("*"))
for path in paths:
    if '.DS_Store' in str(path):
        print('remove stupid folder ', str(path))
        os.remove(path)

subpath = ['training','evaluation','validation']
for path in subpath:
    paths = list(pathlib.Path(os.path.join(ground_path,path)).rglob("*"))
    print(path, len(paths))
#%% to tranfer data to cluster (via tubcloud, but large dataset doesnt work <.<)
import os, shutil
ground_path = '/Users/silingchen/PycharmProjects/Roofpedia_2022/dataset/Geodataset/temp/T/images/19'
#shutil.make_archive(ground_path, 'zip', ground_path)
#shutil.unpack_archive(ground_path+'.zip',ground_path)
for n in os.listdir(ground_path):
    if '.DS_Store' not in str(n) and '.zip' not in n:
        path = os.path.join(ground_path, n)
        shutil.make_archive(path, 'zip', path)
        shutil.rmtree(path)

#%% convert crs
# import necessary packages to work with spatial data in Python
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)
import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
ori_data = gpd.read_file("/Users/silingchen/PycharmProjects/Roofpedia_2022/dataset/Geodataset/temp/Gebäude_2016_fix.geojson")
print(ori_data.crs)
ori_data.loc[0]
# drop some columns to SPEED UP
conv_data  = ori_data.to_crs(epsg=3395)
conv_data.to_file("/Users/silingchen/PycharmProjects/Roofpedia_2022/dataset/Geodataset/temp/Gebaeude_2016_fix_3395.geojson", driver='GeoJSON')

world = world.to_crs("EPSG:3395") # world.to_crs(epsg=3395) wou
ori_data.plot()

#%% BUILDING -> fix geometry -> convert to 3395 with some columns deleted
import geopandas as gpd
bds = gpd.read_file('/Users/silingchen/PycharmProjects/Roofpedia_2022/dataset/Geodataset/2016/Berlin_Gründächer_Gebäude.shp')
bds.columns
import copy
bds_ = copy.deepcopy(bds)[['geometry']]
conv_bds  = bds_.to_crs(epsg=3395)
#conv_bds.head()
conv_bds.to_file("/Users/silingchen/PycharmProjects/Roofpedia_2022/dataset/Geodataset/temp/Gebaeude_2016_3395.geojson", driver='GeoJSON')
"""
conv_bds_ = conv_bds.dissolve()
conv_bds_.to_file("/Users/silingchen/PycharmProjects/Roofpedia_2022/dataset/Geodataset/temp/Gebaeude_2016_3395_dis.geojson", driver='GeoJSON')

from shapely.validation import make_valid
make_valid(bds_)
"""

ori_data = gpd.read_file("/Users/silingchen/Downloads/TRANSFER_zilla/Gebäude_2016/Berlin.geojson")
print(ori_data.crs)
ori_data.head()
rpj_data = ori_data.to_crs(epsg=4326)
rpj_data.crs
rpj_data.head()
rpj_data.to_file("/Users/silingchen/PycharmProjects/Roofpedia_2022/dataset/Geodataset/temp/Gebaeude_2016_3395_4326.geojson", driver='GeoJSON')

### CANNOT BE DONE AT LOCAL MACHINE -> SWITCHED TO CLUSTER
city = rpj_data[['geometry']]

# loading building polygons
city = 'results/01City/' + city_name + '.geojson'
city = gp.GeoDataFrame.from_file(city)[['geometry']]

city['area'] = city['geometry'].map(lambda p: p.area)  #
mask_dir = '/Users/silingchen/Downloads/TRANSFER_zilla/Roofpedia/03Masks/Mod2/Green/Berlin'
features = mask_to_feature(mask_dir)
prediction = gp.GeoDataFrame.from_features(features, crs=4326)

intersections = gp.sjoin(city, prediction, how="inner", op='intersects')
intersections = intersections.drop_duplicates(subset=['geometry'])
intersections.to_file('results/04Results/' + city_name + '_' + target_type + ".geojson", driver='GeoJSON')

conv_data.to_file("/Users/silingchen/PycharmProjects/Roofpedia_2022/dataset/Geodataset/temp/Gebaeude_2016_fix_3395.geojson", driver='GeoJSON')
''' check dependency of geopandas and ggf. install:
conda install pandas fiona shapely pyproj rtree
conda install pygeos --channel conda-forge
'''
#%% 24.02. split GT into 3 sets -> each save as .shp -> convert to xyz tiles



#%% for DOP 2022: merge v1-v3 datasets
import os
import glob
import numpy as np
import shutil
source_path = '/Volumes/T7/2020/DOP20RGB/tiles/versions'
files_target = glob.glob(source_path + '/*/*/*/*.png', recursive=True) ### / for mac
files_target
len(files_target)
short_files = [x.split('/')[-3:] for x in files_target]
short_files_ = ['{}/{}/{}'.format(x[0],x[1],x[2]) for x in short_files]
short_files_set = list(set(short_files_))
take_files = []

for x in short_files_set:
    ind = [i for i, e in enumerate(files_target) if x in e]
    """if type(ind) != list:

        print('found')
        break
    """
    sizes = []
    for id in ind:
        f_size = os.path.getsize(files_target[id])
        #f_size /= 1024.0
        #if f_size > 2*1024:
        sizes.append(f_size)
    if len(sizes) > 0:
        if np.max(sizes) > 2*1024:
            take_file = files_target[sizes.index(np.max(sizes))]
            take_files.append(take_file)

target_path = '/Volumes/T7/2020/DOP20RGB/tiles/sorted'
for take_file in take_files:
    path_h = take_file.split('/')[-3:]
    dest_tile = os.path.join(target_path, path_h[0], path_h[1], path_h[2])
    spath = os.path.join(target_path, path_h[0], path_h[1])
    if not os.path.exists(spath):
        os.makedirs(spath)
    shutil.copy(take_file, dest_tile)

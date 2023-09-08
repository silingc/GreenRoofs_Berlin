''' 
DOP images source: https://fbinter.stadt-berlin.de/fb/berlin/service_intern.jsp?id=a_luftbild2016_rgb@senstadt&type=FEED
files needed: building footprint, green roof ground truth, green roof predicted by RP
PRE-PROCESSING (QGIS): (1) unite layer crs (4326) (2) fix geometries (3) add ID for layers
* ground truth (Gt) dissolved took longer to intersect -> not done
-> (4) intersect GT and RP respectively with BD
   (5) dissolve calculate GR areas for intersect results
   (6) join intersect layers back to BD .shp
   (7) performance analysis
   (8) optimal (not included here): temporal change (plz)
'''
''' 
run predict_and_extract.py in command line: python predict_and_extract.py Berlin_2016_xxxxx Green
ALTERNATIVE: run the session below (adapted from predict_and_extract.py )
'''
import torch
import toml
from src.predict import predict
try:
    import paths as paths
except:
    import src.paths as paths

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["USE_PYGEOS"] = "0"

import geopandas as gp
import os
import numpy as np
import pandas as pd
try:
    import paths as paths
except:
    import src.paths as paths
from src.extract import mask_to_feature
import math

config = toml.load(os.path.join(paths.PROJECT_DIR, 'config/predict-config.toml'))
target_type = 'Green'
def predict_greenroofs_2016(plz):
    city_name = 'Berlin_2016_%s' % plz
    tiles_dir = os.path.join(paths.PROJECT_DIR, "results", '02Images', city_name)
    mask_dir = os.path.join(paths.PROJECT_DIR, "results", "03Masks", target_type, city_name)
    tile_size = config["img_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = config["checkpoint_path"]
    checkpoint_name = config["green_checkpoint"]
    chkpt = torch.load(os.path.join(paths.PROJECT_DIR, checkpoint_path, checkpoint_name), map_location=device)
    predict(tiles_dir, mask_dir, tile_size, device, chkpt)
    return

''' analysis after green roof prediction using Roofpedia '''
import seaborn as sns
import matplotlib.pyplot as plt

def extract_result(res, conditions):
    ''' 'TP','FP','FN','TN' '''
    true_positive = len(res[conditions[0]])
    false_positive = len(res[conditions[1]])
    false_negative = len(res[conditions[2]])
    true_negative = len(res[conditions[3]])
    precision = true_positive / (true_positive + false_positive)   
    accuracy = (true_positive+ true_negative )/ (false_negative + false_positive + true_negative + true_positive)
    iou = true_positive / (true_positive + false_positive + false_negative) # intersection over union
    recall = true_positive / (true_positive + false_negative)  # 0.5
    fscore = 2 * (precision * recall) / (precision + recall)  # 0.09786476868327403
    print('True positive: ', true_positive)
    print('True negative: ', true_negative)
    print('False positive: ', false_positive)
    print('False negative: ', false_negative)
    print('Precision: ', '{0:0.2f}'.format(precision))
    print('Recall: ', '{0:0.2f}'.format(recall))
    print('Accuracy: ', '{0:0.2f}'.format(accuracy))
    print('Intersection over Union (IoU): ', '{0:0.2f}'.format(iou))
    print('F-Score: ', '{0:0.2f}'.format(fscore))
    return false_negative, false_positive, true_negative, true_positive, fscore

def join_and_analyze(plz):
    target_type, city_name = 'Green', 'Berlin_2016_%s' % plz
    mask_dir = os.path.join(paths.PROJECT_DIR, "results", "03Masks", target_type, city_name)
    features = mask_to_feature(mask_dir)
    prediction = gp.GeoDataFrame.from_features(features, crs=4326)
    # loading building polygons
    city = os.path.join(paths.PROJECT_DIR, 'results/01City/' + city_name + '.geojson')
    city = gp.GeoDataFrame.from_file(city)  # [['geometry']]
    # city['area'] = city['geometry'].to_crs({'init': 'epsg:3395'}).map(lambda p: p.area)

    intersections = gp.sjoin(city, prediction, how="inner", op='intersects')
    intersections = intersections.drop_duplicates(subset=['geometry'])
    intersections.to_file(os.path.join(paths.PROJECT_DIR, 'results/04Results/' + city_name + '_' + target_type + '.shp'))

    bds_join = city.sjoin(prediction, how="left", op='intersects')
    bds_join = bds_join.drop_duplicates(subset=['geometry'])
    bds_join['gr_2016'] = [0 if math.isnan(x) else 1 for x in list(bds_join['index_right'])]
    bds_join = bds_join.drop(columns=['index_right'])
    bds_join.to_file(os.path.join(paths.PROJECT_DIR, 'results/04Results/' + city_name + '_' + target_type + '_city.shp'))

    gr_gt = gp.GeoDataFrame.from_file(os.path.join(paths.PROJECT_DIR,
                                                   'results/05Analysis/Berlin_Gründächer_DachteilflächenGebäude_2016_%s.shp' % plz))  # green roof ground truth from Senat (grs in 2016)
    # intersect gt with bds. if inetersection area < 0.01 (green roof area on bd), cannot be taken as green roof (this can be a mismeasurement of gt)
    intersections = gp.sjoin(gr_gt, bds_join, how='inner', op='intersects')
    intersections = intersections.drop_duplicates(subset=['geometry'])
    intersections = intersections.dissolve('bds16_id')
    intersections['area'] = intersections['geometry'].map(lambda p: p.area)
    intersections = intersections[intersections['area'] >= 5e-10]
    intersections = intersections[['geometry', 'gruen_kat', 'area', 'gt2016_id']]
    # intersections.to_file(os.path.join(paths.PROJECT_DIR, 'results/05Analysis/' + city_name + '_' + target_type + '_gt_dis.shp'))

    bds_join_gt = gp.sjoin(bds_join, intersections, how='left', op='intersects')
    bds_join_gt = bds_join_gt.drop_duplicates(subset=['geometry'])
    # bds_join_gt.to_file(os.path.join(paths.PROJECT_DIR, 'results/05Analysis/' + city_name + '_' + target_type + '_city_gt.shp'))
    bds_join_gt['gt_2016'] = [0 if math.isnan(x) else 1 for x in list(bds_join_gt['gt2016_id'])]

    # evaluate prediction performance
    bds_join_gt
    conditions = [
        (bds_join_gt['gt_2016'] == 1) & (bds_join_gt['gr_2016'] == 1),
        (bds_join_gt['gt_2016'] == 0) & (bds_join_gt['gr_2016'] == 1),
        (bds_join_gt['gt_2016'] == 1) & (bds_join_gt['gr_2016'] == 0),
        (bds_join_gt['gt_2016'] == 0) & (bds_join_gt['gr_2016'] == 0)]

    values = ['TP', 'FP', 'FN', 'TN']
    bds_join_gt['code'] = np.select(conditions, values)
    bds_join_gt.to_file(
        os.path.join(paths.PROJECT_DIR, 'results/05Analysis/' + city_name + '_' + target_type + '_city_gt.shp'))
    print('Analysis done for plz = ', plz)
    extract_result(bds_join_gt, conditions)
    return

for plz in [12357, 12627, 10623]:
    predict_greenroofs_2016(plz)
    join_and_analyze(plz)

def confusion_matrix(false_negative, false_positive, true_negative, true_positive, fscore, model):
    cf_matrix = np.array([[true_negative, false_positive], [false_negative, true_positive]])
    # source: https://www.stackvidhya.com/plot-confusion-matrix-in-python-and-why/
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in
    cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
    cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
    zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    ax.set_title('Confusion Matrix - F-score %s \n'% '{0:0.2f}'.format(fscore));
    ax.set_xlabel('Predicted Values') # \n
    ax.set_ylabel('Actual Values ');
    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False', 'True'])
    plt.savefig(os.path.join(paths.PROJECT_DIR, 'results/05Analysis/', 'confusion_matrix_%s.pdf'% model))
    plt.show()
    return
 
"""
This module extracts features from raw data and create a new dataframe that has new features and all the data

@author: Hung Nguyen
"""

import pandas as pd
import glob
import numpy as np
from operator import truediv
import os
import xml.etree.cElementTree as ET
import pickle
import plotly.express as px

def readcsv(the_path):
    """
    Read all the csv files and concatenate into a dataframe
    """

    path = the_path
    all_files = glob.glob(path + "/*.csv")
    li = []
    
    bigX, bigY, bigZ = [],[],[]
#     i = 0
    for filename in all_files:
#         i += 1
#         if i < 3:
            a_df = pd.read_csv(filename)
            li.append(a_df)
            a_df['count']=a_df.groupby('filename')['Name'].transform('count').values
            X,Y,Z = get_X_Y(a_df)
            bigX.append(X)
            bigY.append(Y)
            bigZ.append(Z)
    df = pd.concat(li, axis=0, ignore_index=True)
    

    return df, bigX, bigY, bigZ

def readxml(the_path):
    """
    Read all the XML files and concatenate into a dataframe
    This one is slower. Suggestion: Convert all the XMLs into CSVs
    """
    all_files = list(glob.iglob(os.path.join (the_path, "*.xml")))
    li = []
#     i = 0
    for fn in all_files:
#         i += 1
#         if i < 3:
            dfcols = ['Name','xmin', 'ymin', 'xmax','ymax', 'filename']

            mytree = ET.parse(fn)
            myroot = mytree.getroot()
            rows = myroot.findall('object')
            rows2 = myroot.findall('filename')

            for obj2 in rows2:
                name = obj2.text 
            xml_data = [ [obj[0].text, obj[4][0].text,obj[4][1].text,obj[4][2].text,obj[4][3].text, name] for obj in rows]

            df_xml = pd.DataFrame(xml_data, columns=dfcols)
            df_xml['count']=df_xml.groupby('filename')['Name'].transform('count').values
            
            li.append(df_xml)

    df = pd.concat(li, axis=0, ignore_index=True)
    return df

def get_X_Y(df):
    """
    Get big X and big Y
    """
    X = df.iloc[:,1:5].values
    Y = df.iloc[:,0].values
    Z = df.iloc[:,-1].values
    
    return X,Y,Z


df_classification, bigX_cl, bigY_cl, bigZ_cl = readcsv('csv_classification/')

df_detection, bigX_dt, bigY_dt, bigZ_dt = readcsv('csv_detection/')

def get_center(df, X_df, param):
    """
    Get the center of x or y
    This function serves as a sub-function for get_center_division()
    :param: Can be 'x' or 'y', specify which center to be calculated
    """
    if param == 'x': 
        list_x_center = []
        for i in range(0, len(df)):
            x_center = (X_df[i][2] - X_df[i][0])/2 # (xmax - xmin)/2
            list_x_center.append(x_center)
        return list_x_center
    elif param == 'y':
        list_y_center = []
        for i in range(0, len(df)):
            y_center = (X_df[i][3] - X_df[i][1])/2 # (ymax - ymin)/2
            list_y_center.append(y_center)
        return list_y_center
    else:
        return None
    
def get_center_division(df,X_df,center):
    """
    Get the x divided by width or y divided by height
    :param: Can be 'x' or 'y', specify which center to be calculated
    """
    if center == 'x':
        the_center = get_center(df,X_df,center)
        width = df['width'].values
        xDVwidth = list(map(truediv, the_center, width)) 

        return xDVwidth
    elif center == 'y':
        the_center = get_center(df,X_df,center)
        height = df['height'].values
        yDVheight = list(map(truediv, the_center, height)) 

        return yDVheight
    else:
        return None

def get_maxmin_subtract(df,X_df,param):
    """
    Get the center of x or y
    This function serves as a sub-function for get_subtracted_division()
    :param: Can be 'x' or 'y', specify which center to be calculated
    """
    if param == 'x':
        list_x_subtract = []
        for i in range(0, len(df)):
            list_x_subtract.append(X_df[i][2] - X_df[i][0]) # xmax - xmin
        return list_x_subtract
    elif param == 'y':
        list_y_subtract = []
        for i in range(0, len(df)):
            list_y_subtract.append(X_df[i][3] - X_df[i][1]) # ymax-ymin
        return list_y_subtract
    else:
        return None
    
def get_subtracted_division(df,X_df,center):
    """
    Get the x divided by width or y divided by height
    :param: Can be 'x' or 'y', specify which center to be calculated
    """
    if center == 'x':
        the_center = get_maxmin_subtract(df,X_df,center)
        width = df['width'].values
        xDVwidth = list(map(truediv, the_center, width)) 

        return xDVwidth
    elif center == 'y':
        the_center = get_maxmin_subtract(df,X_df,center)
        height = df['height'].values
        yDVheight = list(map(truediv, the_center, height)) 

        return yDVheight
    else:
        return None
    
X_classification,Y_classification,Z_classification = get_X_Y(df_classification)

X_detection,Y_detection,Z_detection = get_X_Y(df_detection)

def get_xy_coordinate_in_a_single_box(X_cl):
    """
    Get x coordinates and y coordinates and distinguish from each file
    """
    x = []
    y = []
    for i in range(0, len(X_cl)):
        x.append(X_cl[i][0])
        x.append(X_cl[i][2])
        y.append(X_cl[i][1])
        y.append(X_cl[i][3])
    c = 2
    fi_x= lambda x, c: [tuple(x[i:i+c]) for i in range(0, len(x), c)]
    fi_y= lambda y, c: [tuple(y[i:i+c]) for i in range(0, len(y), c)]
    x = fi_y(x,c)
    y = fi_y(y,c)
    return x,y

def get_xy_coordinate():
    """
    Get all the x and y coordinates to a big list
    """
    x = []
    y = []
    for i in range(0, len(bigX_cl)):
        xx,yy = get_xy_coordinate_in_a_single_box(bigX_cl[i])
        x.append(xx)
        y.append(yy)
    return x,y
x_coords,y_coords = get_xy_coordinate()

"""
This part finds the relationship of y-coordinates
"""
def get_over_lap(a,b):
    """
    Get the interval overlapping between two lists
    """
    return max(0, min(a[1],b[1])-max(a[0],b[0]))

def get_ovl(box, i):
    """
    Get the list of interval overlapping of the desired box and other boxes in y_coords
    """
    rs = []
    for j in range(len(y_coords[i])):
        rs.append(get_over_lap(box,y_coords[i][j]))
    return rs

def count_sameline(box, i):
    """
    Count the number of other boxes that are on the same line with the given box
    using a given interval
    """
    ovl = get_ovl(box, i)
    c = 0
    for i in ovl:
        if i >= 29:
            c+=1
    return c

def create_list_sameline():
    """
    Create a list of boxes that count the number of boxes on a same line
    """
    lst = []
    for i in range(0,len(y_coords)): # Loop over the 2D lists(files) inside // i = 0 - ...
        for j in range(0,len(y_coords[i])): # Loop over the 1D list in the 2D lists // j = 0 - len(y_coords[0]) = 0 - 9
            lst.append(count_sameline(y_coords[i][j], i))
    return lst
"""
This part finds the relationship of x-coordinates
"""
# Part 1
def get_boxes_on_row(box, i):
    """
    Given a single box, append to the list all the boxes that lie on a same line
    :param: box y-coordinates of a single box to determine whether they are on a same line
    :param: i the ith file
    """
    ovl = get_ovl(box, i)
    lst_bor = []
    time = 0
    for j in range(0, len(ovl)):
        if ovl[j] >= 29:
            lst_bor.append(x_coords[i][j])
    return lst_bor
    
def get_pos_in_row(box, i, j):
    """
    Given a single box, determine the position(index) of the box on the line
    """
    lst_bor = get_boxes_on_row(box, i)
    box = x_coords[i][j] # convert
    lst_sorted = lst_bor.copy()
    lst_sorted.sort()
    index = 0
    for boxes in lst_bor:
        if box == boxes:
            single_ix = lst_sorted.index(boxes)
            index += single_ix
    return index

def create_list_posline():
    """
    Final func:
        Create a list of positions on rows of each box
    """
    lst = []
    for i in range(0,len(y_coords)): # Loop over the 2D lists(files) inside // i = 0 - ...
        for j in range(0,len(y_coords[i])): # Loop over the 1D list in the 2D lists // j = 0 - len(y_coords[0]) = 0 - 9
            lst.append(get_pos_in_row(y_coords[i][j], i, j))
    return lst

# Part 2
def calculate_prev_dis(box,i, j):
    """
    Given a single box, get the distance between a given box to the previous box
    Constraint: if box is at the position index 0, then distance = 0
    """
    lst_bor = get_boxes_on_row(box, i)
    box = x_coords[i][j] # convert
    lst_sorted = lst_bor.copy()
    lst_sorted.sort()
    distance = 0
    for boxes in lst_bor:
        if box == boxes:
            single_ix = lst_sorted.index(boxes)
            if single_ix != 0:
                distance += lst_sorted[single_ix][0] - lst_sorted[single_ix -1][1] # xmin[current] - xmax[current-1]
            else:
                distance = 0
    return distance

def calculate_next_dis(box,i,j):
    """
    Given a single box, get the distance between a given box to the next box
    Constraint: if box is at the position index 0, then distance = 0
    """
    lst_bor = get_boxes_on_row(box, i)
    box = x_coords[i][j] # convert
    lst_sorted = lst_bor.copy()
    lst_sorted.sort()
    lst_index = []
    distance = 0
    for boxes in lst_bor:
        if box == boxes:
            single_ix = lst_sorted.index(boxes)
            if single_ix < len(lst_sorted) - 1:
                distance += lst_sorted[single_ix+1][0] - lst_sorted[single_ix][1] # xmin[current+1] - xmax[current]
            else:
                distance = 0
    return distance

def get_all_prev_dis():
    """
    Get all the distances of all boxes to the previous boxes
    """
    lst = []
    for i in range(0,len(y_coords)): # Loop over the 2D lists(files) inside // i = 0 - ...
        for j in range(0,len(y_coords[i])): # Loop over the 1D list in the 2D lists // j = 0 - len(y_coords[0]) = 0 - 9
            lst.append(calculate_prev_dis(y_coords[i][j], i, j))
    return lst

def get_all_next_dis():
    """
    Get all the distances of all boxes to the next boxes
    """
    lst = []
    for i in range(0,len(y_coords)): # Loop over the 2D lists(files) inside // i = 0 - ...
        for j in range(0,len(y_coords[i])): # Loop over the 1D list in the 2D lists // j = 0 - len(y_coords[0]) = 0 - 9
            lst.append(calculate_next_dis(y_coords[i][j], i, j))
    return lst

def test():
    print(get_boxes_on_row(y_coords[0][0], 0))
    a = get_boxes_on_row(y_coords[0][0], 0)
    a.sort()
    print(a)
    print(x_coords[0])
    print('true pos: ', get_pos_in_row(y_coords[0][0], 0,0)) # True position on a line in picture
    print(get_all_prev_dis())
    print(get_all_next_dis())
    
def get_features(df,X_df):
    """
    Get the features to feed ML
    """
    x_centerDVwidth = get_center_division(df, X_df, 'x')
    y_centerDVheight = get_center_division(df,X_df, 'y')

    x_subtractDVwidth = get_subtracted_division(df,X_df, 'x')
    y_subtractDVheight = get_subtracted_division(df,X_df, 'y')
    
    return x_centerDVwidth, y_centerDVheight, x_subtractDVwidth, y_subtractDVheight
def create_newdf(df, X_df):
    """
    Create a new dataframe that removes xmin ymin xmax ymax and has new features
    """
    x_centerDVwidth, y_centerDVheight, x_subtractDVwidth, y_subtractDVheight = get_features(df, X_df)
    
    df = df.assign(x_centerDVwidth =x_centerDVwidth )
    df = df.assign(y_centerDVheight =y_centerDVheight )
    df = df.assign(x_subtractDVwidth =x_subtractDVwidth)
    df = df.assign(y_subtractDVheight =y_subtractDVheight )

    
    df = df[['x_centerDVwidth','y_centerDVheight','x_subtractDVwidth','y_subtractDVheight','count', 'Name']]
    return df

new_df_cl = create_newdf(df_classification, X_classification)
new_df_cl = new_df_cl.assign(NumBoxInLine = create_list_sameline())
new_df_cl = new_df_cl.assign(PosBoxOnLine = create_list_posline())
new_df_cl = new_df_cl.assign(PrevDis = get_all_prev_dis())
new_df_cl = new_df_cl.assign(NextDis = get_all_next_dis())

new_df_dt = create_newdf(df_detection, X_detection)

def get_new_XY(df):
    X = df.iloc[:,:4].values
    Y = df.iloc[:,-1].values
    return X,Y

newX_cl,newY_cl = get_new_XY(new_df_cl)
newX_cl = newX_cl.tolist()

newX_dt,newY_dt = get_new_XY(new_df_dt)
newX_dt = newX_dt.tolist()

def get_detected_class(df_cl,df_dt, X_cl, X_dt, Y_cl, Y_dt):
    """
    Get the detected class from df2 with if equal coordinates
    :param: df1 the classification dataframe
    :param: df2 the detection dataframe
    :return: the list of featured_class taken from df2
    """
    result_featured = []
    pos = 0
    for i in range(0,len(X_cl)):
        if X_cl[i] in X_dt:
            position_of_X_dt = X_dt.index(X_cl[i]) # The position of the desired ROW in X_dt
            result_featured.append(Y_dt[position_of_X_dt]) # Getting the desired feature from the specific position
        else:
            result_featured.append("NaN")
    return result_featured

a = get_detected_class(new_df_cl,new_df_dt,newX_cl,newX_dt,newY_cl,newY_dt)

new_df_cl = new_df_cl.assign(FeaturedClass = a)

new_df_cl = new_df_cl[['Name','x_centerDVwidth','y_centerDVheight','x_subtractDVwidth','y_subtractDVheight','NumBoxInLine','PosBoxOnLine','count','PrevDis','NextDis','FeaturedClass']]

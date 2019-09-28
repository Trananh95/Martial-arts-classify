import numpy as np
import os


#config dimension of vector action:
Dimension = 14
windowsize1 = 100
windowsize2 = 50

def PathToAllFile(origin_dir):
    ALL_path = []
    child_dirs = os.listdir(origin_dir)
    for child in child_dirs:
        # print(origin_dir)
        # print(child)
        # print(origin_dir+"\\"+child)
        child_child_dirs = os.listdir(origin_dir+"/"+child)


        for child_child in child_child_dirs:
            ALL_path.append(origin_dir+"/"+child+"/"+child_child)
    return ALL_path
def dis_to_line(point,pline,point_in_line):
    vec=point_in_line-point
    cohuong=[vec[1]*pline[2]-pline[1]*vec[2],vec[2]*pline[0]-pline[2]*vec[0],vec[0]*pline[1]-pline[0]*vec[1]]
    lenght=np.sqrt((cohuong[0]**2+cohuong[1]**2+cohuong[2]**2)/(pline[0]**2+pline[1]**2+pline[2]**2))
    return lenght
def spine_line(point1,point2):
    return np.array([point2[0]-point1[0],point2[1]-point1[1],point2[2]-point1[2]])
def LoadData(path_to_file):
    X = np.zeros((1,14),dtype  = np.float32)
    Y = []
    spine_line_1 = []
    origin_coordinates = np.zeros(3)
    all_frames = open(path_to_file,"r").readlines()

    for i , frame in enumerate(all_frames):
        # X = np.zeros((1, 42), dtype=np.float32)
        # print(frame)
        y = int(all_frames[0].replace(",", ".").split("#")[61])

        Toado = np.zeros((1, 42), dtype=np.float32)
        frame = frame.replace(",", ".")
        coordinates = frame.split("#")
        X_upper = coordinates[1:4] + coordinates[13:34]
        origin_coordinates = coordinates[34:37]
        origin_coordinates = np.array(origin_coordinates).astype((np.float32))
        X_lower = coordinates[37:55]  # Doan giua 31:37 la ko can, doan sau la ban chan 55:61 cung khong can
        Toado = np.append(Toado, (np.array(X_upper + X_lower).astype(np.float32).reshape(1, 42)), axis=0)

        for i, x in enumerate(Toado):
            # print(x)
            if (i % 3 == 0):
                x = x - origin_coordinates[0]
            if (i % 3 == 1):
                x = x - origin_coordinates[1]
            if (i % 3 == 2):
                x = x - origin_coordinates[2]
            # print("gia tri sau khi doi la:")
            # print(x)
        Y.append(y)
        Toado = np.delete(Toado, 0, axis=0)
        distant = np.zeros((1, Dimension), dtype=np.float32)
        Toado = Toado.reshape(Dimension, 3)
        spine_line_1.append(spine_line(Toado[7], origin_coordinates))
        for i, toado in enumerate(Toado):
            dis = dis_to_line(toado, spine_line_1[0], origin_coordinates)
            distant[0][i] = dis
        # print(distant.shape)
        X = np.append(X, distant, axis=0)
    X = np.delete(X, 0, axis=0)

    # Y = np.delete(Y, 0,axis=0)
    # print(X)
    return X,Y
#case1: Chuyen doi kich thuoc cua cac window ve 100 frame(ma tran 2 chieu 100*14)
def NormalizeMaxtrixAction(matrix_of_action,matrix_lable):
    Y = []
    x = np.zeros((1, Dimension))
    avg = 1 if len(matrix_of_action) > windowsize1 else 0
    if avg == 0:
        for i in range(len(matrix_of_action), windowsize1):
            matrix_of_action = np.append(matrix_of_action, x, axis=0)
        X = matrix_of_action
    else:
        X = np.zeros((1, Dimension))
        frame_lists = np.array_split(matrix_of_action, windowsize1, axis=0)
        # print(frame_lists)
        for i, array in enumerate(frame_lists):
            if array.shape[0] > 1:
                array = ((array[-1] - array[0]) / (array.shape[0])).reshape(1, Dimension)
            else:
                array = array.reshape(1, Dimension)

            if (np.isnan(array[0][0])):
                pass
            else:
                X = np.append(X, array, axis=0)
        X = np.delete(X, 0, axis=0)
    for i in range(0, len(X)):
        Y.append(matrix_lable[0])
    if len(X) != len(Y):
        return " X lenght must equal to Y lenght, please check again"

    return X, Y
#case2: Chuyen doi kich thuoc cua cac window thanh 50 frame (ma tran 2 chieu 50*14)
# Neu so frame cua hanh dong < 50 thi them cac vevtor 0 co kich thuoc 1*14 vao sau cac frame
# Neu so frame cua hanh dong > 50 thi lay trung binh 50 frame trong tong so cac frame
def DownSizeAction(matrix_of_action,matric_label): #
    Y = []
    x = np.zeros((1,Dimension))
    avg =  1 if len(matrix_of_action)>windowsize2 else 0
    if avg ==0:
        for i in range(len(matrix_of_action),windowsize2):
            matrix_of_action = np.append(matrix_of_action, x, axis = 0)
        X = matrix_of_action
    else:
        X = np.zeros((1, Dimension))
        frame_lists = np.array_split(matrix_of_action, windowsize2, axis=0)
        # print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
        # print(frame_lists)
        for i, array in enumerate(frame_lists):
            if array.shape[0] > 1:
                array = ((array[-1] - array[0]) / (array.shape[0])).reshape(1, Dimension)
            else:
                if array.shape !=(0,Dimension):
                    array = array.reshape(1, Dimension)

            if (np.isnan(array[0][0])):
                pass
            else:
                X = np.append(X, array, axis=0)
        X = np.delete(X, 0, axis=0)
    for i in range (0,len(X)):
        Y.append(matric_label[0])
    if len(X)!=len(Y):
        return " X lenght must equal to Y lenght, please check again"

    return X,Y
def ConvertAllActionsTo3DMatrixCase1(list_of_paths):
    X = np.zeros((1,windowsize2,Dimension),dtype= np.float32)
    Y = []
    for j, path in enumerate(list_of_paths):
        matrix, label = LoadData(path)
        x,y = DownSizeAction(matrix,label)
        x = x.reshape((1,windowsize2,Dimension))
        # print(label)


        X=np.append(X,x,axis=0)
        Y.append(y[0])
    # if len(X) != len(Y):
    #     print("error")
    X = np.delete(X,0, axis=0)

    #print(X)
    Y = np.array(Y)
    Y = Y.reshape(len(Y),1)
    # y = y.view()

    # cong = 0
    # for hd in y:
    #     if hd == 1:
    #         cong = cong + 1
    # print(cong)
    return X,Y
list_file  = PathToAllFile("Kinect-dataset")

X,Y = ConvertAllActionsTo3DMatrixCase1(list_file)
# print(Y)
# print(X.shape,Y.shape)
# # print(Y)


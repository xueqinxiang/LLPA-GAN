import os
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import shutil

if __name__ == "__main__":
    data_path = '../../data/CC3M/raw_images/train/cc3m_test'
    data_path_image = '../../data/CC3M/raw_images/train/cc3m_image'
    data_path_txt = '../../data/CC3M/raw_images/train/cc3m_txt'
    data_path_pickle = '../../data/CC3M/raw_images/train/filenames.pickle'
    filelist = os.listdir(data_path)
    flienames = []
    total_num = len(filelist)
    for item in filelist:
        if item.endswith('.jpg'):
            src = os.path.join(os.path.abspath(data_path), item)
            dst = os.path.join(os.path.abspath(data_path), 'CC3M_train_' + item)
            filename = 'CC3M_train_' + item[:-4]
            flienames.append(filename)
            try:
                os.rename(src, dst)
                shutil.copy(dst, data_path_image)
                print('converting %s to %s ...' % (src, dst))
            except:
                continue

        if item.endswith('.txt'):
            src = os.path.join(os.path.abspath(data_path), item)
            dst = os.path.join(os.path.abspath(data_path), 'CC3M_train_' + item)
            try:
                os.rename(src, dst)
                shutil.copy(dst, data_path_txt)
                print('converting %s to %s ...' % (src, dst))
            except:
                continue

    with open(data_path_pickle, 'wb') as f:
        pickle.dump(flienames, f)
        f.close()

    if os.path.isfile(data_path_pickle):
            with open(data_path_pickle, 'rb') as f:
                filenames_test = pickle.load(f)
    print('total %d to rename & converted jpgs', total_num)


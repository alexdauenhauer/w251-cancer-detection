# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
from querymachine import QueryMachine
import matplotlib.pyplot as plt
import pydicom
import os
import pandas as pd
from tqdm import tqdm
# from sklearn.model_selection import train_test_split


# %%
qm = QueryMachine()
s = qm.getSeries()


# %%
savepath = '/data/CBIS-DDSM'
os.makedirs(savepath, exist_ok=True)
# b_dir = savepath + '/benign'
# m_dir = savepath + '/malignant'
# raw_dir = savepath + '/raw_dcm'
train_dir = savepath + '/train'
valid_dir = savepath + '/valid'
# os.makedirs(b_dir, exist_ok=True)
# os.makedirs(m_dir, exist_ok=True)
# os.makedirs(raw_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)
for pth in [train_dir, valid_dir]:
    os.makedirs(os.path.join(pth, 'benign'), exist_ok=True)
    os.makedirs(os.path.join(pth, 'malignant'), exist_ok=True)


# %%
calc = pd.read_csv('calc_case_description_train_set.csv')
mass = pd.read_csv('mass_case_description_train_set.csv')
labels_train = calc.append(mass, sort=False)
calc = pd.read_csv('calc_case_description_test_set.csv')
mass = pd.read_csv('mass_case_description_test_set.csv')
labels_test = calc.append(mass, sort=False)
print(labels_train.shape, labels_test.shape)


# %%
labels_train = labels_train.loc[:, [
    c for c in labels_train.columns if 'image file path' in c or 'pathology' in c]]
labels_test = labels_test.loc[:, [
    c for c in labels_test.columns if 'image file path' in c or 'pathology' in c]]


# %%
def getLabel(ds, labels_train, labels_test):
    train = True
    label = None
    sid = ds.SeriesInstanceUID
    pid = ds.PatientID
    for c in labels_train.columns[1:]:
        for i, item in enumerate(labels_train[c]):
            if sid in item.split('/') or pid in item.split('/'):
                label = labels_train.pathology.iloc[i]
    if not label:
        train = False
        for c in labels_test.columns[1:]:
            for i, item in enumerate(labels_test[c]):
                if sid in item.split('/') or pid in item.split('/'):
                    label = labels_test.pathology.iloc[i]
    return train, label


# %%
def downloadImages(ids, labels_train, labels_test, savepath):
    roi_counter = 0
    img_counter = 0
    for item in tqdm(ids):
        series_id = item['SeriesInstanceUID']
        if 'ROI' in item['SeriesDescription']:
            roi_counter += int(item['ImageCount'])
        else:
            img_counter += int(item['ImageCount'])
        qm.getSeriesImages(series_id, savepath=savepath)
        images = [f for f in os.listdir(savepath) if '.dcm' in f]
        for f in images:
            filepath = os.path.join(savepath, f)
            try:
                ds = pydicom.dcmread(filepath)
            except:
                continue
            os.remove(filepath)
            train, label = getLabel(ds, labels_train, labels_test)
            if not label:
                return ds
            if train:
                jpg_savepath = savepath + '/train'
            else:
                jpg_savepath = savepath + '/valid'
            jpg_savepath = jpg_savepath + '/' + label.split('_')[0].lower()
            jpg_filepath = os.path.join(jpg_savepath, f)
            jpg_filepath = jpg_filepath.replace('.dcm', '.jpg')
            ds.decompress()
            img = ds.pixel_array
            plt.imsave(jpg_filepath, ds.pixel_array,
                       vmin=0, vmax=2**16, format='jpg')
    return roi_counter, img_counter


# %%
roi_counter, img_counter = downloadImages(
    s.json(), labels_train, labels_test, savepath)


# %%
counter = 0
for dirname, subdirs, filenames in os.walk(savepath):
    for f in filenames:
        counter += 1
# %%
print("images saved: ", counter)
print("images counted: ", img_counter)
print("roi images counted: ", roi_counter)

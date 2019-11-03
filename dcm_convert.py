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
response = qm.getSeries()

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
labels_train['set'] = 'train'
labels_train.set
calc = pd.read_csv('calc_case_description_test_set.csv')
mass = pd.read_csv('mass_case_description_test_set.csv')
labels_test = calc.append(mass, sort=False)
labels_test['set'] = 'test'
print(labels_train.shape, labels_test.shape)
labels = labels_train.append(labels_test)
# labels.shape


# %%
labels.index = range(labels.shape[0])


# %%
cropped = [r['SeriesInstanceUID']
           for r in response.json() if 'cropped' in r['SeriesDescription']]
ids = [r['SeriesInstanceUID']
       for r in response.json() if 'full' in r['SeriesDescription']]
ids.extend(cropped)
print(len(ids), len(response.json()))


# %%
labs = {}
for f in ids:
    for row in labels.loc[:, [c for c in labels.columns if 'file path' in c]].itertuples(index=True, name=None):
        for name in row[1:]:
            if f in name.split('/'):
                label = labels.loc[row[0], 'pathology'].split('_')[0].lower()
                img_set = labels.loc[row[0], 'set']
                labs[f] = (label, img_set)
print(len(labs))


# %%
train_counter = 0
test_counter = 0
for k, v in labs.items():
    if v[1] == 'train':
        train_counter += 1
    else:
        test_counter += 1
print(train_counter, test_counter, train_counter + test_counter)

# %%


def downloadImages(response_list, labs, savepath):
    roi_counter = 0
    img_counter = 0
    failed_reads = 0
    for item in tqdm(response_list):
        series_id = item['SeriesInstanceUID']
        if series_id not in labs.keys():
            continue
#         if 'ROI' in item['SeriesDescription']:
#             roi_counter += int(item['ImageCount'])
#         else:
#             img_counter += int(item['ImageCount'])
        label, img_set = labs[series_id]
        qm.getSeriesImages(series_id, savepath=savepath)
        images = [f for f in os.listdir(savepath) if '.dcm' in f]
        for f in images:
            filepath = os.path.join(savepath, f)
            ds = pydicom.dcmread(filepath)
#             try:
#                 ds = pydicom.dcmread(filepath)
#             except:
#                 continue
            os.remove(filepath)
#             if not label:
#                 return ds
            if img_set == 'train':
                jpg_savepath = savepath + '/train'
            else:
                jpg_savepath = savepath + '/valid'
            jpg_savepath = jpg_savepath + '/' + label
            jpg_name = series_id + '_' + f
            jpg_filepath = os.path.join(jpg_savepath, jpg_name)
            jpg_filepath = jpg_filepath.replace('.dcm', '.jpg')
            ds.decompress()
            img = ds.pixel_array
            plt.imsave(jpg_filepath, ds.pixel_array,
                       vmin=0, vmax=2**16, format='jpg')
#     return roi_counter, img_counter


# %%
# roi_counter, img_counter = downloadImages(s.json(), labels_train, labels_test, savepath)
downloadImages(response.json(), labs, savepath)


# %%


# %%


# %%


# %%


# %%
# def getLabel(ds, labels_train, labels_test):
#     train = True
#     label = None
#     sid = ds.SeriesInstanceUID
#     pid = ds.PatientID
#     for c in labels_train.columns[1:]:
#         for i, item in enumerate(labels_train[c]):
#             if sid in item.split('/') or pid in item.split('/'):
#                 label = labels_train.pathology.iloc[i]
#     if not label:
#         train = False
#         for c in labels_test.columns[1:]:
#             for i, item in enumerate(labels_test[c]):
#                 if sid in item.split('/') or pid in item.split('/'):
#                     label = labels_test.pathology.iloc[i]
#     return train, label


# # %%


# # %%
# s = set()
# for d in response.json():
#     s.add(d['SeriesInstanceUID'])
# len(s)


# # %%


# # %%
# s.json()[0]


# # %%
# rois = 0
# imgs = 0
# for d in response.json():
#     if 'ROI' in d['SeriesDescription']:
#         rois += d['ImageCount']
#     else:
#         imgs += d['ImageCount']
# rois, imgs


# # %%


# # %%


# # %%
# labels.columns


# # %%
# labels['image file path'].unique().shape


# # %%


# # %%
# labels_train = labels_train.loc[:, [
#     c for c in labels_train.columns if 'image file path' in c or 'pathology' in c]]
# labels_test = labels_test.loc[:, [
#     c for c in labels_test.columns if 'image file path' in c or 'pathology' in c]]
# labels = labels_train.append(labels_test)
# labels.shape


# # %%
# labels.head()


# # %%


# # %%
# D = []
# for d in os.listdir(raw_dir):
#     filename = os.path.join(raw_dir, d)
#     ds = pydicom.dcmread(filename)
#     D.append(ds)
# len(D)


# # %%
# ds = D[0]


# # %%
# ds


# # %%
# uid = '/'.join([ds.PatientID, ds.StudyInstanceUID, ds.SeriesInstanceUID])
# # uid = '/'.join([ds.PatientID, ds.SeriesInstanceUID, ds.StudyInstanceUID])
# uid


# # %%
# uid in labels['image file path']


# # %%
# uid in labels['cropped image file path']


# # %%
# uid in labels['ROI mask file path']


# # %%
# pids = [l.split('/')[0] for l in labels['ROI mask file path']]
# len(pids)


# # %%
# pids[0]


# # %%
# counter = 0
# for p in pids:
#     if ds.PatientID in p:
#         counter += 1
# counter


# # %%
# s = set()
# for ds in D:
#     s.add(ds.SeriesDescription)
# s


# # %%
# ds.PatientID


# # %%
# def getLabel(ds, labels_train, labels_test):
#     train = True
#     label = None
#     sid = ds.SeriesInstanceUID
#     pid = ds.PatientID
#     for c in labels_train.columns[1:]:
#         for i, item in enumerate(labels_train[c]):
#             if sid in item.split('/') or pid in item.split('/'):
#                 label = labels_train.pathology.iloc[i]
#     if not label:
#         train = False
#         for c in labels_test.columns[1:]:
#             for i, item in enumerate(labels_test[c]):
#                 if sid in item.split('/') or pid in item.split('/'):
#                     label = labels_test.pathology.iloc[i]
#     return train, label


# # %%
# def downloadImages(ids, labels_train, labels_test, savepath):
#     roi_counter = 0
#     img_counter = 0
#     for item in tqdm(ids):
#         series_id = item['SeriesInstanceUID']
#         if 'ROI' in item['SeriesDescription']:
#             roi_counter += int(item['ImageCount'])
#         else:
#             img_counter += int(item['ImageCount'])
#         qm.getSeriesImages(series_id, savepath=savepath)
#         images = [f for f in os.listdir(savepath) if '.dcm' in f]
#         for f in images:
#             filepath = os.path.join(savepath, f)
#             try:
#                 ds = pydicom.dcmread(filepath)
#             except:
#                 continue
#             os.remove(filepath)
#             train, label = getLabel(ds, labels_train, labels_test)
#             if not label:
#                 return ds
#             if train:
#                 jpg_savepath = savepath + '/train'
#             else:
#                 jpg_savepath = savepath + '/valid'
#             jpg_savepath = jpg_savepath + '/' + label.split('_')[0].lower()
#             jpg_filepath = os.path.join(jpg_savepath, f)
#             jpg_filepath = jpg_filepath.replace('.dcm', '.jpg')
#             ds.decompress()
#             img = ds.pixel_array
#             plt.imsave(jpg_filepath, ds.pixel_array,
#                        vmin=0, vmax=2**16, format='jpg')
#     return roi_counter, img_counter


# # %%
# roi_counter, img_counter = downloadImages(
#     s.json(), labels_train, labels_test, savepath)


# # %%
# counter = 0
# for dirname, subdirs, filenames in os.walk(savepath):
#     for f in filenames:
#         counter += 1
# counter


# # %%
# img_counter


# # %%
# counter


# # %%
# ds.PatientID


# # %%
# downloadImages(s.json(), labels_train, labels_test, savepath)


# # %%
# counter = 0
# for dirname, subdirs, filenames in os.walk(savepath):
#     for f in filenames:
#         counter += 1
# print(counter)


# # %%
# counter = 0
# for dirname, subdirs, filenames in os.walk(savepath):
#     for f in filenames:
#         counter += 1
# print(counter)


# # %%
# savepath


# # %%
# sid = ds.SeriesInstanceUID
# for c in labels.columns[1:]:
#     for i, item in enumerate(labels[c]):
#         if sid in item.split('/'):
#             label = labels.pathology.iloc[i]


# # %%
# label


# # %%
# labels.shape


# # %%
# sid = 'Mass-Test_P_01595_LEFT_CC_1'
# label = None
# for c in labels.columns[1:]:
#     for i, item in enumerate(labels[c]):
#         if sid in item.split('/'):
#             label = labels.pathology.iloc[i]
# print(label)


# # %%
# downloadImages(test[:10], labels, savepath, train=False)


# # %%


# # %%
# calc = pd.read_csv('calc_case_description_train_set.csv')
# print(calc.shape)
# mass = pd.read_csv('mass_case_description_train_set.csv')
# df = calc.append(mass)
# print(df.shape)


# # %%
# dcm_files = [os.path.join(savepath, f)
#              for f in os.listdir(savepath) if '.dcm' in f]
# dcm_files


# # %%
# img_files = df.loc[:, [
#     c for c in calc.columns if 'image file path' in c or 'pathology' in c]]
# img_files.head()


# # %%
# ds.SeriesInstanceUID


# # %%
# for f in dcm_files:
#     ds = pydicom.dcmread(f)
#     print(ds.SeriesDescription)


# # %%
# for f in dcm_files:
#     ds = pydicom.dcmread(f)
#     sop = ds.SeriesInstanceUID
#     for c in img_files.columns:
#         for i, item in enumerate(img_files[c]):
#             if sop in item.split('/'):
#                 print(f, i, item, c, '\n')


# # %%
# for item in img_files.iloc[1591, :]:

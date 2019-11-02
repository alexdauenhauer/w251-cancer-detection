# %%
import copy
import io
import json
import os
import pickle
import zipfile
import requests

# %%


class QueryMachine(object):

    def __init__(self):
        self.base_url = 'https://services.cancerimagingarchive.net/services/v3/TCIA/query/'
        self.headers = {'api_key': 'asdlkfasdfasasdlfjasdlkfj'}

    def getSeries(self):
        query = 'getSeries'
        url = self.base_url + query
        params = {'Collection': 'CBIS-DDSM'}
        response = requests.get(url, headers=self.headers, params=params)
        return response

    def getSeriesImages(self, SeriesInstanceUID, savepath=None):
        query = 'getImage'
        url = self.base_url + query
        params = {'SeriesInstanceUID': SeriesInstanceUID}
        response = requests.get(url, headers=self.headers, params=params)
        z = zipfile.ZipFile(io.BytesIO(response.content))
        if savepath:
            z.extractall(savepath)
        else:
            z.extractall()

    def getSOPInstanceUIDs(self, SeriesInstanceUID):
        query = 'getSOPInstanceUIDs'
        url = self.base_url + query
        params = {'SeriesInstanceUID': SeriesInstanceUID}
        response = requests.get(url, headers=self.headers, params=params)
        return response

    def getSingleImage(self, SeriesInstanceUID, SOPInstanceUID):
        query = 'getSingleImage'
        url = self.base_url + query
        params = {
            'SeriesInstanceUID': SeriesInstanceUID,
            'SOPInstanceUID': SOPInstanceUID
        }
        response = requests.get(url, headers=self.headers,
                                params=params, stream=False)
        return response


# # %%
# q = QueryMachine()

# # %%
# r = q.getSeries()
# r.json()
# # %%
# r = q.getSOPInstanceUIDs(
#     '1.3.6.1.4.1.9590.100.1.2.193596412512123752718353896074113724101')
# r.json()

# # %%
# r = q.getSingleImage(
#     '1.3.6.1.4.1.9590.100.1.2.193596412512123752718353896074113724101',
#     '1.3.6.1.4.1.9590.100.1.2.346506859512074654709111389522842853221'
# )
# # plt.imshow(img.pixel_array)
# # %%
# r = q.getSeriesImages(
#     '1.3.6.1.4.1.9590.100.1.2.193596412512123752718353896074113724101', savepath='/home/alex')
# # %%
# impath = '/home/alex/1-296.dcm'
# ds = pydicom.dcmread(impath)

# # %%
# ds
# # %%
# ds.Rows
# # %%

# # %%
# print(r.encoding)

# z = zipfile.ZipFile(io.BytesIO(r.content))
# z.extractall()

# # %%
# i = Image.open(BytesIO(r.content))

# # %%
# type(r.content)

# # %%
# type(pickle.loads(r.content))

# # %%
# requests.request()

# # %%
# !pwd

from pyts.image import GramianAngularField

def gaf_transform(windows):
    gaf = GramianAngularField(method='summation')
    images = gaf.fit_transform(windows)
    return images.reshape(images.shape[0], -1)

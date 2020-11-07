# from open images, extract n images from a small set of classes plus n random images not from any of those classes


# test subjects ten images from each class are set aside as test subjects

# mean percent of the 10 nearest neighbors for each test subject that belong to the same class

# http://s3.amazonaws.com/open-images-dataset/validation/000595fe6fee6369.jpg


import pandas as pd

validation = pd.read_csv("/Users/racoon/Downloads/validation-annotations-bbox.csv")

print(validation.head(8))

validation[(validation['LabelName']=='/m/03k3r') & (validation["Confidence"] != 1)]
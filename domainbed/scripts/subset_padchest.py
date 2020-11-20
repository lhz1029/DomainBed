import pandas as pd


df = pd.read_csv('padchest/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv')
classes = ['atelectasis', 'cardiomegaly', 'consolidation', 'pulmonary edema', 'pleural effusion', 'pericardial effusion', 'normal', 'pneumonia', 'pneumothorax']
df_new = df[(df.Labels.dtype!=float)&df.Labels.str.contains('|'.join(classes))]
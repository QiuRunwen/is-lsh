# -*- coding: utf-8 -*-
"""
From Audobon Society Field Guide; 
mushrooms described in terms of physical characteristics;
classification: poisonous or edible
https://archive.ics.uci.edu/dataset/73/mushroom
@author: QiuRunwen
"""

import os
from zipfile import ZipFile
import pandas as pd
if __name__ == "__main__":
    import util # 单独运行本脚本用
else:
    from . import util  # 当作一个module被别人 import 时用 


def load_mushroom(*, data_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data"), drop_useless=True, num_sample=None, verbose=False):
    # 1. read/uncompress data

    headers = ['classes','cap-shape', 'cap-surface', 'cap-color', 'bruises?', 'odor', 
               'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 
               'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
               'stalk-color-above-ring','stalk-color-below-ring','veil-type',
               'veil-color', 'ring-number','ring-type','spore-print-color', 'population','habitat']

    with ZipFile(os.path.join(data_dir, 'mushroom', 'mushroom.zip')) as zf:
        with zf.open('agaricus-lepiota.data') as f: #
            df = pd.read_csv(f, header=None, names=headers, sep=',', na_values=["?"])
    
    # assert df.shape[1]==23
    # 2. convert numeric/categorical columns
    
    #  0. classes:                  edible=e, poisonous=p
    #  1. cap-shape:                bell=b,conical=c,convex=x,flat=f,
    #                               knobbed=k,sunken=s
    #  2. cap-surface:              fibrous=f,grooves=g,scaly=y,smooth=s
    #  3. cap-color:                brown=n,buff=b,cinnamon=c,gray=g,green=r,
    #                               pink=p,purple=u,red=e,white=w,yellow=y
    #  4. bruises?:                 bruises=t,no=f
    #  5. odor:                     almond=a,anise=l,creosote=c,fishy=y,foul=f,
    #                               musty=m,none=n,pungent=p,spicy=s
    #  6. gill-attachment:          attached=a,descending=d,free=f,notched=n
    #  7. gill-spacing:             close=c,crowded=w,distant=d
    #  8. gill-size:                broad=b,narrow=n
    #  9. gill-color:               black=k,brown=n,buff=b,chocolate=h,gray=g,
    #                               green=r,orange=o,pink=p,purple=u,red=e,
    #                               white=w,yellow=y
    # 10. stalk-shape:              enlarging=e,tapering=t
    # 11. stalk-root:               bulbous=b,club=c,cup=u,equal=e,
    #                               rhizomorphs=z,rooted=r,missing=?
    # 12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
    # 13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
    # 14. stalk-color-above-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o,
    #                               pink=p,red=e,white=w,yellow=y
    # 15. stalk-color-below-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o,
    #                               pink=p,red=e,white=w,yellow=y
    # 16. veil-type:                partial=p,universal=u
    # 17. veil-color:               brown=n,orange=o,white=w,yellow=y
    # 18. ring-number:              none=n,one=o,two=t
    # 19. ring-type:                cobwebby=c,evanescent=e,flaring=f,large=l,
    #                               none=n,pendant=p,sheathing=s,zone=z
    # 20. spore-print-color:        black=k,brown=n,buff=b,chocolate=h,green=r,
    #                               orange=o,purple=u,white=w,yellow=y
    # 21. population:               abundant=a,clustered=c,numerous=n,
    #                               scattered=s,several=v,solitary=y
    # 22. habitat:                  grasses=g,leaves=l,meadows=m,paths=p,
    #                               urban=u,waste=w,woods=d
    cat_cols = headers
    for col in cat_cols:
        if not isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype('category')
    
    # 3. simple feature extraction

    
    # 4. compute class label
    y_col = 'classes'
    df[y_col] = df[y_col].map({'e':1,'p':0})
    # assert df[y_col].notna().all()

    # 5. drop_useless
    if drop_useless:
        # a. manually remove useless columns
        
        # b. auto identified useless columns
        useless_cols_dict = util.find_useless_colum(df)
        df = util.drop_useless(df, useless_cols_dict, verbose=verbose)

    # 6. sampling by class
    if num_sample is not None:
        df = util.sampling_by_class(df, y_col, num_sample)

        # remove categorical cols with too few samples_per_cat
            
    return df, y_col

if __name__ == "__main__":
    df,y_col = load_mushroom(verbose=True)
    df.info()
# -*- coding: utf-8 -*-

# Tommy Gilmore & Joey Wysocki
# License Plate Detection
# 
import os
from os import path  
import kagglehub
from zipfile import ZipFile
import glob 
import kagglehub
import xml.etree.ElementTree as xmlet
import pandas as pd
from src.utilities import imshow_from_path, convert_xy_bounds_to_centered_xywh
from src.logger import logger as logging 
import shutil

# Download latest version

def download_dataset():
    assumed_datasetpath = path.join(os.path.expanduser("~"), ".cache/kagglehub/datasets/andrewmvd/car-plate-detection/versions/1")

    if path.exists(assumed_datasetpath): 
        logging.info(f"Dataset already exists. Skipping download.")
    else:
        logging.info("Downloading dataset")
        dataset_path = kagglehub.dataset_download("andrewmvd/car-plate-detection")
        if path.exists(dataset_path):
            logging.info(f"Dataset downloaded to {dataset_path}")
        else:
            logging.error("Dataset download failed. Please check your Kaggle API key or internet connection.")
            exit()
        assumed_datasetpath = dataset_path
    return assumed_datasetpath

def parse_dataset(datasetpath, load_existing_annotations=True):
    """Builds dataset from a dataset path. This function assumes that there is an annotations folder in the dataset path.
    This annotations folder shall contain the annotations for the dataset in an xml schema.

    Args:
        datasetpath (str, path-like): path to dataset
    """
    logging.info("Parsing dataset")
    # Unzip the dataset
    assert path.exists(datasetpath), "Dataset path does not exist"
    assert path.exists(path.join(datasetpath, "annotations")), "Dataset annotations"
    files = glob.glob(path.join(datasetpath, "annotations", "*.xml"))
    labels_dict = dict(filepath=[],imgpath=[], imgname = [], xmin=[],xmax=[],ymin=[],ymax=[], img_width=[], img_height=[])
    if load_existing_annotations and path.exists("data/labels.csv"):
        return pd.read_csv("data/labels.csv")
    else: 
        logging.info("Parsing dataset")
        # Parse the xml schema for the dataset
        for filename in files:

            info = xmlet.parse(filename)
            root = info.getroot()
            member_name   = root.find('filename')
            member_object = root.find('object')
            member_size   = root.find('size')
            labels_info   = member_object.find('bndbox')
            xmin = int(labels_info.find('xmin').text)
            xmax = int(labels_info.find('xmax').text)
            ymin = int(labels_info.find('ymin').text)
            ymax = int(labels_info.find('ymax').text)
            img_width = int(member_size.find('width').text)
            img_height = int(member_size.find('height').text)
            # Pull out image name, remove the .png extension
            imgname    = member_name.text.split(".")[0]

            labels_dict['filepath'].append(filename)
            labels_dict['imgpath'].append(filename.replace("annotations", "images").replace("xml", "png"))
            labels_dict['imgname'].append(imgname)
            labels_dict['xmin'].append(xmin)
            labels_dict['xmax'].append(xmax)
            labels_dict['ymin'].append(ymin)
            labels_dict['ymax'].append(ymax)
            labels_dict['img_width'].append(img_width)  
            labels_dict['img_height'].append(img_height)    

    return pd.DataFrame(labels_dict)


def UpdateDataFrameToYamlFormat(split_name, Input_Dataframe):
    # Define paths for labels and images
    labels_path = os.path.join(LPG.OUTPUTS_DIR, 'datasets', 'cars_license_plate_new', split_name, 'labels')
    images_path = os.path.join(LPG.OUTPUTS_DIR, 'datasets', 'cars_license_plate_new', split_name, 'images')

    # Create directories if they don't exist
    os.makedirs(labels_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)

    # Iterate over each row in the DataFrame
    for _, row in Input_Dataframe.iterrows():
        img_name = row['imgname'];
        img_extension = '.png'

        # Calculate YOLO format coordinates
        x_center = (row['xmin'] + row['xmax']) / 2 / row['img_width']
        y_center = (row['ymin'] + row['ymax']) / 2 / row['img_height']
        width = (row['xmax'] - row['xmin']) / row['img_width']
        height = (row['ymax'] - row['ymin']) / row['img_height']

        # Save labels in YOLO format
        label_path = os.path.join(labels_path, f'{img_name}.txt')
        with open(label_path, 'w') as file:
            file.write(f"0 {x_center:.4f} {y_center:.4f} {width:.4f} {height:.4f}\n")

        # Copy image to the images directory
        try:
            shutil.copy(row['imgpath'], os.path.join(images_path, img_name + img_extension))
        except Exception as e:
            logging.error(f"Failed to copy image {row['imgpath']} to {os.path.join(images_path, img_name + img_extension)}: {e}")

    print(f"Created '{images_path}' and '{labels_path}'")

def PullInPlateTruth(df):

    # Data from going through the dataset
    # N/A means that the plate was not visible or readable
    # Can be mutiple plates in one image

    data = """ 
    Car_Photo_Id Plate_Number
    0   KL01CA2555
    1   PG MN112
    10  TN 37 C5 2765
    100 WWS5550
    101 HR 26 BC 5514
    102 68 611 36
    103 N/A
    104 NL60 LXB
    105 SBV J966S
    105 SBA 1234A
    105 JPK 6546
    105 FBF 1234A
    106 N/A
    107 MH 20 BQ 20
    108 MH01AV8866
    109 CZ17 K0D
    11  WOR 516K
    110 FI
    111 MH 20 EE 7598
    112 SHAKNBK
    113 MPEACHW
    114 AP 29 BP 585
    115 M666 YOB
    116 MK 35 32
    117 RND4MG3921
    118 JA62 UAR
    119 N/A
    12  MH12BG7237
    120 B 2228HM
    121 N/A
    122 MANISA
    123 HR 26 BC 5514
    124 N/A
    125 RP 66887
    126 KA 05MG1909
    127 N/A
    128 0X65 AWD
    129 OPEC LOL
    13  BJR216
    130 HR 26 BR 9044
    131 AP 29 BP 585
    132 N/A
    133 1268
    134 L19 TLC
    135 M 17108
    135 FV 12332
    136 U 15474
    137 N/A
    138 TN 02 BL 9
    139 N/A
    14  ALR 486
    140 CH01AN0001
    141 N/A
    142 F1
    143 1F U3348
    143 9NZ1017
    144 PEN15
    145 CH01AN0001
    146 N/A
    147 KL01CA2555
    148 MH 01 BB 550
    149 7VIG263
    15  TN21BY0166
    150 DL8CX 4850
    151 DL3CBD5092
    152 DL49 AK49
    153 MH 20 BQ 20
    154 KA 09 MA 2662
    155 MCB 025
    156 PRIV ATE
    157 PTB557K
    158 695299
    159 DL7C N 5617
    16  N/A
    160 007PLATE.COM
    161 M666 YOB
    162 NH0BB 5992
    163 LR33 TEE
    164 GT
    165 TN21AU 1153
    166 IM A CAR
    167 15 LK 10898
    168 2222
    169 PRIV ATE
    17  YSX 213
    170 983
    171 BJR216
    172 UK 17 09
    173 WMY 9051
    174 N/A
    175 DL 1N 4268
    176 EV09FTW
    177 G00D
    178 522 92Z
    179 4SZW590
    18  M666 YOB
    180 MPEACHW
    181 JHI HAD
    182 V12 LAF
    183 BRIT 0001
    184 SWIFT
    185 ZB064 MF 90 00
    186 26 SPF 4
    187 ALR 486
    188 N/A
    189 CH01AN0001
    19  AFR 2011
    190 KL54A2670
    191 KA 03 AB 3380
    192 E4 GLE
    193 889 VSM
    194 TN 21 BZ 0768
    195 CIAZ
    196 KA 221
    197 LR33 TEE
    198 MH01AV8866
    199 MH01AE8017
    2   PRE NUP
    20  N/A
    200 KA 03 AB 3380
    201 LR33 TEE
    202 KWID
    203 CZ17 K0D
    204 N BYOND
    205 NOTACOP
    206 HR 26 BU 0380
    207 K817GR
    208 V8
    209 RX08880
    21  1RQ
    210 MH01AE8017
    211 F 65022
    212 KL 65H4383
    213 DID 2
    214 3NYB472
    215 MP G9 CU 5600
    215 MH 20 CH 9984
    215 TN 06 Q 7765
    215 TN 86 H 5
    215 TN 42 4002
    215 WB 34 AC 1080
    216 OPEC LOL
    217 ARF 2011
    218 2525
    219 KA 03 MG 2784
    22  YAMRAJ
    220 P055585
    221 N/A
    222 26 SPF 4
    223 VHA 634
    223 VHB 431 
    223 VHA 800
    224 K1 08 AU 777
    225 JA62 UAR
    226 TN 16 S 4523
    227 MCB 025
    228 N/A
    229 91245
    23  9214
    230 LR33 TEE
    231 GB18 TCE
    232 D 13
    233 UP16TC1366
    234 IM4U 555
    235 GJ03JL0126
    236 N/A
    237 P00S1LYR
    238 A D00008
    239 15 LK 10898
    24  HR 26 BC 5514
    240 1 CD AF
    241 3SAM123
    242 PRE NUP
    243 KA 03 MG 2784
    244 21 801 27
    245 HR 26 BR 9044
    246 TS 009 TC 268
    247 TN 37 C5 2765
    248 MH12DE1433
    249 DA 42 N 0852
    25  BYHM 136
    250 PG MN112
    251 H982 FKL
    252 DAN 54P
    253 N/A
    254 GOT2P00
    255 MH20EJ0364
    256 YD63LB
    257 OK LA
    258 KA 01 AC 5957
    259 MH 20 EE 7598
    26  M 17108
    26  FV 1232
    260 1M4U 5333
    260 30T7
    261 TS08ER1643
    262 1CYA H8N
    263 MH 02 CB 4545
    264 FAMU 2010
    265 M 771276
    266 N/A
    267 7222
    267 DA 722
    268 KUV
    269 FVE 5131
    27  DZ17 YXR
    270 PG18299
    271 YES BOSS
    272 RX61 GDU
    273 BRIT 0001
    274 MANISA
    275 BATMAN
    276 CZ17 K0D
    277 N/A
    278 48654
    279 MMN 306
    28  AP 1 1AM 0 9 1
    280 N/A
    281 MH 01 DB 0001
    282 68 611 36
    283 N/A
    284 KA 01 MA 2662
    285 HR 26 BR 9044
    286 MIDLYPH
    287 DL7C N 5617
    288 BATMAN
    289 S32H
    29  9214
    290 S 0V8199
    291 KA 02 A 6579
    292 AK 01 88
    293 GB19 TCE
    294 01 CC 1A 0001
    295 64 31
    295 58 44
    296 145
    297 IM4U 555
    298 LAMB0
    299 SDN7484U
    3   DZ17 YXR
    30  SBA 1234A
    30  SBV 1966S
    30  JPK 6546
    30  FBF 1234A
    300 M 17108
    300 FV 1232
    301 G526 JHD
    302 KL 01 CC 50
    303 KL 54 A 2670
    304 HNYCHIL3
    305 SG0 S1U
    305 SGQ51 U
    306 LAWYER
    307 M 906090K
    308 HR 26 BC 5514
    309 EAB0001
    31  YD64LB
    310 4GETOIL
    311 TN21AT8349
    312 PU18 BES
    313 NBEYOND
    314 6PIV728
    315 J98257
    316 MH 02 BB 2
    316 MH 02 JA 2
    317 HR 26 BC 5514
    318 ALR 486
    319 FALLYOU
    32  MANISA
    320 KA 09 MA 2662
    321 LOL OIL
    322 700 V
    323 TN99F2378
    324 V12LAF
    325 N/A
    326 DL8CX 4850
    327 PG MN112
    328 IM4U 555
    329 16M
    33  0X65 AWD
    330 MG 6800
    330 MR 6751
    331 HR 26 CB 1900
    332 DZ17 YXR
    333 MH 20EJ0364
    334 E80LA
    335 NOTACOP
    336 DL8CX 4850
    337 N/A
    338 N/A
    339 007PLATE.COM
    34  DL7C N 5617
    340 VX54FVL
    341 TS009 TC 298
    342 MANISA
    343 SKIP GAS
    344 GOOGLE
    345 0MG M00V
    346 KA 03 AB 3380
    347 N/A
    348 EAB0001
    349 PY56 BXU
    35  AK 01 88
    350 B 10612
    351 CB7 R605
    352 N/A
    353 KA 03 AB 3380
    354 N/A
    355 BKTP 665
    356 BSMART2
    357 AK 01 88
    358 PZ62FDX
    358 PZ62FDZ
    359 R362GGL
    36  MIDLYPH
    360 DX 103
    361 MPEACHW
    362 N/A
    363 WIW 123
    364 DL3 CBD 5092
    365 TN 21 BZ 0768
    366 MH 20 BQ 20
    367 GOOGLE
    368 D13
    369 15 LK 10898
    37  FAST
    370 CH01AN0001
    371 e20
    372 HR26CE1485
    373 TN 21 BC 6225
    374 S7 JDW
    375 N/A
    376 B 2228 HM
    377 BYHM 136
    378 4141
    379 MH14 GN 9293
    38  OMG BCKY
    380 W 718 AX
    381 20 89563
    382 GRAIG
    383 YASSCZP
    384 DX 103
    385 EV09FTW
    386 AB 44887
    387 HR 26 AZ 5927
    388 MR 32
    389 AB 44887
    39  MH15BD8877
    390 5H798 
    390 5H799
    391 BKWL 324
    392 DZ G49
    393 M 771276
    394 MH 12 MR 0613
    395 1CYA H8N
    396 REAP3R
    397 9214
    398 MMN 306
    399 0X65 AWB
    4   PU18 BES
    40  M 17108
    40  FV 1232
    400 889 VSM
    401 HR 26 BC 5514
    402 SGQ 51 U
    402 SGQ 51 U
    403 066 RAF
    403 067 RAF
    404 BMW
    405 MH 14 EC 3587
    406 1268
    407 PETES
    408 N/A
    409 RK 977AF
    41  YNTZDBG
    410 EWW POOP
    411 VIPER
    412 LR33 TEE
    413 64 31
    413 58 44
    414 DRUNK
    415 MH 12 NE 8922
    416 172 TMJ
    417 BAD 231
    418 0X65 AWD
    419 MH14 GN 9239
    42  BYHM 136
    420 SKIP GAS
    421 N/A
    422 MH20EE7596
    423 P3RV P
    424 KA 03 MG 2784
    425 G526 JHD
    426 GRAIG
    427 01 CC 1A 0001
    428 DZ17 YXR
    429 KAISER
    43  MH 20 BQ 20
    430 BCG 986
    431 KL BOSS
    432 DL49 AK49
    44  DL3CBF3907
    44  DL2CAG0786
    45  IMGROOT
    46  MH14 GN 9239
    47  5H798
    47  5H799
    48  ALR 486
    49  304 61C
    5   N/A
    50  PU18 BES
    51  YES BOSS
    52  MH15BD8877
    53  CH10 0SE
    54  LTM 378
    55  9214
    56  KL BOSS
    57  BKWL 324
    58  MH 01 DB 0001
    59  MH 46 P 1661
    6   802 11N
    60  0S 802 HN
    61  MMN 306
    62  WATT UP
    63  BIG TEAM
    64  TAXI
    65  0R INNIE
    66  9214
    67  MH 14 BN 7077
    68  BJR216
    69  KWID
    7   YSX 213
    70  A156
    71  LNIJGJZ
    72  DL7C N 5617
    73  TN99F2378
    74  CZ17 K0D
    75  N/A
    76  MH 31 EA 1RSS
    77  HR 26CU6799
    78  KL54A2670
    79  4GETOIL
    8   G526 JHD
    80  BAD 231
    81  ROY 9
    82  UP16TC1366
    83  AU 001
    84  HR 26 BC 5514
    85  VU63FTY
    86  MH14DX9937
    87  TSLA S1
    88  2151
    89  N/A
    9   MH01AV8866
    90  N/A
    91  304 61C
    92  A D00008
    93  N/A
    94  NBEYOND
    95  A D00008
    96  PG MN112
    97  KA05MG1909
    98  YWORRY
    99  KA 09 MA 2662
    """
    # Process the raw data into a list of tuples
    lines = data.strip().split("\n")
    processed_data = [line.split(maxsplit=1) for line in lines]

    # Convert to DataFrame
    new_df = pd.DataFrame(processed_data[1:], columns=processed_data[0])

    # Handle duplicates in the index column
    new_df["Car_Photo_Id"] = new_df["Car_Photo_Id"].astype(str)
    new_df["imgname"] = "Cars" + new_df["Car_Photo_Id"]
    new_df["Plate_Number"] = new_df["Plate_Number"].astype(str)

    # Since we can have multiple plates in the same image, need to do an outer join
    # Display the DataFrame
    merged_df = pd.merge(df, new_df, on="imgname", how="outer")
    return merged_df


def createYamlFormattedData(train_df, val_df, test_df, output_dir, save=True):
    """Creates the YOLO format for the data and saves the labels and images to the appropriate directories

    Args:
        train_df (pd.DataFrame): Training data containing image paths and bounding box coordinates
        val_df (pd.DataFrame): Validation data containing image paths and bounding box coordinates
        test_df (pd.DataFrame): Testing dataframe
        output_dir (str, path-like): Output directory
    
    Returns:
        str: Path to the datasets.yaml file
    """
    new_train_df = updateDataFrameToYamlFormat('train', train_df, output_dir=output_dir, save=save)
    new_val_df = updateDataFrameToYamlFormat('val', val_df, output_dir=output_dir, save=save)
    new_test_df = updateDataFrameToYamlFormat('test', test_df, output_dir=output_dir, save=save)

    datasets_yaml = f'''
    path: {output_dir}

    train: train/images
    val: val/images
    test: test/images

    # number of classes
    nc: 1

    # class names
    names: ['license_plate']
    '''

    # Write the content to the datasets.yaml file
    output_file_name = path.join(output_dir, 'datasets.yaml')
    if save:
        with open(output_file_name, 'w') as file:
            logging.info(f"Writing datasets.yaml to {path.join(output_dir, 'datasets.yaml')}")
            file.write(datasets_yaml)
    
    return output_file_name, new_train_df, new_val_df, new_test_df
    
def updateDataFrameToYamlFormat(split_name, df, output_dir, save=True):
    """Converts a DataFrame to the YOLO format and saves the labels and images to the appropriate directories

    Args:
        split_name (str): Name of the split (train, test, validation)
        output_dir (pd.DataFrame): DataFrame containing image paths and bounding box coordinates
        output_dir (str): Output directory to save the labels and images
        save (bool): Whether to save the labels and images to the directories
    """
    # Define paths for labels and images
    labels_path = os.path.join(output_dir, split_name, 'labels') 
    images_path = os.path.join(output_dir, split_name, 'images')
    output_df = df.copy()
    if save:
        # Create directories if they don't exist
        os.makedirs(labels_path, exist_ok=True)
        os.makedirs(images_path, exist_ok=True)

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        img_name = row['imgname']
        img_extension = '.png'

        # Calculate YOLO format coordinates, which are normalized and based on box center and width/height
        row_df = convert_xy_bounds_to_centered_xywh(row)
        x_center = row_df['x_center']
        y_center = row_df['y_center']
        width = row_df['width']
        height = row_df['height']
        output_df.loc[output_df['imgname'] == img_name, 'x_center'] = x_center
        output_df.loc[output_df['imgname'] == img_name, 'y_center'] = y_center
        output_df.loc[output_df['imgname'] == img_name, 'width'] = width
        output_df.loc[output_df['imgname'] == img_name, 'height'] = height

        # Save labels in YOLO format
        label_path = os.path.join(labels_path, f'{img_name}.txt')
        output_df.loc[output_df['imgname'] == img_name, 'label_path'] = label_path
        if save:
            with open(label_path, 'w') as file:
                file.write(f"0 {x_center:.4f} {y_center:.4f} {width:.4f} {height:.4f}\n")

            # Copy image to the images directory
            try:
                shutil.copy(row['imgpath'], os.path.join(images_path, img_name + img_extension))
            except Exception as e:
                logging.error(f"Failed to copy image {row['imgpath']} to {os.path.join(images_path, img_name + img_extension)}: {e}")
    if save:
        logging.info(f"Created '{images_path}' and '{labels_path}'")
    return output_df


if __name__ == "__main__":
    datasetpath = download_dataset()
    imshow_from_path(path.join(datasetpath, "images/1.png"))
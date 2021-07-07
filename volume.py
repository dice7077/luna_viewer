import pydicom
import os
import numpy as np
import math
import pandas as pd


# dcm_dirから3D画像をnp.arrayで取得
# 次元はz, y, xの順
def get_mra_volume(dcm_dir):
    print(dcm_dir)
    warning_flag = False  # 何かしらwarningが出た場合はTrueにして返す

    filenames = os.listdir(dcm_dir)
    filenames = [a for a in filenames if a[-4:] == '.dcm']
    filenames.sort()
    volume = []

    pos_dists = []
    flag = 0

    # 各スライスについて処理していく
    num_primary = 0
    j = 0
    for i, filename in enumerate(filenames):
        dcm_data = pydicom.dcmread(os.path.join(dcm_dir, filename))

        # MIPが入っていることがあるので除外する
        if ('ORIGINAL' not in dcm_data[0x0008, 0x0008].value):
            continue

        # 画像を１スライスずつ追加
        slice_img = dcm_data.pixel_array
        volume.append(slice_img)

        pos_tmp = dcm_data[0x0020, 0x0032].value  # Image Position
        pos_df_tmp = pd.DataFrame({'num': [j],
                                   # 'img': [slice_img],
                               'tmp0': [pos_tmp[0]],
                               'tmp1': [pos_tmp[1]],
                               'tmp2': [pos_tmp[2]]})
        j += 1

        # pixel spacingを取得する（どのスライスも同じ値のはずなので１回だけ取得する）
        # image position
        if flag == 0:
            pos_df = pos_df_tmp.copy()

            spacing_tmp1 = dcm_data[0x0028, 0x0030]  # Pixel Spacing
            y_spacing = float(spacing_tmp1[0])
            x_spacing = float(spacing_tmp1[1])
            pos_tmp1 = dcm_data[0x0020, 0x0032]  # Image Position
            flag = 1
        else:
            pos_df = pd.concat([pos_df,pos_df_tmp])

            spacing_tmp2 = dcm_data[0x0028, 0x0030]  # Pixel Spacing
            if spacing_tmp1 != spacing_tmp2:
                print('alert: spacing info differs between slices')
                print(dcm_dir)
                print(spacing_tmp1, 'and', spacing_tmp2)
                print()
                warning_flag = True
            pos_tmp2 = dcm_data[0x0020, 0x0032]  # Image Position
            pos_dists.append(math.sqrt(
                (pos_tmp1[0] - pos_tmp2[0]) ** 2 + (pos_tmp1[1] - pos_tmp2[1]) ** 2 + (pos_tmp1[2] - pos_tmp2[2]) ** 2))
            pos_tmp1 = pos_tmp2

    # Image Positionのz座標の順にsliceを並び替える
    pos_df = pos_df.sort_values('tmp2')
    volume = np.array(volume)
    volume = np.array(volume[pos_df['num'].values])
    # 画像の正規化
    volume = (volume - volume.min()) / (volume.max() - volume.min()) * 255
    volume = volume.astype(np.uint8)

    pos_df = pos_df.diff()
    pos_df['dist'] = (pos_df['tmp0']**2 + pos_df['tmp1']**2 + pos_df['tmp2']**2) ** 0.5

    # image_positionを使う方法
    slice_interval = np.median(np.array(pos_df['dist'].values[1:]))

    # spacingをnp.arrayとして保持
    spacing = np.array([slice_interval, y_spacing, x_spacing])

    # return volume, spacing, warning_flag
    return volume, spacing
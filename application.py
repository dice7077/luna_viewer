import plotly.express as px
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_vtk
import numpy as np
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import os
import pandas as pd
import math
import SimpleITK as sitk
from collections import OrderedDict
from scipy.ndimage.morphology import binary_erosion
# import cv2

from volume import get_mra_volume

try:
   # VTK 9+
   from vtkmodules.vtkImagingCore import vtkRTAnalyticSource
except:
  # VTK =< 8
  from vtk.vtkImagingCore import vtkRTAnalyticSource
  
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

df_orig = pd.read_csv('./data/eval_fppc1.00_sens0.8190_thresh0.74_analysis_space.csv')
test_sid_list = list(df_orig['series_id'].unique())
# trial_options = [{'label': i, 'value': i} for i in trial_list]
trial_options = []

category_df = pd.read_csv('./data/location_category.csv')
category_list = list(category_df.location_category.unique())
category_options = [{'label': i, 'value': i} for i in category_list]

class CTvolume:
    """
    name
    spacing
    D,H,W
    cz,cy,cx
    pixel_values
    pixel_values_masked
    """

    # datapathには.npyファイルか、dicom directoryを渡す
    def __init__(self, datapath='default', spacingpath=None, maskpath=None):
        # デフォルトでは真っ暗なvolumeを返す
        if datapath == 'default':
            self.name = 'default'
            self.pixel_values = np.zeros((100, 100, 100))
            self.spacing = [1, 1, 1]
        # 3Dのnp.arrayを読み込む
        elif datapath[-4:] == '.npy':
            self.name = datapath.split('/')[-1][:-4]
            tmp = np.load(datapath)
            if len(tmp.shape) == 4:
                self.pixel_values = tmp[0,:,:,:]
            else:
                self.pixel_values = tmp


            try:
                self.spacing = list(np.load(spacingpath))
            except:
                self.spacing = [1, 1, 1]
        # .mhdファイルを読みこむ
        elif datapath[-4:] == '.mhd':
            self.name = datapath.split('/')[-1][:-4]
            itkimage = sitk.ReadImage(datapath)
            img = sitk.GetArrayFromImage(itkimage)
            img = np.clip(img, -1200, 600)
            img = (img + 1200) / 1800 * 255
            self.pixel_values = img

            try:
                with open(datapath) as f:
                    contents = f.readlines()
                    meta = OrderedDict()
                    for content in contents:
                        content = content.replace('\n','').split('=')
                        meta[content[0].strip()] = content[1].strip()

                self.spacing = [float(a) for a in meta['ElementSpacing'].split()][::-1]
            except:
                self.spacing = [1, 1, 1]

        # その他はディレクトリ下にDICOM seriesが入っているパターンとみなす
        else:
            self.name = datapath.split('/')[-1]
            self.pixel_values, self.spacing = get_mra_volume(datapath)

        try:
            self.mask = np.load(maskpath)
        except:
            self.mask = np.ones_like(self.pixel_values).astype('bool')
        mask_tmp = binary_erosion(self.mask, iterations=1)
        self.pixel_values_masked = self.pixel_values * mask_tmp

        self.pixel_values_annot_axial = np.zeros((100,100,100,3))
        self.pixel_values_annot_coronal = np.zeros((100,100,100,3))
        self.pixel_values_annot_saggital = np.zeros((100,100,100,3))

        self.D, self.H, self.W = self.pixel_values.shape
        self.cz, self.cy, self.cx = self.D//2, self.H//2, self.W//2

    def set_annot(self, _df):
        img = self.pixel_values
        axial = np.array([img]*3).transpose(1,2,3,0).copy()
        saggital = axial.copy()
        coronal = axial.copy()

        def mark_point(axial, coronal, saggital, x, y, z, d, color):
            # color2 = tuple([min(255,round(a*1.5)) for a in color])

            for _z in range(z-1,z+2):
                slice_tmp = axial[_z,:,:,:].copy()
#                 cv2.circle(slice_tmp, (x, y), round(d*1.5), color, thickness=1)
                axial[_z,:,:,:] = slice_tmp

            for _y in range(y-1, y+2):
                slice_tmp = coronal[:,_y,:,:]
#                 cv2.circle(slice_tmp, (x, z), round(d*1.5), color, thickness=1)
                coronal[:, _y, :, :] = slice_tmp

            for _x in range(x-1, x+2):
                slice_tmp = saggital[:,:,_x,:].copy()
#                 cv2.circle(slice_tmp, (y, z), round(d*1.5), color, thickness=1)
                saggital[:,:,_x,:] = slice_tmp

            return axial, coronal, saggital


        # TP
        for i,row in _df[_df['TP']==1].iterrows():
            color = (0, 255, 0)
            z,y,x,d = row['z'],row['y'],row['x'],row['d']
            axial, coronal, saggital = mark_point(axial, coronal, saggital, x, y, z, d, color)

        # FN
        for i,row in _df[_df['FN']==1].iterrows():
            color = (255, 0, 0)
            z,y,x,d = row['z_annot'],row['y_annot'],row['x_annot'],row['d_annot']
            axial, coronal, saggital = mark_point(axial, coronal, saggital, x, y, z, d, color)

        # FP
        for i,row in _df[_df['FP']==1].iterrows():
            color = (255, 102, 0)
            z,y,x,d = row['z'],row['y'],row['x'],row['d']
            axial, coronal, saggital = mark_point(axial, coronal, saggital, x, y, z, d, color)

        # nonFP
        for i,row in _df[_df['nonFP']==1].iterrows():
            color = (165, 0, 255)
            z,y,x,d = row['z'],row['y'],row['x'],row['d']
            axial, coronal, saggital = mark_point(axial, coronal, saggital, x, y, z, d, color)

        self.pixel_values_annot_axial = axial
        self.pixel_values_annot_coronal = coronal
        self.pixel_values_annot_saggital = saggital


default_fig = px.imshow(np.zeros((100,100)), binary_string=True, zmin=0, zmax=255)
volume = CTvolume()

config = {
    'modeBarButtons': [
        ['drawline','drawrect'],
        ['pan2d'],
        ['zoomIn2d','zoomOut2d','resetScale2d']
    ],
    'scrollZoom': True,
    'displaylogo': False
}

application = dash.Dash(__name__)
server = application.server

application.layout = html.Div([
    html.H2('select annotation directory'),
    dcc.Input(
        id='annot_filepath',
        type='text',
        value='./data/tmp.csv',
        style={
            'width': '50%',
            'height': '50px',
            'font-size': '20px'
        }
    ),
    html.H2('select sample'),
    html.H4('enter data directory'),
    dcc.Input(
        id='data_dir',
        type='text',
        value='./data/volume',
        style={
            'width': '50%',
            'height': '50px',
            'font-size': '20px'
        }
    ),
    dcc.Dropdown(
        id='subdir_list',
        value=None,
        placeholder='select subdir',
        style={
            'width': '800px'
        }
    ),
    html.Br(),
    html.H4('filter sample by name'),
    dcc.Input(
        id='name_filter',
        type='text',
        value='',
        placeholder='type in part of name to filter',
        style={
            'width': '500px',
            'height': '30px',
            'font-size': '15px'
        }
    ),
    html.Br(),
    html.H4('select sample'),
    dcc.Dropdown(
        id='case_list',
        value=None,
        placeholder='type in name to filter',
        style={
            'width': '800px'
        }
    ),
    html.Br(),
    html.Button('load volume', id='load_button', n_clicks=0, style={'width':'180px', 'height':'30px', 'font-size':'15px'}),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),

    dcc.RadioItems(
        id='annotation_switch',
        options=[
            {'label': 'show_annotations', 'value': 'show'},
            {'label': 'hide_annotations', 'value': 'hide'}
        ],
        value='show'
    ),
    html.Br(),
    html.Br(),
    html.Div([
        html.Div(
            [
                dcc.Slider(id='axial_slider', min=0, max=500, step=1, value=50, tooltip={'always_visible':True}, marks={}),
                dcc.Graph(id='axial_slicer', figure=default_fig, style={'width':'900px', 'height':'900px'}, config={'modeBarButtons':[['pan2d']], 'scrollZoom': True}, hoverData=None),
            ],
            style={'float': 'left', 'width': '900px', 'height': '900px'}
        ),
        html.Div(
            [
                dcc.Slider(id='coronal_slider', min=0, max=500, step=1, value=50, tooltip={'always_visible':True}, marks={}),
                dcc.Graph(id='coronal_slicer', figure=default_fig, style={'width':'900px', 'height':'900px'}, config={'modeBarButtons':[['pan2d']], 'scrollZoom': True}, hoverData=None),
                dcc.Slider(id='saggital_slider', min=0, max=500, step=1, value=50, tooltip={'always_visible':True}, marks={}),
                dcc.Graph(id='saggital_slicer', figure=default_fig, style={'width':'900px', 'height':'900px'}, config={'modeBarButtons':[['pan2d']], 'scrollZoom': True}, hoverData=None),
                # dcc.Graph(id='saggital', figure=default_fig, style={'width': '1200px', 'height': '550px'},
                #           config=config),
            ],
            style={'float': 'left', 'height': '1500px'}
        ),
        html.Div(style={'clear': 'both'})
    ]),




    html.H2(id='mip_title', children='Partial MIP'),
    html.H3(children='Select keypoints to crop surrounding regions'),
    html.H4(children='select detection model'),
    dcc.RadioItems(
        id='spacing_switch',
        options=[
            {'label': 'pixel', 'value': 'pixel'},
            {'label': 'mm', 'value': 'mm'}
        ],
        value='pixel'
    ),
    dcc.Dropdown(
        id='trial_name',
        options=trial_options,
        value=None,
        style={
            'width': '800px'
        }
    ),
    html.H5('unlabeled annotations'),
    dcc.RadioItems(
        id='tpfnfp',
        options=[
            {'label': 'None', 'value': 'None'}
        ],
        value='None',
        labelStyle={'display': 'block'}
    ),
    html.Div(
        [
            html.H3(children='volume is currently cropped at the following coordinates'),
            html.Div([
                'xrange: ',
                dcc.Input(id='xmin', value=0, type='number'),
                dcc.Input(id='xmax', value=volume.W, type='number')
                # dcc.Input(id='xmin', value=125, type='number'),
                # dcc.Input(id='xmax', value=204, type='number')
            ]),
            html.Div([
                'yrange: ',
                dcc.Input(id='ymin', value=0, type='number'),
                dcc.Input(id='ymax', value=volume.H, type='number')
                # dcc.Input(id='ymin', value=259, type='number'),
                # dcc.Input(id='ymax', value=331, type='number')
            ]),
            html.Div([
                'zrange: ',
                dcc.Input(id='zmin', value=0, type='number'),
                dcc.Input(id='zmax', value=volume.D, type='number')
                # dcc.Input(id='zmin', value=61, type='number'),
                # dcc.Input(id='zmax', value=97, type='number')
            ]),
            html.Button('RESET', id='reset_button', n_clicks=0),
            html.Br(),
            html.Br(),
            html.Div('draw rectangle in axial, coronal, or saggital MIP images to define ROI for partial MIPs and volume rendering'),
            html.Div('please select ROI to be below 2,000,000 in order to view VR'),
            html.Div('scroll mouse to zoom partial MIP images'),
            html.Div('select pan to move the images around'),
        ],
        style={
            'width': '1000px',
            'float': 'left'
        }
    ),
    html.Div(
        [
            html.H3(id='selected_annot', children='Annotation for ...'),
            html.Div([
                'location category: ',
                dcc.Dropdown(
                    id='location_category',
                    options=category_options,
                    value=None,
                    style={
                        'width': '400px'
                    }
                ),
            ]),
            html.Div([
                'location subcategory: ',
                dcc.Dropdown(
                    id='location_subcategory',
                    options=[],
                    value=None,
                    style={
                        'width': '400px'
                    }
                ),
            ]),
            html.Div([
                'location free text: ',
                dcc.Input(id='location_freetext', value='', type='text', style={'width':'800px', 'height':'30px'}),
            ]),
            html.Div([
                'aneurysm size (mm): ',
                dcc.Input(id='aneurysm_size', value=-1, type='number'),
            ]),
            html.Div([
                'risk score: ',
                dcc.Dropdown(
                    id='risk',
                    options=[{'label': i, 'value': i} for i in range(1,5)] + [{'label': 'not aneurysm', 'value': 'na'}],
                    value=None,
                    style={
                        'width': '400px'
                    }
                ),
            ]),
            html.Div([
                'confidence score: ',
                dcc.Dropdown(
                    id='confidence',
                    options=[{'label': i, 'value': i} for i in range(5)],
                    value=None,
                    style={
                        'width': '400px'
                    }
                ),
            ]),
            html.Div([
                'comment: ',
                dcc.Input(id='comment', value='', type='text', style={'width':'800px', 'height':'30px'}),
            ]),
            html.Button('save annotation', id='save_annotation', n_clicks=0)
        ],
        style={
            'width': '1000px',
            'float': 'left'
        }
    ),
    html.Div(
        style={
            'clear':'both'
        }
    ),
    html.Div([
        html.Div(
            [
                dcc.Graph(id='axial', figure=default_fig, style={'width':'900px', 'height':'900px'}, config=config, hoverData=None),
                html.H5(id='axial_length', children='Draw line in axial MIP to measure length'),
                html.H5(id='coronal_length', children='Draw line in coronal MIP to measure length'),
                html.H5(id='saggital_length', children='Draw line in saggital MIP to measure length'),
                html.Button('reset lenght measurement', id='reset_length', n_clicks=0)
            ],
            style={'float':'left', 'width':'900px', 'height':'900px'}
        ),
        html.Div(
            [
                dcc.Graph(id='coronal', figure=default_fig, style={'width':'1200px', 'height':'550px'}, config=config),
                dcc.Graph(id='saggital', figure=default_fig, style={'width':'1200px', 'height':'550px'}, config=config),
            ],
            style={'float':'left', 'height':'1500px'}
        ),
        html.Div(style={'clear':'both'})
    ]),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.H2(id='show_volume', children='Show Volume'),
    html.Div(id='volume_render', style={'width': '70%', 'height': '1000px'})
])


def dim_outer_roi(img, x_range, y_range):
    img[:y_range[0],:] = img[:y_range[0],:].copy() // 3 + 127 * 2 // 3
    img[y_range[1]:,:] = img[y_range[1]:,:].copy() // 3 + 127 * 2 // 3
    img[y_range[0]:y_range[1],:x_range[0]] = img[y_range[0]:y_range[1],:x_range[0]].copy() // 3 + 127 * 2 // 3
    img[y_range[0]:y_range[1],x_range[1]:] = img[y_range[0]:y_range[1],x_range[1]:].copy() // 3 + 127 * 2 // 3
    return img


def draw_roi(img, x_range, y_range):
    img[y_range[0],x_range[0]:x_range[1]-1] = 255
    img[y_range[1]-1,x_range[0]:x_range[1]-1] = 255
    img[y_range[0]:y_range[1]-1,x_range[0]] = 255
    img[y_range[0]:y_range[1]-1,x_range[1]-1] = 255
    return img


def get_coord(relayout_data):
    shape = relayout_data['shapes'][-1]
    x0 = int(shape['x0'])
    x1 = int(shape['x1'])
    y0 = int(shape['y0'])
    y1 = int(shape['y1'])
    x0, x1 = min(x0,x1), max(x0,x1)
    y0, y1 = min(y0,y1), max(y0,y1)
    return x0,x1,y0,y1


def get_tpfnfp_options(df):
    options=[
        {'label': 'None', 'value': 'None'}
    ]

    # TP
    for i,row in df[df.TP==1].iterrows():
        l = f'TP z:{row["z_annot"]}_y:{row["y_annot"]}_x:{row["x_annot"]}_d:{row["d_annot"]}_location:{row["location_category"]}'
        v = f'TP_{row["z_annot"]}_{row["y_annot"]}_{row["x_annot"]}_d:{row["d_annot"]}'
        options.append({'label':l, 'value':v})

    # FN
    for i,row in df[df.FN==1].iterrows():
        l = f'FN z:{row["z_annot"]}_y:{row["y_annot"]}_x:{row["x_annot"]}_d:{row["d_annot"]}_location:{row["location_category"]}'
        v = f'FN_{row["z_annot"]}_{row["y_annot"]}_{row["x_annot"]}_d:{row["d_annot"]}'
        options.append({'label':l, 'value':v})

    # FP
    for i,row in df[df.FP==1].iterrows():
        l = f'FP z:{row["z"]}_y:{row["y"]}_x:{row["x"]}'
        v = f'FP_{row["z"]}_{row["y"]}_{row["x"]}'
        options.append({'label':l, 'value':v})

    # nonFP
    for i,row in df[df.nonFP==1].iterrows():
        l = f'nonFP z:{row["z"]}_y:{row["y"]}_x:{row["x"]}'
        v = f'nonFP_{row["z"]}_{row["y"]}_{row["x"]}'
        options.append({'label':l, 'value':v})

    return options


def get_tpfnfp_slider_marks(df):

    axial_marks = dict()
    coronal_marks = dict()
    saggital_marks = dict()
    j=1
    for i,row in df.iterrows():
        if row['TP'] == 1:
            # label = 'TP'
            color = '#00ff00'
        elif row['FN'] == 1:
            # label = 'FN'
            color = '#ff0000'
        elif row['FP'] == 1:
            # label = 'FP'
            color = '#ff6600'
        elif row['nonFP'] == 1:
            # label = 'nonFP'
            color = '#a500ff'
        else:
            continue
        label = ''
        axial_marks[int(row['z'])] = {'label': label, 'style': {'color': color}}
        coronal_marks[int(row['y'])] = {'label': label, 'style': {'color': color}}
        saggital_marks[int(row['x'])] = {'label': label, 'style': {'color': color}}
        j+=1

    return axial_marks, coronal_marks, saggital_marks


@application.callback(
    Output('axial_slicer', 'figure'),
    Input('load_button', 'n_clicks'),
    Input('axial_slider', 'value'),
    Input('annotation_switch', 'value'),
    Input('mip_title', 'children'),
)
def update_axial_slicer(load_clks, z_coord, annotation_switch, title):
    if annotation_switch == 'hide':
        img = volume.pixel_values[z_coord,:,:]
    elif annotation_switch == 'show':
        img = volume.pixel_values_annot_axial[z_coord,:,:,:]
    fig_axial_slicer = px.imshow(img, binary_string=True, zmin=0, zmax=255)
    fig_axial_slicer.update_layout(dragmode="pan", width=900, hovermode=False)
    return fig_axial_slicer


@application.callback(
    Output('coronal_slicer', 'figure'),
    Input('load_button', 'n_clicks'),
    Input('coronal_slider', 'value'),
    Input('annotation_switch', 'value'),
    Input('mip_title', 'children'),
)
def update_coronal_slicer(load_clks, y_coord, annotation_switch, title):
    if annotation_switch == 'hide':
        img = volume.pixel_values[:,y_coord,:]
    if annotation_switch == 'show':
        img = volume.pixel_values_annot_coronal[:,y_coord,:,:]
    fig_coronal_slicer = px.imshow(img, binary_string=True, zmin=0, zmax=255)
    fig_coronal_slicer.update_layout(dragmode="pan", height=650, hovermode=False, yaxis=dict(autorange=True))
    return fig_coronal_slicer


@application.callback(
    Output('saggital_slicer', 'figure'),
    Input('load_button', 'n_clicks'),
    Input('saggital_slider', 'value'),
    Input('annotation_switch', 'value'),
    Input('mip_title', 'children'),
)
def update_saggital_slicer(load_clks, x_coord, annotation_switch, title):
    if annotation_switch == 'hide':
        img = volume.pixel_values[:,:,x_coord]
    if annotation_switch == 'show':
        img = volume.pixel_values_annot_saggital[:,:,x_coord,:]
    fig_saggital_slicer = px.imshow(img, binary_string=True, zmin=0, zmax=255)
    fig_saggital_slicer.update_layout(dragmode="pan", height=650, hovermode=False, yaxis=dict(autorange=True))
    return fig_saggital_slicer


# data directory選択の補助用にサブディレクトリのリストを表示する
@application.callback(
    Output('subdir_list', 'options'),
    Input('data_dir', 'value')
)
def get_subdir_list(data_dir):
    data_list = os.listdir(data_dir)
    data_list = [{'label': i, 'value': i} for i in data_list]
    return data_list

@application.callback(
    Output('data_dir', 'value'),
    Input('subdir_list', 'value'),
    State('data_dir', 'value'),
    prevent_initial_call=True
)
def append_subdir(subdir, data_dir):
    return os.path.join(data_dir, subdir)

# data directoryの中からサンプル名をリストアップする
@application.callback(
    Output('case_list', 'options'),
    Input('data_dir', 'value'),
    Input('name_filter', 'value'),
    prevent_initial_call=False
)
def get_data_list(data_dir, name_filter):
    data_list = os.listdir(data_dir)
    data_list = [a for a in data_list if name_filter in a]

    data_list = [a for a in data_list if a[:-4] in test_sid_list] # result_fileに入っている症例だけを選ぶ

    data_list = [{'label': i, 'value': i} for i in data_list]

    return data_list


# アノテーションの保存
@application.callback(
    Output('save_annotation', 'n_clicks'),
    Input('save_annotation', 'n_clicks'),
    State('case_list', 'value'),
    State('trial_name', 'value'),
    State('tpfnfp', 'value'),
    State('location_category', 'value'),
    State('location_subcategory', 'value'),
    State('location_freetext', 'value'),
    State('aneurysm_size', 'value'),
    State('risk', 'value'),
    State('confidence', 'value'),
    State('comment', 'value'),
    State('annot_filepath', 'value'),
    prevent_initial_call=True
)
def save_annotation(save_clicks, case_name, trial_name, tpfnfp, location_category, location_subcategory,
                    location_freetext, aneurysm_size, risk, confidence, comment, annot_filepath):
    try:
        annot = pd.read_csv(annot_filepath)
    except:
        annot = pd.DataFrame({'trial_name':[], 'series_id':[], 'mark_id':[], 'mark_type':[], 'zc':[], 'yc':[], 'xc':[],
                              'location_category':[], 'location_subcategory':[], 'location_freetext':[], 'size':[], 'risk':[], 'confidence':[], 'comment':[]})

    annot_save = annot[(annot.series_id!=os.path.splitext(case_name)[0]) | (annot.trial_name!=trial_name) | (annot.mark_id!=tpfnfp)]
    tmp = tpfnfp.split('_')
    mark_type, zc, yc, xc = tmp[:4]
    annot_save = annot_save.append({'trial_name':trial_name, 'series_id':os.path.splitext(case_name)[0], 'mark_id':tpfnfp, 'mark_type':mark_type, 'zc':zc, 'yc':yc, 'xc':xc,
                                    'location_category':location_category, 'location_subcategory':location_subcategory, 'location_freetext':location_freetext, 'size':aneurysm_size, 'risk':risk,
                                    'confidence':confidence, 'comment':comment}, ignore_index=True)
    annot_save.sort_values(['trial_name', 'series_id', 'mark_id'], inplace=True)
    annot_save.to_csv(annot_filepath, index=None)

    return 0


# アノテーションの読み込みはここ
@application.callback(
    Output('selected_annot', 'children'),
    Output('location_category', 'value'),
    Output('location_subcategory', 'value'),
    Output('location_freetext', 'value'),
    Output('aneurysm_size', 'value'),
    Output('risk', 'value'),
    Output('confidence', 'value'),
    Output('comment', 'value'),
    Output('location_subcategory', 'options'),
    Input('tpfnfp', 'value'),
    Input('location_category', 'value'),
    State('location_subcategory', 'value'),
    State('location_freetext', 'value'),
    State('aneurysm_size', 'value'),
    State('risk', 'value'),
    State('confidence', 'value'),
    State('comment', 'value'),
    State('case_list', 'value'),
    State('annot_filepath', 'value'),
    prevent_initial_call=True
)
def read_annotation_labels(tpfnfp, location_category, location_subcategory, location_freetext, aneurysm_size, risk, confidence, comment, case_name, annot_filepath):
    ctx = dash.callback_context

    # location_categoryの切り替えでトリガーされた場合は、初期化をスキップする
    if ctx.triggered[0]['prop_id'] != 'location_category.value':
        location_category = ''
        location_subcategory = ''
        location_freetext = ''
        aneurysm_size = -1
        risk = 'blank'
        confidence = 'blank'
        comment = ''

        if tpfnfp is not None:
            # アノテーションを読み込む
            try:
                annot = pd.read_csv(annot_filepath)
                annot = annot[annot['series_id']==os.path.splitext(case_name)[0]]
                annot = annot[annot['mark_id']==tpfnfp]
                location_category = annot.location_category.values[0]
                location_subcategory = annot.location_subcategory.values[0]
                location_freetext = annot.location_freetext.values[0]
                aneurysm_size = annot['size'].values[0]
                risk = annot.risk.values[0]
                confidence = annot.confidence.values[0]
                comment = annot.comment.values[0]
            except:
                pass
    else:
        location_subcategory = None


    # location_categoryが選択済みの場合は、location subcategoryの選択範囲を変更する
    subcat_options = []
    if location_category!='':
        subcat_list = list(category_df[category_df.location_category==location_category].location_subcategory.values)
        subcat_options = [{'label': i, 'value': i} for i in subcat_list]

    try:
        if math.isnan(comment):
            comment = ''
    except:
        pass

    return tpfnfp, location_category, location_subcategory, location_freetext, aneurysm_size, risk, confidence, comment, subcat_options


# MIP表示の変更を伴う処理はすべてここ
@application.callback(
    Output('mip_title', 'children'),
    Output('axial', 'figure'),
    Output('coronal', 'figure'),
    Output('saggital', 'figure'),
    Output('axial', 'relayoutData'),
    Output('coronal', 'relayoutData'),
    Output('saggital', 'relayoutData'),
    Output('xmin', 'value'),
    Output('xmax', 'value'),
    Output('ymin', 'value'),
    Output('ymax', 'value'),
    Output('zmin', 'value'),
    Output('zmax', 'value'),
    Output('axial_slider', 'max'),
    Output('axial_slider', 'value'),
    Output('axial_slider', 'marks'),
    Output('coronal_slider', 'max'),
    Output('coronal_slider', 'value'),
    Output('coronal_slider', 'marks'),
    Output('saggital_slider', 'max'),
    Output('saggital_slider', 'value'),
    Output('saggital_slider', 'marks'),
    Input('load_button', 'n_clicks'),
    Input('reset_button', 'n_clicks'),
    Input('axial', 'relayoutData'),
    Input('coronal', 'relayoutData'),
    Input('saggital', 'relayoutData'),
    Input('xmin', 'value'),
    Input('xmax', 'value'),
    Input('ymin', 'value'),
    Input('ymax', 'value'),
    Input('zmin', 'value'),
    Input('zmax', 'value'),
    Input('spacing_switch', 'value'),
    Input('trial_name', 'value'),
    Input('tpfnfp', 'value'),
    Input('axial_slider', 'value'),
    Input('coronal_slider', 'value'),
    Input('saggital_slider', 'value'),
    State('data_dir', 'value'),
    State('case_list', 'value'),
    prevent_initial_call=True
)
def update_crop(load_clicks, reset_clicks, axial_data, coronal_data, saggital_data, xmin, xmax, ymin, ymax, zmin, zmax,
                spacing_switch, trial_name, tpfnfp, axial_slider_value, coronal_slider_value, saggital_slider_value, data_dir, case_name):
    global volume
    ctx = dash.callback_context

    if ctx.triggered[0]['prop_id'] in ['axial_slider.value','coronal_slider.value','saggital_slider.value']:
        raise PreventUpdate

    # loadボタンのクリックで関数がトリガーされていたら、画像を読み込む
    # loadボタンのクリックで関数がトリガーされていたら、検出結果のradio buttonを更新する
    # loadボタンのクリックで関数がトリガーされていたら、アノテーションを読み込む
    if ctx.triggered[0]['prop_id'] == 'load_button.n_clicks':
        datapath = os.path.join(data_dir, case_name)
        spacing_dir = ''
        mask_dir = '/mnt/project/brain/aneurysm/takamiya/temporary/data/dataset1_luna888/mask_spacing1mm_cropped_zyx'
        spacingpath = os.path.join(spacing_dir, case_name)
        maskpath = os.path.join(mask_dir, case_name)
        try:
            volume = CTvolume(datapath, spacingpath, maskpath)
            xmin = ymin = zmin = 0
            xmax, ymax, zmax = volume.W, volume.H, volume.D

            # TP,FN,FPのラジオボタンを作成する
            df = df_orig[(df_orig.series_id == volume.name)]
            options = get_tpfnfp_options(df)

            # アノテーション付きの画像を作成してvolumeオブジェクト内に保存する
            volume.set_annot(df)
        except:
            # パスが間違っていたら更新しない
            print('filepath does not exist')
            raise PreventUpdate

        axial_slider_value = volume.pixel_values.shape[0]//2
        coronal_slider_value = volume.pixel_values.shape[1]//2
        saggital_slider_value = volume.pixel_values.shape[2]//2

    # trial_nameが切り替えられたら、検出結果のradio buttonを更新する
    if ctx.triggered[0]['prop_id'] == 'trial_name.value':
        # TP,FN,FPのラジオボタンを作成する
        # df = df_orig[(df_orig.series_id == volume.name)]
        options = get_tpfnfp_options(df)

    # resetボタンのクリックで関数がトリガーされていたら、ROIの範囲を画像全体に戻す
    if ctx.triggered[0]['prop_id'] == 'reset_button.n_clicks':
        xmin = ymin = zmin = 0
        xmax, ymax, zmax = volume.W, volume.H, volume.D

    # MIP画像に四角を描いたことで関数がトリガーされていたら、ROIの座標を描いた四角に合わせて更新
    if ctx.triggered[0]['prop_id'] in ['axial.relayoutData', 'coronal.relayoutData', 'saggital.relayoutData']:
        if (axial_data is not None) and ('shapes' in axial_data) and (axial_data['shapes'][-1]['type']=='rect'):
            xmin,xmax,ymin,ymax = get_coord(axial_data)
            axial_data = None
        elif (coronal_data is not None) and ('shapes' in coronal_data) and (coronal_data['shapes'][-1]['type']=='rect'):
            xmin,xmax,zmin,zmax = get_coord(coronal_data)
            coronal_data = None
        elif (saggital_data is not None) and ('shapes' in saggital_data) and (saggital_data['shapes'][-1]['type']=='rect'):
            ymin,ymax,zmin,zmax = get_coord(saggital_data)
            saggital_data = None
        else:
            raise PreventUpdate

    # TP,FN,FPを選択したことで関数がトリガーされていたら、ROIの座標をそれに合わせて更新
    # TP,FN,FPを選択したことで関数がトリガーされていたら、アノテーションを読み込み
    # Noneが選択されていたら何もしない
    if ctx.triggered[0]['prop_id'] == 'tpfnfp.value':
        if (tpfnfp != None) & (tpfnfp != 'None'):
            tmp = tpfnfp.split('_')
            mark_type, zc, yc, xc = tmp[:4]
            if spacing_switch == 'mm':
                zc, yc, xc = float(zc)/volume.spacing[0], float(yc)/volume.spacing[1], float(xc)/volume.spacing[2]
            h = 25
            zmin, zmax, ymin, ymax, xmin, xmax = int(float(zc))-h, int(float(zc))+h, int(float(yc))-h, int(float(yc))+h, int(float(xc))-h, int(float(xc))+h
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            zmin = max(0, zmin)
            xmax = min(volume.W-1, xmax)
            ymax = min(volume.H-1, ymax)
            zmax = min(volume.D-1, zmax)

            axial_slider_value, coronal_slider_value, saggital_slider_value = int(zc), int(yc), int(xc)

    # MIP画像の準備
    axial = volume.pixel_values_masked[zmin:zmax,:,:].max(axis=0)
    coronal = volume.pixel_values_masked[:,ymin:ymax,:].max(axis=1)
    saggital = volume.pixel_values_masked[:,:,xmin:xmax].max(axis=2)
    axial = draw_roi(axial, [xmin,xmax], [ymin,ymax])
    coronal = draw_roi(coronal, [xmin,xmax], [zmin,zmax])
    saggital = draw_roi(saggital, [ymin,ymax], [zmin,zmax])
    fig_axial = px.imshow(axial, binary_string=True, zmin=0, zmax=255)
    fig_coronal = px.imshow(coronal, binary_string=True, zmin=0, zmax=255)
    fig_saggital = px.imshow(saggital, binary_string=True, zmin=0, zmax=255)
    fig_axial.update_layout(dragmode="drawrect", width=900, hovermode=False)
    fig_coronal.update_layout(dragmode="drawrect", yaxis=dict(autorange=True), height=650, hovermode=False)
    fig_saggital.update_layout(dragmode="drawrect", yaxis=dict(autorange=True), height=650, hovermode=False)

    axial_slider_marks = {i*50: {'label':str(i*50)} for i in range(volume.pixel_values.shape[0]//50)}
    coronal_slider_marks = {i*50: {'label':str(i*50)} for i in range(volume.pixel_values.shape[1]//50)}
    saggital_slider_marks = {i*50: {'label':str(i*50)} for i in range(volume.pixel_values.shape[2]//50)}

    df = df_orig[(df_orig.series_id == volume.name)]
    axial_slider_marks2, coronal_slider_marks2, saggital_slider_marks2 = get_tpfnfp_slider_marks(df)
    axial_slider_marks = {**axial_slider_marks, **axial_slider_marks2}
    coronal_slider_marks = {**coronal_slider_marks, **coronal_slider_marks2}
    saggital_slider_marks = {**saggital_slider_marks, **saggital_slider_marks2}

    return f'Partial MIP for {volume.name}', fig_axial, fig_coronal, fig_saggital, \
           axial_data, coronal_data, saggital_data, xmin, xmax, ymin, ymax, zmin, zmax, \
           volume.pixel_values.shape[0], axial_slider_value, axial_slider_marks, \
           volume.pixel_values.shape[1], coronal_slider_value, coronal_slider_marks, \
           volume.pixel_values.shape[2], saggital_slider_value, saggital_slider_marks


# 候補点の選択肢を切り替える
@application.callback(
    Output('tpfnfp', 'options'),
    Output('tpfnfp', 'value'),
    Input('mip_title', 'children'),
    Input('load_button', 'n_clicks'),
    Input('save_annotation', 'n_clicks'),
    Input('trial_name', 'value'),
    Input('tpfnfp', 'value'),
    State('data_dir', 'value'),
    State('case_list', 'value'),
    State('annot_filepath', 'value'),
    prevent_initial_call=True
)
def update_candidates(mip_title, load_clicks, save_clicks, trial_name, tpfnfp, data_dir, case_name, annot_filepath):
    ctx = dash.callback_context

    datapath = os.path.join(data_dir, case_name)
    spacing_dir = '/mnt/project/brain/aneurysm/takamiya/dataset_for_deepseed/dataset3/spacing_zyx'
    spacingpath = os.path.join(spacing_dir, case_name)

    try:
        annot = pd.read_csv(annot_filepath)
        annot = annot[(annot.series_id==os.path.splitext(case_name)[0]) | (annot.trial_name==trial_name)]
        mark_list = list(annot['mark_id'].values)
    except:
        mark_list = []

    # TP,FN,FPのラジオボタンを作成する
    df = df_orig[(df_orig.series_id == volume.name)]
    options_tmp = get_tpfnfp_options(df)

    options1 = options_tmp

    if ctx.triggered[0]['prop_id'] == 'tpfnfp.value':
        return options1, tpfnfp
    else:
        return options1, 'None'


# volume rendering
@application.callback(
    Output('volume_render', 'children'),
    Output('show_volume', 'children'),
    Input('xmin', 'value'),
    Input('xmax', 'value'),
    Input('ymin', 'value'),
    Input('ymax', 'value'),
    Input('zmin', 'value'),
    Input('zmax', 'value'),
    State('case_list', 'value'),
    State('tpfnfp', 'value'),
    prevent_initial_call=True
)
def show_cropped_volume(xmin, xmax, ymin, ymax, zmin, zmax, case_name, tpfnfp):
    if (xmax-xmin) * (ymax-ymin) * (zmax-zmin) > 2000000:
        raise PreventUpdate

    vol_tmp = volume.pixel_values_masked[zmin:zmax, ymin:ymax, xmin:xmax].copy()
    content = dash_vtk.View([
        dash_vtk.VolumeRepresentation([
            dash_vtk.VolumeController(),
            dash_vtk.ImageData(
                dimensions=[xmax-xmin, ymax-ymin, zmax-zmin],
                origin=[-(xmax-xmin)//2, -(ymax-ymin)//2, -(zmax-zmin)//2],
                spacing=volume.spacing[::-1],
                # spacing=[1, 1, 3],
                children=[
                    dash_vtk.PointData([
                        dash_vtk.DataArray(
                            registration="setScalars",
                            values=list(vol_tmp.flatten()),
                        )
                    ])
                ],
            ),
        ])
    ])
    return content, f'Show Volume of {case_name}'


# MIP画像に線を引いて長さを測定
@application.callback(
    Output('axial_length', 'children'),
    Input('axial', 'relayoutData'),
    Input('reset_length', 'n_clicks'),
    prevent_initial_call=True
)
def measure_axial_length(axial_data, reset_clicks):
    ctx = dash.callback_context
    if ctx.triggered[0]['prop_id'] == 'reset_length.n_clicks':
        return 'Draw line in axial MIP to measure length'

    if (axial_data is not None) and ('shapes' in axial_data) and (axial_data['shapes'][-1]['type'] == 'line'):
        xd, yd = abs(axial_data['shapes'][-1]['x0'] - axial_data['shapes'][-1]['x1']), abs(axial_data['shapes'][-1]['y0'] - axial_data['shapes'][-1]['y1'])
        xd *= volume.spacing[2]
        yd *= volume.spacing[1]
        dist = (xd ** 2 + yd ** 2) ** 0.5
        return f'length of line in coronal MIP is {str(round(dist, 3))} mm'
    else:
        raise PreventUpdate


@application.callback(
    Output('coronal_length', 'children'),
    Input('coronal', 'relayoutData'),
    Input('reset_length', 'n_clicks'),
    prevent_initial_call=True
)
def measure_coronal_length(coronal_data, reset_clicks):
    ctx = dash.callback_context
    if ctx.triggered[0]['prop_id'] == 'reset_length.n_clicks':
        return 'Draw line in coronal MIP to measure length'

    if (coronal_data is not None) and ('shapes' in coronal_data) and (coronal_data['shapes'][-1]['type'] == 'line'):
        xd, zd = abs(coronal_data['shapes'][-1]['x0'] - coronal_data['shapes'][-1]['x1']), abs(coronal_data['shapes'][-1]['y0'] - coronal_data['shapes'][-1]['y1'])
        xd *= volume.spacing[2]
        zd *= volume.spacing[0]
        dist = (xd ** 2 + zd ** 2) ** 0.5
        return f'length of line in coronal MIP is {str(round(dist, 3))} mm'
    else:
        raise PreventUpdate


@application.callback(
    Output('saggital_length', 'children'),
    Input('saggital', 'relayoutData'),
    Input('reset_length', 'n_clicks'),
    prevent_initial_call=True
)
def measure_saggital_length(saggital_data, reset_clicks):
    ctx = dash.callback_context
    if ctx.triggered[0]['prop_id'] == 'reset_length.n_clicks':
        return 'Draw line in saggital MIP to measure length'

    if (saggital_data is not None) and ('shapes' in saggital_data) and (saggital_data['shapes'][-1]['type'] == 'line'):
        yd, zd = abs(saggital_data['shapes'][-1]['x0'] - saggital_data['shapes'][-1]['x1']), abs(saggital_data['shapes'][-1]['y0'] - saggital_data['shapes'][-1]['y1'])
        yd *= volume.spacing[1]
        zd *= volume.spacing[0]
        dist = (yd ** 2 + zd ** 2) ** 0.5
        return f'length of line in saggital MIP is {str(round(dist, 3))} mm'
    else:
        raise PreventUpdate


# application.run_server(debug=True, host='0.0.0.0')



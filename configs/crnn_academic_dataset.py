log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
label_convertor = dict(
    type='CTCConvertor', dict_type='DICT36', with_unknown=False, lower=True)
model = dict(
    type='CRNNNet',
    preprocessor=None,
    backbone=dict(type='VeryDeepVgg', leaky_relu=False, input_channels=1),
    encoder=None,
    decoder=dict(type='CRNNDecoder', in_channels=512, rnn_flag=True),
    loss=dict(type='CTCLoss'),
    label_convertor=dict(
        type='CTCConvertor',
        dict_type='DICT36',
        with_unknown=False,
        lower=True),
    pretrained=None)
img_norm_cfg = dict(mean=[127], std=[127])
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=100,
        max_width=100,
        keep_aspect_ratio=False),
    dict(type='Normalize', mean=[127], std=[127]),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['filename', 'resize_shape', 'text', 'valid_ratio'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=32,
        max_width=None,
        keep_aspect_ratio=True),
    dict(type='Normalize', mean=[127], std=[127]),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'resize_shape', 'valid_ratio', 'img_norm_cfg',
            'ori_filename', 'img_shape', 'ori_shape'
        ])
]
train_root = 'data/mixture/Syn90k'
train_img_prefix = 'data/mixture/Syn90k/mnt/ramdisk/max/90kDICT32px'
train_ann_file = 'data/mixture/Syn90k/label.lmdb'
train = dict(
    type='OCRDataset',
    img_prefix='data/mixture/Syn90k/mnt/ramdisk/max/90kDICT32px',
    ann_file='data/mixture/Syn90k/label.lmdb',
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='lmdb',
        parser=dict(type='LineJsonParser', keys=['filename', 'text'])),
    pipeline=None,
    test_mode=False)
train_list = [
    dict(
        type='OCRDataset',
        img_prefix='data/mixture/Syn90k/mnt/ramdisk/max/90kDICT32px',
        ann_file='data/mixture/Syn90k/label.lmdb',
        loader=dict(
            type='AnnFileLoader',
            repeat=1,
            file_format='lmdb',
            parser=dict(type='LineJsonParser', keys=['filename', 'text'])),
        pipeline=None,
        test_mode=False)
]
test_root = 'data/mixture'
test_img_prefix1 = 'data/mixture/IIIT5K/'
test_img_prefix2 = 'data/mixture/svt/'
test_img_prefix3 = 'data/mixture/icdar_2013/'
test_img_prefix4 = 'data/mixture/icdar_2015/'
test_img_prefix5 = 'data/mixture/svtp/'
test_img_prefix6 = 'data/mixture/ct80/'
test_ann_file1 = 'data/mixture/IIIT5K/test_label.txt'
test_ann_file2 = 'data/mixture/svt/test_label.txt'
test_ann_file3 = 'data/mixture/icdar_2013/test_label_1015.txt'
test_ann_file4 = 'data/mixture/icdar_2015/test_label.txt'
test_ann_file5 = 'data/mixture/svtp/test_label.txt'
test_ann_file6 = 'data/mixture/ct80/test_label.txt'
test1 = dict(
    type='OCRDataset',
    img_prefix='data/mixture/IIIT5K/',
    ann_file='data/mixture/IIIT5K/test_label.txt',
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)
test2 = dict(
    type='OCRDataset',
    img_prefix='data/mixture/svt/',
    ann_file='data/mixture/svt/test_label.txt',
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)
test3 = dict(
    type='OCRDataset',
    img_prefix='data/mixture/icdar_2013/',
    ann_file='data/mixture/icdar_2013/test_label_1015.txt',
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)
test4 = dict(
    type='OCRDataset',
    img_prefix='data/mixture/icdar_2015/',
    ann_file='data/mixture/icdar_2015/test_label.txt',
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)
test5 = dict(
    type='OCRDataset',
    img_prefix='data/mixture/svtp/',
    ann_file='data/mixture/svtp/test_label.txt',
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)
test6 = dict(
    type='OCRDataset',
    img_prefix='data/mixture/ct80/',
    ann_file='data/mixture/ct80/test_label.txt',
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)
test_list = [
    dict(
        type='OCRDataset',
        img_prefix='data/mixture/IIIT5K/',
        ann_file='data/mixture/IIIT5K/test_label.txt',
        loader=dict(
            type='AnnFileLoader',
            repeat=1,
            file_format='txt',
            parser=dict(
                type='LineStrParser',
                keys=['filename', 'text'],
                keys_idx=[0, 1],
                separator=' ')),
        pipeline=None,
        test_mode=True),
    dict(
        type='OCRDataset',
        img_prefix='data/mixture/svt/',
        ann_file='data/mixture/svt/test_label.txt',
        loader=dict(
            type='AnnFileLoader',
            repeat=1,
            file_format='txt',
            parser=dict(
                type='LineStrParser',
                keys=['filename', 'text'],
                keys_idx=[0, 1],
                separator=' ')),
        pipeline=None,
        test_mode=True),
    dict(
        type='OCRDataset',
        img_prefix='data/mixture/icdar_2013/',
        ann_file='data/mixture/icdar_2013/test_label_1015.txt',
        loader=dict(
            type='AnnFileLoader',
            repeat=1,
            file_format='txt',
            parser=dict(
                type='LineStrParser',
                keys=['filename', 'text'],
                keys_idx=[0, 1],
                separator=' ')),
        pipeline=None,
        test_mode=True),
    dict(
        type='OCRDataset',
        img_prefix='data/mixture/icdar_2015/',
        ann_file='data/mixture/icdar_2015/test_label.txt',
        loader=dict(
            type='AnnFileLoader',
            repeat=1,
            file_format='txt',
            parser=dict(
                type='LineStrParser',
                keys=['filename', 'text'],
                keys_idx=[0, 1],
                separator=' ')),
        pipeline=None,
        test_mode=True),
    dict(
        type='OCRDataset',
        img_prefix='data/mixture/svtp/',
        ann_file='data/mixture/svtp/test_label.txt',
        loader=dict(
            type='AnnFileLoader',
            repeat=1,
            file_format='txt',
            parser=dict(
                type='LineStrParser',
                keys=['filename', 'text'],
                keys_idx=[0, 1],
                separator=' ')),
        pipeline=None,
        test_mode=True),
    dict(
        type='OCRDataset',
        img_prefix='data/mixture/ct80/',
        ann_file='data/mixture/ct80/test_label.txt',
        loader=dict(
            type='AnnFileLoader',
            repeat=1,
            file_format='txt',
            parser=dict(
                type='LineStrParser',
                keys=['filename', 'text'],
                keys_idx=[0, 1],
                separator=' ')),
        pipeline=None,
        test_mode=True)
]
optimizer = dict(type='Adadelta', lr=1.0)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[])
runner = dict(type='EpochBasedRunner', max_epochs=5)
checkpoint_config = dict(interval=1)
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/Syn90k/mnt/ramdisk/max/90kDICT32px',
                ann_file='data/mixture/Syn90k/label.lmdb',
                loader=dict(
                    type='AnnFileLoader',
                    repeat=1,
                    file_format='lmdb',
                    parser=dict(
                        type='LineJsonParser', keys=['filename', 'text'])),
                pipeline=None,
                test_mode=False)
        ],
        pipeline=[
            dict(type='LoadImageFromFile', color_type='grayscale'),
            dict(
                type='ResizeOCR',
                height=32,
                min_width=100,
                max_width=100,
                keep_aspect_ratio=False),
            dict(type='Normalize', mean=[127], std=[127]),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=['filename', 'resize_shape', 'text', 'valid_ratio'])
        ]),
    val=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/IIIT5K/',
                ann_file='data/mixture/IIIT5K/test_label.txt',
                loader=dict(
                    type='AnnFileLoader',
                    repeat=1,
                    file_format='txt',
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/svt/',
                ann_file='data/mixture/svt/test_label.txt',
                loader=dict(
                    type='AnnFileLoader',
                    repeat=1,
                    file_format='txt',
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/icdar_2013/',
                ann_file='data/mixture/icdar_2013/test_label_1015.txt',
                loader=dict(
                    type='AnnFileLoader',
                    repeat=1,
                    file_format='txt',
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/icdar_2015/',
                ann_file='data/mixture/icdar_2015/test_label.txt',
                loader=dict(
                    type='AnnFileLoader',
                    repeat=1,
                    file_format='txt',
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/svtp/',
                ann_file='data/mixture/svtp/test_label.txt',
                loader=dict(
                    type='AnnFileLoader',
                    repeat=1,
                    file_format='txt',
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/ct80/',
                ann_file='data/mixture/ct80/test_label.txt',
                loader=dict(
                    type='AnnFileLoader',
                    repeat=1,
                    file_format='txt',
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True)
        ],
        pipeline=[
            dict(type='LoadImageFromFile', color_type='grayscale'),
            dict(
                type='ResizeOCR',
                height=32,
                min_width=32,
                max_width=None,
                keep_aspect_ratio=True),
            dict(type='Normalize', mean=[127], std=[127]),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'filename', 'resize_shape', 'valid_ratio', 'img_norm_cfg',
                    'ori_filename', 'img_shape', 'ori_shape'
                ])
        ]),
    test=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/IIIT5K/',
                ann_file='data/mixture/IIIT5K/test_label.txt',
                loader=dict(
                    type='AnnFileLoader',
                    repeat=1,
                    file_format='txt',
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/svt/',
                ann_file='data/mixture/svt/test_label.txt',
                loader=dict(
                    type='AnnFileLoader',
                    repeat=1,
                    file_format='txt',
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/icdar_2013/',
                ann_file='data/mixture/icdar_2013/test_label_1015.txt',
                loader=dict(
                    type='AnnFileLoader',
                    repeat=1,
                    file_format='txt',
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/icdar_2015/',
                ann_file='data/mixture/icdar_2015/test_label.txt',
                loader=dict(
                    type='AnnFileLoader',
                    repeat=1,
                    file_format='txt',
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/svtp/',
                ann_file='data/mixture/svtp/test_label.txt',
                loader=dict(
                    type='AnnFileLoader',
                    repeat=1,
                    file_format='txt',
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/ct80/',
                ann_file='data/mixture/ct80/test_label.txt',
                loader=dict(
                    type='AnnFileLoader',
                    repeat=1,
                    file_format='txt',
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True)
        ],
        pipeline=[
            dict(type='LoadImageFromFile', color_type='grayscale'),
            dict(
                type='ResizeOCR',
                height=32,
                min_width=32,
                max_width=None,
                keep_aspect_ratio=True),
            dict(type='Normalize', mean=[127], std=[127]),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'filename', 'resize_shape', 'valid_ratio', 'img_norm_cfg',
                    'ori_filename', 'img_shape', 'ori_shape'
                ])
        ]))
evaluation = dict(interval=1, metric='acc')
cudnn_benchmark = True

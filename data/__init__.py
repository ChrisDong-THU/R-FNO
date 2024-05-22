from .dataset import CylinderMask, CylinderVoronoi, SSTMask, SSTVoronoi, ChannelMask, ChannelVoronoi

dataset_dict = {
    'cy':{
        'mask': CylinderMask,
        'voronoi': CylinderVoronoi
    },
    'sst':{
        'mask': SSTMask,
        'voronoi': SSTVoronoi
    },
    'ch':{
        'mask': ChannelMask,
        'voronoi': ChannelVoronoi
    }
}
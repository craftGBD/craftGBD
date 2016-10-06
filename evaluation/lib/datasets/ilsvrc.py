# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2016 Sensetime, CUHK
# Written by Yang Bin, Wang Kun
# --------------------------------------------------------

import  os
import os.path as osp
import datasets.ds_utils as ds_utils
import cPickle
import uuid
import scipy.io as sio
import scipy.sparse
import numpy as np
import xml.etree.ElementTree as ET
from datasets.imdb import imdb
from fast_rcnn.config import cfg

class ilsvrc(imdb):
    """ ILSVRC """
    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'ilsvrc_' + year + '_' + image_set)
        # ILSVRC specific config options
        self.config = {'top_k' : 2000,
                       'use_salt' : True,
                       'cleanup' : True,
                       'min_size' : 2}
        # name, paths
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                                         else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'ILSVRC' + self._year)
        self._classes = ('__background__',  # always index 0
                         'n02672831', 'n02691156', 'n02219486', 'n02419796', 'n07739125',
                         'n02454379', 'n07718747', 'n02764044', 'n02766320', 'n02769748',
                         'n07693725', 'n02777292', 'n07753592', 'n02786058', 'n02787622',
                         'n02799071', 'n02802426', 'n02807133', 'n02815834', 'n02131653',
                         'n02206856', 'n07720875', 'n02828884', 'n02834778', 'n02840245',
                         'n01503061', 'n02870880', 'n02883205', 'n02879718', 'n02880940',
                         'n02892767', 'n07880968', 'n02924116', 'n02274259', 'n02437136',
                         'n02951585', 'n02958343', 'n02970849', 'n02402425', 'n02992211',
                         'n01784675', 'n03000684', 'n03001627', 'n03017168', 'n03062245',
                         'n03063338', 'n03085013', 'n03793489', 'n03109150', 'n03128519',
                         'n03134739', 'n03141823', 'n07718472', 'n03797390', 'n03188531',
                         'n03196217', 'n03207941', 'n02084071', 'n02121808', 'n02268443',
                         'n03249569', 'n03255030', 'n03271574', 'n02503517', 'n03314780',
                         'n07753113', 'n03337140', 'n03991062', 'n03372029', 'n02118333',
                         'n03394916', 'n01639765', 'n03400231', 'n02510455', 'n01443537',
                         'n03445777', 'n03445924', 'n07583066', 'n03467517', 'n03483316',
                         'n03476991', 'n07697100', 'n03481172', 'n02342885', 'n03494278',
                         'n03495258', 'n03124170', 'n07714571', 'n03513137', 'n02398521',
                         'n03535780', 'n02374451', 'n07697537', 'n03584254', 'n01990800',
                         'n01910747', 'n01882714', 'n03633091', 'n02165456', 'n03636649',
                         'n03642806', 'n07749582', 'n02129165', 'n03676483', 'n01674464',
                         'n01982650', 'n03710721', 'n03720891', 'n03759954', 'n03761084',
                         'n03764736', 'n03770439', 'n02484322', 'n03790512', 'n07734744',
                         'n03804744', 'n03814639', 'n03838899', 'n07747607', 'n02444819',
                         'n03908618', 'n03908714', 'n03916031', 'n00007846', 'n03928116',
                         'n07753275', 'n03942813', 'n03950228', 'n07873807', 'n03958227',
                         'n03961711', 'n07768694', 'n07615774', 'n02346627', 'n03995372',
                         'n07695742', 'n04004767', 'n04019541', 'n04023962', 'n04026417',
                         'n02324045', 'n04039381', 'n01495701', 'n02509815', 'n04070727',
                         'n04074963', 'n04116512', 'n04118538', 'n04118776', 'n04131690',
                         'n04141076', 'n01770393', 'n04154565', 'n02076196', 'n02411705',
                         'n04228054', 'n02445715', 'n01944390', 'n01726692', 'n04252077',
                         'n04252225', 'n04254120', 'n04254680', 'n04256520', 'n04270147',
                         'n02355227', 'n02317335', 'n04317175', 'n04330267', 'n04332243',
                         'n07745940', 'n04336792', 'n04356056', 'n04371430', 'n02395003',
                         'n04376876', 'n04379243', 'n04392985', 'n04409515', 'n01776313',
                         'n04591157', 'n02129604', 'n04442312', 'n06874185', 'n04468005',
                         'n04487394', 'n03110669', 'n01662784', 'n03211117', 'n04509417',
                         'n04517823', 'n04536866', 'n04540053', 'n04542943', 'n04554684',
                         'n04557648', 'n04530566', 'n02062744', 'n04591713', 'n02391049')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self.set_proposal_method('slide')
        self.competition_mode(False)
        # Dataset splits that have ground-truth annotations (test splits
        # do not have gt annotations)
        self._gt_splits = ('trainval', 'val1')

        assert os.path.exists(self._devkit_path), \
                'ILSVRCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'Data', self._image_set,
                                  index + '.JPEG')
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _get_default_path(self):
        """
        Return the default path where ILSVRC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'ILSVRCdevkit' + self._year)

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /ILSVRCdevkit2013/ILSVRC2013/ImageSets/Main/val2.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_ilsvrc_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def _get_box_file(self, index):
        file_name = index + '.mat'
        return osp.join(self._image_set, file_name)

    def slide_roidb(self):
        return self._roidb_from_proposals('slide')

    def _roidb_from_proposals(self, method):
        """
        Creates a roidb from pre-computed proposals of a particular methods.
        """
        top_k = self.config['top_k']
        cache_file = osp.join(self.cache_path, self.name +
                              '_{:s}_top{:d}'.format(method, top_k) +
                              '_roidb.pkl')

        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{:s} {:s} roidb loaded from {:s}'.format(self.name, method,
                                                            cache_file)
            return roidb

        if self._image_set in self._gt_splits:
            gt_roidb = self.gt_roidb()
            method_roidb = self._load_proposals(method, gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, method_roidb)
        else:
            roidb = self._load_proposals(method, None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote {:s} roidb to {:s}'.format(method, cache_file)
        return roidb

    def _load_proposals(self, method, gt_roidb):
        """
        Load pre-computed proposals in the format provided by Jan Hosang:
        http://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-
          computing/research/object-recognition-and-scene-understanding/how-
          good-are-detection-proposals-really/
        """
        box_list = []
        top_k = self.config['top_k']
        valid_methods = ['slide']
        assert method in valid_methods

        print 'Loading {} boxes'.format(method)
        for i, index in enumerate(self._image_index):
            if i % 1000 == 0:
                print '{:d} / {:d}'.format(i + 1, len(self._image_index))

            box_file = osp.join(
                cfg.DATA_DIR, 'ilsvrc_proposals', method, 'mat',
                self._get_box_file(index))

            raw_data = sio.loadmat(box_file)['boxes']
            boxes = np.maximum(raw_data[:, 0:4] - 1, 0).astype(np.uint16)
            # Remove duplicate boxes and very small boxes and then take top k
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            boxes = boxes[:top_k, :]
            box_list.append(boxes)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_ilsvrc_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', self._image_set, index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        height = float(tree.find('size').find('height').text)
        width = float(tree.find('size').find('width').text)
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based

            x1 = max(float(bbox.find('xmin').text) - 1, 0.0)
            y1 = max(float(bbox.find('ymin').text) - 1, 0.0)
            x2 = min(float(bbox.find('xmax').text) - 1, width - 1)
            y2 = min(float(bbox.find('ymax').text) - 1, height - 1)
            assert x2 > x1, '{}: xmax should be greater than xmin'.format(filename)
            assert y2 > y1, '{}: ymax should be greater than ymin'.format(filename)
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def _write_ilsvrc_results_file(self, all_boxes, res_file):
        with open(res_file, 'wt') as f:
            for im_ind, index in enumerate(self.image_index):
                for cls_ind, cls in enumerate(self.classes):
                    if cls == '__background__':
                        continue
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # Expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:d} {:d} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(im_ind + 1, cls_ind, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def evaluate_detections(self, all_boxes, output_dir, startIdx=0, endIdx=-1):
        res_file = os.path.join(output_dir, ('detections_' +
                                             self._image_set +
                                             self._year +
                                             '_results_'))
        if self.config['use_salt'] and 0:
            res_file += '_{}'.format(str(uuid.uuid4()))
        res_file = res_file  + str(startIdx) + 'to' + str(endIdx)
        res_file += '.txt'
        print res_file
        self._write_ilsvrc_results_file(all_boxes, res_file)
        # Optionally cleanup results txt file
        if self.config['cleanup'] and 0:
            os.remove(res_file)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

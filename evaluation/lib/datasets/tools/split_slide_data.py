import sys
import scipy.io as sio

def split_slide_data(mat_file_name, image_set_file, output_dir):
    raw_data = sio.loadmat(mat_file_name)['boxes'].ravel()
    with open(image_set_file) as f:
        image_index = [x.strip() for x in f.readlines()]
    for i in xrange(raw_data.shape[0]):
        key = image_index[i]
        boxes = raw_data[i]
        print '{} / {} {}'.format(i + 1, raw_data.shape[0], key)
        sio.savemat(output_dir + '/' + key, {'boxes': boxes})

if __name__ == '__main__':
    mat_file_name = sys.argv[1]
    image_set_file = sys.argv[2]
    output_dir = sys.argv[3]
    split_slide_data(mat_file_name, image_set_file, output_dir)

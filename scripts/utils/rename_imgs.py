from os import listdir
from os.path import isfile, join

DIR_PATH = "images/output_imgs/exp3"

onlyfiles = [f for f in listdir(DIR_PATH) if isfile(join(DIR_PATH, f))]

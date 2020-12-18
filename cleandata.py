import os
import re
import shutil

path = 'DocBank_500K_txt/DocBank_500K_txt'
path_extracted = 'extracted'

def walkFile_txt(path):
    if not os.path.isdir(path_extracted):
        os.makedirs(path_extracted)
    for root, dirs, files in os.walk(path):
        for f in files:
            if (f.find(".txt")) > -1:
                with open(os.path.join(path, f), 'r', encoding='utf-8') as fi:
                    txt = fi.read()
                if (len(re.findall("table\n", txt)) != 0):
                    with open('filelist.txt', 'a+', encoding='utf-8') as output:
                        output.write(f + '\n')
                        shutil.copyfile(os.path.join(path, f),(path_extracted + '/' + f))
                        print(f)

walkFile_txt(path)

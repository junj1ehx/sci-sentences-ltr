import os
import re
import shutil
listPath = 'F:\PycharmProject\deim\one-column.txt'
output = 'F:\PycharmProject\deim\DocBank_500K_txt\ddataset'
docbank = 'F:\PycharmProject\deim\DocBank_500K_txt\DocBank_500K_txt'

with open(listPath, 'r', encoding='utf-8')as f:
    filelist = f.read().split('\n')
for file in filelist:
    shutil.copyfile(os.path.join(docbank, file), os.path.join(output, file))



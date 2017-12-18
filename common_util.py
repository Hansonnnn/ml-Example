import configparser
import pandas as pd




def get_path(section, path_name):
    conf = configparser.ConfigParser()
    conf.read('/Users/hanzhao/PycharmProjects/ml-example/file/config/ml.conf', encoding='utf-8')
    return conf.get(section, path_name)



def get_data(self, file_name, sep=',', encoding='utf-8', **kwargs):
    """@description withdraw data from csv
       @:param file_name  the file of csv formatter
       @:param sep  file's seperator  default is ',' 
       @:param encoding encoding way content when read file 
       @:param kwargs another parameter ,for example, usecols to specialize which one column should be read"""

    if kwargs.get('usecols') is not None:
        usecols = kwargs.get('usecols')
        data = pd.read_csv(file_name, sep=sep, encoding=encoding, usecols=usecols)
        return data
    data = pd.read_csv(file_name, sep=sep, encoding=encoding)
    return data

def write_file(self,file_path,text,encoding='utf-8'):
    """@description  util with write file"""
    with open(file_path,'w',encoding=encoding) as f:
        f.write(text)


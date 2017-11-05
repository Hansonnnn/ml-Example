import configparser


def get_path(section, path_name):
    conf = configparser.ConfigParser()
    conf.read('/Users/hanzhao/PycharmProjects/ml-example/file/config/ml.conf', encoding='utf-8')
    return conf.get(section, path_name)

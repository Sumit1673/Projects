import argparse
parser = argparse.ArgumentParser(description='KMeans')
args_list = []

def add_agr_grp(grp_name):
    arg = parser.add_argument_group(grp_name)
    args_list.append(arg)
    return arg

kmeans_params = add_agr_grp('kmeans_params')
kmeans_params.add_argument("--image_path", type=str, default='t1.png')
kmeans_params.add_argument("--k", type=int, default=5)
kmeans_params.add_argument("--norm", type=str, default='L2')
kmeans_params.add_argument("--iter", type=str, default=10)

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
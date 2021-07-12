# Copyright: (c) 2021, Edwin G. W. Peters

def safe_get_from_list(data,attr,default_value):
    """
    Returns data['attr'] if attr is in data, else returns default_value
    """
    return data[attr] if attr in data.keys() else default_value

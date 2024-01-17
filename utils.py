import re


def determine_dimensions(collection: dict):
    """
    Determines the spatial dimensions of an input geojson.
    """
    w, h, b = None, None, None
    for feature_ in collection["features"]:
        property_names = list(feature_["properties"].keys())
        for prop_name_ in property_names:
            rows = feature_["properties"][prop_name_]
            if rows is not None:
                b = len(feature_["properties"])
                h = len(feature_["properties"][prop_name_])        # number of rows
                w = len(feature_["properties"][prop_name_][0])     # number of columns (values per row)
                break
        if h is not None: break
    return w, h, b


def file_to_tree_type_name(file_name: str, identifier: str) -> str:
    '''
    This function extracts the tree type name for a given geojson file.

    file_name: name of geojson file
    '''

    tree_type = re.search(f"([A-Z][a-z]+_[a-z]+)_{identifier}.geojson", file_name).group(1)
    return tree_type


def sample_file_to_tree_type(file_name: str) -> str:
    try:
        tree_type = re.search("([A-Z][a-z]+_[a-z]+)-", file_name).group(1)
    except:
        raise AttributeError(f"Error while parsing filename '{file_name}'")
    return tree_type

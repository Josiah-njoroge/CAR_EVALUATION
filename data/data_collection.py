from ucimlrepo import fetch_ucirepo 

def get_data(id: int, as_frame: bool = True):
    """
    Fetches a dataset from the UCI Machine Learning Repository.

    Parameters:
    id (int): The unique identifier of the dataset to fetch.
    as_frame (bool): If True, returns the data as pandas DataFrames. Default is True.

    Returns:
    tuple: A tuple containing the features and targets of the dataset.
    """
    dataset = fetch_ucirepo(id=id)
    X = dataset.data.features 
    y = dataset.data.targets 
    metadata = dataset.metadata
    variables = dataset.variables
    
    return X, y, metadata, variables

if __name__ == "__main__":
    X, y, metadata, variables = get_data(id=19)
    print("Features:\n", X.head())
    print("Targets:\n", y.head())
    print("Metadata:\n", metadata)
    print("Variables:\n", variables)
    
    
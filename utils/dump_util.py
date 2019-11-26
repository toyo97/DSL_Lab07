import pandas as pd


def dump_to_file(indices, labels, filename: str):
    """
    Dump the evaluated labels to a CSV file
    """
    assert len(indices) == len(labels)
    df = pd.DataFrame({'Id': indices, 'Predicted': labels})
    df.to_csv(filename, index=False)

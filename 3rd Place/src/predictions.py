from src.preprocessing import create_training_and_testing_data
import gc
import pandas as pd

    
def get_predictions(model, test_meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    It takes a trained model and a test meta dataframe as input and returns a dataframe with the
    predictions for each sample
    
    :param model: the model to use for prediction
    :param test_meta_df: the test meta dataframe
    :type test_meta_df: pd.DataFrame
    :return: A dataframe with the sample_id as the index and the predicted probabilities for each class
    as the columns.
    """
    
    # Get the test data with features
    test_df = create_training_and_testing_data(test_meta_df, labels=None)
    
    test_df['preds'] = model.predict(test_df)
    
    res_df = pd.pivot_table(test_df[['sample_id', 'target_label', 'preds']],
                            index='sample_id',
                            columns='target_label',
                            values='preds')
    
    ## Mapping the labels to the original labels
    target_inv_mapper = {
    0: 'basalt',
    1: 'carbonate',
    2: 'chloride',
    3: 'iron_oxide',
    4: 'oxalate',
    5: 'oxychlorine',
    6: 'phyllosilicate',
    7: 'silicate',
    8: 'sulfate',
    9: 'sulfide'}
    
    
    res_df.columns = [target_inv_mapper[c] for c in res_df.columns]
    
    res_df = res_df.reset_index()
    
    del test_df
    _ = gc.collect()
    
    return res_df
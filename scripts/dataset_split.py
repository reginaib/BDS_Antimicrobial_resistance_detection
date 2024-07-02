import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(filtered, antibiotic_columns, test_size=0.15, validation_size=0.15):
    # Initialize a dictionary to store the split data for each antibiotic
    split_results = {}
    
    # Remove the target columns (antibiotics) to isolate features
    all_features = filtered.drop(columns=antibiotic_columns)
    
    # Iterate over each antibiotic to create individual splits
    for antibiotic in antibiotic_columns:
        # Drop NaN values from the antibiotic column to clean up labels
        labels = filtered[antibiotic].dropna()
        
        # Select corresponding features for the non-NaN labels
        features = all_features.loc[labels.index]
        
        # Split data into temporary and test datasets based on the given test_size
        # This first split segregates the test data
        X_temp, X_test, y_temp, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)
        
        # Adjust validation size to account for the reduced size after test split
        adjusted_validation_size = validation_size / (1 - test_size)
        
        # Split the temporary data into final training and validation datasets
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=adjusted_validation_size, random_state=42)
        
        # Store the splits in a dictionary with keys corresponding to the data types
        split_results[antibiotic] = {
            'X_train': X_train,  
            'X_valid': X_val,    
            'X_test': X_test,   
            'y_train': y_train, 
            'y_valid': y_val,   
            'y_test': y_test    
        }
    
    return split_results


def process_and_combine_data(split_results, antibiotic_columns):
    # Initialize a list to hold all the processed dataframes
    all_data = []

    # Process the data for each specified antibiotic
    for antibiotic in antibiotic_columns:
        # Retrieve the split data for the current antibiotic from the dictionary
        datasets = split_results[antibiotic]
        
        # Copy datasets to prevent modifying the original data during processing
        X_train = datasets['X_train'].copy()
        X_valid = datasets['X_valid'].copy()
        X_test = datasets['X_test'].copy()

        # Pop the species column from the data; it's assumed to be the last column in the feature set
        species_train = X_train.pop(X_train.columns[-1])
        species_valid = X_valid.pop(X_valid.columns[-1])
        species_test = X_test.pop(X_test.columns[-1])

        # Assign labels from the split results to respective variables
        res_train = datasets['y_train']
        res_valid = datasets['y_valid']
        res_test = datasets['y_test']

        # Add a new column to indicate the type of data split: 0 for train, 1 for valid, 2 for test
        X_train['split'] = 0
        X_valid['split'] = 1
        X_test['split'] = 2

        # Store the species and resistance data in their respective training, validation, and test sets
        X_train['y_species'] = species_train
        X_train[antibiotic] = res_train

        X_valid['y_species'] = species_valid
        X_valid[antibiotic] = res_valid

        X_test['y_species'] = species_test
        X_test[antibiotic] = res_test

        # Append the processed training, validation, and test sets for the current antibiotic to the list
        all_data.extend([X_train, X_valid, X_test])

    # Combine all processed data into a single DataFrame
    final_combined_df = pd.concat(all_data, ignore_index=True)
    return final_combined_df
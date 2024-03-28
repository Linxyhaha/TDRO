## Data descriptions 

For each dataset, we provide interactions, item timestamps, item features, warm-start/cold-start item sets, and user/item mappings. Below is the detailed descriptions for each type of files for your reference.

**Interactions**

- {Training/validation/testing}_dict.npy: The files contain the interactions in the training, validation, and testing set in a dictionary, respectively. The key of the dictionary is the user ID, and the value is a list of the interacted item IDs of the user in chronological order. Note that validation/testing_dict contain both warm and cold items, i.e., "all" setting.
  - For example, 0: [1,4,7,10]
- {validation/testing}_{warm/cold}_dict: The files contain the interactions of the warm/cold items in the validation or testing sets in a dictionary. The key of the dictionary is the user ID, and the value is a list of the interacted warm/cold item IDs in chronological order, i.e., "warm" and "cold" setting. 
  - For example, 0: [2,11]

**item timestamps**

- item_time_dict.npy: item_time_dict.npy: This file stores the timestamp of the first appearance of the item in the dataset, i.e., the earliest timestamp of each item. It is stored in a dictionary, where the key is the original item ID, and the value is the earliest timestamp of the item.


**item features**

- item_pca_map.npy (taking Amazon as an example): a dictionary of item visual features, where key is the item id, and the value is the 64-dimension feature vector.

**Warm/Cold item sets**

- warm_item.npy: This file stores the set of the warm item IDs, i.e., the items that appear in the training set.
- cold_item.npy: This file stores the set of the cold item IDs, i.e., the items that do not appear in the training set.

**Mapping**

- user_map.npy: This file stores the mapping of the users in a dictionary, where the key is the original user ID, and the value is the mapped user ID. All data used in experiments are based on the mapped IDs.
- item_map.npy: This file stores the mapping of the items in a dictionary, where the key is the original item ID, and the value is the mapped item ID. All data used in experiments are based on the mapped IDs.


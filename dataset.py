import torch
from torch.utils.data import Dataset
import pandas as pd
import ast

class TokenizedDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Parse string representations of lists back to lists
        input_ids1 = ast.literal_eval(row['input_ids1'])
        attention_mask1 = ast.literal_eval(row['attention_mask1'])
        input_ids2 = ast.literal_eval(row['input_ids2'])
        attention_mask2 = ast.literal_eval(row['attention_mask2'])
        label = int(row['label'])
        issue_id1 = int(row['issue_id1'])
        issue_id2 = int(row['issue_id2'])
        
        return {
            'input_ids1': torch.tensor(input_ids1, dtype=torch.long),
            'attention_mask1': torch.tensor(attention_mask1, dtype=torch.long),
            'input_ids2': torch.tensor(input_ids2, dtype=torch.long),
            'attention_mask2': torch.tensor(attention_mask2, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float),
            'issue_id1': issue_id1,
            'issue_id2': issue_id2
        }

if __name__ == "__main__":
    # Example usage
    import os
    
    relative_path = "eclipse"
    dataset_dir = f"datasets/{relative_path}/"
    cur_dir = os.path.dirname(__file__)
    csv_path = os.path.join(cur_dir, dataset_dir, "tokenized_pairs.csv")
    
    if os.path.exists(csv_path):
        dataset = TokenizedDataset(csv_path)
        print(f"Dataset size: {len(dataset)}")
        sample = dataset[0]
        print("Sample keys:", sample.keys())
        print("Label:", sample['label'])
    else:
        print(f"File not found: {csv_path}")

import torch
import torch.nn as nn
from transformers import AutoModel

class SiameseBERT(nn.Module):
    def __init__(self, model_name="bert-base-uncased", projection_dim=128):
        super(SiameseBERT, self).__init__()
        
        # Load the BERT model (shared weights)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Linear layer to project the embeddings to a score/vector
        self.fc = nn.Linear(self.bert.config.hidden_size, projection_dim)
        
    def single_bert(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the CLS token embedding (index 0)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.fc(cls_output)

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):

     

        # Get scores (projected vectors) for both inputs
        score1 = self.single_bert(input_ids1, attention_mask1)
        score2 = self.single_bert(input_ids2, attention_mask2)
        
        return score1, score2
      
    def create_embedding(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]
           
        return cls_output

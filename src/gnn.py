import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sbert import SiameseBERT


def _add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    device = edge_index.device
    loop = torch.arange(num_nodes, device=device)
    loop_index = torch.stack([loop, loop], dim=0)
    return torch.cat([edge_index, loop_index], dim=1)

def normalize_edge_index(edge_index: torch.Tensor, num_nodes: int):
    edge_index2 = _add_self_loops(edge_index, num_nodes=num_nodes)
    row, col = edge_index2[0], edge_index2[1]
    deg = torch.bincount(row, minlength=num_nodes).float()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return edge_index2, edge_weight

class GCN(nn.Module):
    def __init__(self, in_dim: int, hid: int = 256, out: int = 128, dropout: float = 0.1):
        super().__init__()
        self.c1 = GCNConv(in_dim, hid, bias=False)
        self.c2 = GCNConv(hid, out, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.c1(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.c2(x, edge_index, edge_weight=edge_weight)
        return x

class SBERTGCN(nn.Module):
    def __init__(
        self,
        sbert_model: nn.Module,
        num_nodes: int,
        gcn_in_dim: int,
        gcn_hid: int = 128,
        gcn_out: int = 128,
        gcn_dropout: float = 0.1,
        l: float = 0.7,
        sbert_dim: int | None = None,
    ):
        super().__init__()
        self.sbert = sbert_model
        self.gcn = GCN(in_dim=gcn_in_dim, hid=gcn_hid, out=gcn_out, dropout=gcn_dropout)
        self.l = float(l)
        self.num_nodes = num_nodes
        self.gcn_out_dim = gcn_out
        
        # Projection layer if SBERT and GCN dims don't match
        self.sbert_proj = None
        if sbert_dim is not None:
            self._maybe_init_proj(sbert_dim, torch.device('cpu'))

    def _maybe_init_proj(self, sbert_out_dim: int, device: torch.device):
        if self.sbert_proj is None:
            if sbert_out_dim == self.gcn_out_dim:
                self.sbert_proj = nn.Identity()
            else:
                self.sbert_proj = nn.Linear(sbert_out_dim, self.gcn_out_dim, bias=False)
            self.sbert_proj.to(device)

    def forward(self, node_emb, idx1, idx2, input_ids1, attention_mask1, input_ids2, attention_mask2, edge_index, edge_weight):
        # 1. Get current BERT semantic outputs
        s1, s2 = self.sbert(input_ids1, attention_mask1, input_ids2, attention_mask2)

        # If needed, project SBERT outputs to match GCN output dim
        self._maybe_init_proj(s1.size(-1), s1.device)
        s1 = self.sbert_proj(s1)
        s2 = self.sbert_proj(s2)
        
        # 2. PERSISTENCE STEP: Update the global embedding table
        # We use .detach() because we don't want to backpropagate through 
        # the entire history of the table, only the current batch.
        with torch.no_grad():
            node_emb.weight[idx1.long()] = s1.detach()
            node_emb.weight[idx2.long()] = s2.detach()
        
        # 3. Use the updated table for the GCN
        # Now 'x' contains CURRENT BERT outputs for this batch 
        # AND PREVIOUS BERT outputs for any node already seen.
        x = node_emb.weight 

        x_g = self.gcn(x, edge_index, edge_weight)

        g1, g2 = x_g[idx1.long()], x_g[idx2.long()]

        # 4. Weighted sum fusion (no concatenation)
        combined1 = self.l * s1 + (1.0 - self.l) * g1
        combined2 = self.l * s2 + (1.0 - self.l) * g2

        return combined1, combined2
# --- Dummy SBERT for testing ---
class _DummyPairSBERT(nn.Module):
    def __init__(self, vocab_size: int = 100, out_dim: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, out_dim)

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        x1 = self.emb(input_ids1).mean(dim=1)
        x2 = self.emb(input_ids2).mean(dim=1)
        return x1, x2

def main() -> None:
    torch.manual_seed(0)

    num_nodes = 100
    feat_dim = 128
    
    # 1. Setup Graph
    edges = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    edge_index, edge_weight = normalize_edge_index(edges, num_nodes=num_nodes)
    node_feats = torch.randn(num_nodes, feat_dim)

    # 2. Setup Batch (Indices and Text)
    # Batch size B = 4
    idx1 = torch.tensor([0, 2, 4, 6], dtype=torch.long)
    idx2 = torch.tensor([1, 3, 5, 7], dtype=torch.long)
    
    B, T = idx1.size(0), 128
    vocab_size = 100
    input_ids1 = torch.randint(0, vocab_size, (B, T))
    input_ids2 = torch.randint(0, vocab_size, (B, T))
    attn_mask = torch.ones(B, T, dtype=torch.long)

    # 3. Initialize Model
    sbert_mock  = SiameseBERT(model_name="bert-base-uncased", projection_dim=feat_dim)
    model = SBERTGCN(
        sbert_model=sbert_mock,
        num_nodes=num_nodes,
        gcn_in_dim=feat_dim,
        gcn_out=feat_dim,
        l=0.5
    )

    # 4. Forward Pass
    out1, out2 = model(
        node_feats=node_feats,
        idx1=idx1, idx2=idx2,
        input_ids1=input_ids1, attention_mask1=attn_mask,
        input_ids2=input_ids2, attention_mask2=attn_mask,
        edge_index=edge_index, edge_weight=edge_weight
    )

    print(f"Output shapes: {out1.shape}, {out2.shape}")
    sims = F.cosine_similarity(out1, out2, dim=1)
    loss = (1.0 - sims).mean()
    loss.backward()
    print(f"Loss: {loss.item():.4f} - Backward pass successful.")

if __name__ == "__main__":
    main()
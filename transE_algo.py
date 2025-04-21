import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Hyperparameters
embedding_dim = 50  # Embedding size
learning_rate = 0.01
margin = 1.0
num_epochs = 100
batch_size = 32

# Sample entity and relation sets
entities = {"h1", "h2", "t1", "t2"}
relations = {"r1", "r2"}
triplets = [("h1", "r1", "t1"), ("h2", "r2", "t2")]  # Training data

# Mapping entities and relations to indices
entity2id = {ent: i for i, ent in enumerate(entities)}
relation2id = {rel: i for i, rel in enumerate(relations)}

# Define the TransE Model
class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(TransE, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # Initialize embeddings
        nn.init.uniform_(self.entity_embeddings.weight, -6/np.sqrt(embedding_dim), 6/np.sqrt(embedding_dim))
        nn.init.uniform_(self.relation_embeddings.weight, -6/np.sqrt(embedding_dim), 6/np.sqrt(embedding_dim))
    
    def forward(self, h, r, t):
        h_embed = self.entity_embeddings(h)
        r_embed = self.relation_embeddings(r)
        t_embed = self.entity_embeddings(t)
        return h_embed + r_embed - t_embed
    
    def score(self, h, r, t):
        return torch.norm(self.forward(h, r, t), p=2, dim=1)

# Create model
model = TransE(len(entities), len(relations), embedding_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.MarginRankingLoss(margin=margin)

def get_corrupted_triplet(triplet):
    h, r, t = triplet
    if np.random.rand() > 0.5:
        h = np.random.choice(list(entities - {h}))  # Corrupt head
    else:
        t = np.random.choice(list(entities - {t}))  # Corrupt tail
    return (h, r, t)

# Training Loop
for epoch in range(num_epochs):
    batch_loss = 0
    for _ in range(batch_size):
        # Sample positive triplet
        pos_triplet = triplets[np.random.randint(len(triplets))]
        corrupted_triplet = get_corrupted_triplet(pos_triplet)
        
        h, r, t = torch.tensor(entity2id[pos_triplet[0]]), torch.tensor(relation2id[pos_triplet[1]]), torch.tensor(entity2id[pos_triplet[2]])
        h_corrupt, r_corrupt, t_corrupt = torch.tensor(entity2id[corrupted_triplet[0]]), torch.tensor(relation2id[corrupted_triplet[1]]), torch.tensor(entity2id[corrupted_triplet[2]])
        
        optimizer.zero_grad()
        pos_score = model.score(h.unsqueeze(0), r.unsqueeze(0), t.unsqueeze(0))
        neg_score = model.score(h_corrupt.unsqueeze(0), r_corrupt.unsqueeze(0), t_corrupt.unsqueeze(0))
        loss = loss_function(pos_score, neg_score, torch.tensor([-1.0]))  # Minimize pos, maximize neg
        loss.backward()
        optimizer.step()
        batch_loss += loss.item()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {batch_loss / batch_size}")

print("Training complete.")
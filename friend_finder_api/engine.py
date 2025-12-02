import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
import ast
import os
from torch_geometric.nn import SAGEConv, to_hetero
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

# ==========================================================
# 1. MODEL ARCHITECTURE (Matches Training EXACTLY)
# ==========================================================
class SocialGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.bn = nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn(x) 
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return F.normalize(x, p=2, dim=-1)

# ==========================================================
# 2. FRIEND FINDER LOGIC CLASS
# ==========================================================
class FriendFinderEngine:
    def __init__(self, model_dir="model_files"):
        print("⏳ Loading Friend Finder System (Social Graph Only)...")
        self.device = torch.device('cpu') 

        # A. Load Artifacts
        try:
            base_path = os.path.dirname(os.path.abspath(__file__))
            full_model_dir = os.path.join(base_path, model_dir)
            
            with open(os.path.join(full_model_dir, "artifacts.pkl"), "rb") as f:
                self.artifacts = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ Could not find artifacts.pkl in {full_model_dir}")
            
        # Robust DataFrame Loading
        self.df = self.artifacts.get('df') 
        if self.df is None:
             self.df = self.artifacts.get('df_search')

        self.scaler = self.artifacts['scaler']
        self.metadata = self.artifacts['metadata']
        
        # Maps (Strictly Social Maps Only)
        self.hobby_map = self.artifacts.get('hobby_map', {})
        self.major_map = self.artifacts.get('major_map', {})
        
        # Create Lowercase Lookup for Hobbies (Crucial for matching input "guitar" to DB "Guitar")
        self.hobby_map_lower = {str(k).lower().strip(): v for k, v in self.hobby_map.items()}

        # Counts
        self.num_majors = len(self.major_map) if self.major_map else 1
        self.num_hobbies = len(self.hobby_map) if self.hobby_map else 1
        
        # B. Load Vectors
        self.db_vectors = np.load(os.path.join(full_model_dir, "user_embeddings.npy"))
        
        # C. Initialize KNN
        self.knn = NearestNeighbors(n_neighbors=50, metric='cosine')
        self.knn.fit(self.db_vectors)

        # D. Load Models
        print("⏳ Loading MPNet & GNN...")
        self.text_encoder = SentenceTransformer('all-mpnet-base-v2')
        
        self.model = SocialGNN(hidden_channels=256, out_channels=64)
        self.model = to_hetero(self.model, self.metadata, aggr='sum')
        
        try:
            # Try main name first, then fallback
            pth_path = os.path.join(full_model_dir, "gnn_model.pth")
            if not os.path.exists(pth_path):
                pth_path = os.path.join(full_model_dir, "gnn_state_dict.pth")
                
            state = torch.load(pth_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.eval()
        except Exception as e:
             raise RuntimeError(f"❌ Error loading model weights: {e}")
        
        print("✅ Friend Finder Ready!")

    def _process_new_user_features(self, user_data):
        """
        Creates embedding for new user.
        STRICTLY follows training logic: User + Major + Hobby (No Courses).
        """
        # --- 1. Features (772 dims) ---
        full_text = f"{user_data.get('unique_quality', '')}. {user_data.get('story', '')}"
        text_emb = self.text_encoder.encode([full_text])
        
        # Map Year string to int (Same as training)
        y_map = {'Freshman': 1, 'Sophomore': 2, 'Junior': 3, 'Senior': 4, 'Graduate': 5, 'PhD': 6}
        y_num = y_map.get(user_data.get('year'), 1)
        
        # Scale Numbers (Same scaler as training)
        age = user_data.get('age', 20)
        gpa = user_data.get('gpa', 3.0)
        num_feats = self.scaler.transform([[age, gpa, y_num]])
        
        # One-Hot Sex (Same logic as training)
        sex_val = 1 if user_data.get('sex') == 'Female' else 0
        sex_feat = np.array([[sex_val]])
        
        # Combine
        raw_features = np.hstack([text_emb, num_feats, sex_feat])
        x_user = torch.tensor(raw_features, dtype=torch.float)

        # --- 2. Map Connections ---
        major_idx = self.major_map.get(user_data.get('major'), -1)
        
        hobby_indices = []
        for h in user_data.get('hobbies', []):
            h_clean = str(h).lower().strip() 
            if h_clean in self.hobby_map_lower:
                hobby_indices.append(self.hobby_map_lower[h_clean])

        # --- 3. Build Edges ---
        # We need BOTH directions because training used T.ToUndirected()
        
        # Major Edges
        if major_idx != -1:
            edge_studies = torch.tensor([[0], [major_idx]], dtype=torch.long)
            edge_rev_studies = torch.tensor([[major_idx], [0]], dtype=torch.long)
        else:
            edge_studies = torch.empty((2, 0), dtype=torch.long)
            edge_rev_studies = torch.empty((2, 0), dtype=torch.long)

        # Hobby Edges
        if hobby_indices:
            edge_likes = torch.tensor([[0] * len(hobby_indices), hobby_indices], dtype=torch.long)
            edge_rev_likes = torch.tensor([hobby_indices, [0] * len(hobby_indices)], dtype=torch.long)
        else:
            edge_likes = torch.empty((2, 0), dtype=torch.long)
            edge_rev_likes = torch.empty((2, 0), dtype=torch.long)

        # --- 4. Inference ---
        with torch.no_grad():
            x_dict = {
                'user': x_user,
                'major': torch.eye(self.num_majors),
                'hobby': torch.eye(self.num_hobbies)
            }
            
            # Map edges matching training metadata names
            # Note: PyG 'ToUndirected' usually adds 'rev_' prefix or flips names.
            # We fill the dictionary based on what the model expects (self.metadata).
            edge_index_dict = {}
            
            # Populate known edges
            if ('user', 'studies', 'major') in self.metadata[1]:
                edge_index_dict[('user', 'studies', 'major')] = edge_studies
            if ('major', 'rev_studies', 'user') in self.metadata[1]: # Standard PyG reverse name
                edge_index_dict[('major', 'rev_studies', 'user')] = edge_rev_studies
                
            if ('user', 'likes', 'hobby') in self.metadata[1]:
                edge_index_dict[('user', 'likes', 'hobby')] = edge_likes
            if ('hobby', 'rev_likes', 'user') in self.metadata[1]:
                edge_index_dict[('hobby', 'rev_likes', 'user')] = edge_rev_likes

            # Fill any remaining keys with empty tensors to prevent crash
            for key in self.metadata[1]:
                if key not in edge_index_dict:
                    edge_index_dict[key] = torch.empty((2, 0), dtype=torch.long)

            out_dict = self.model(x_dict, edge_index_dict)
            return out_dict['user'].numpy()

    def find_friends(self, user_data):
        # 1. Vectorize
        query_vector = self._process_new_user_features(user_data)
        
        # 2. KNN Retrieval
        distances, indices = self.knn.kneighbors(query_vector, n_neighbors=100)
        
        # 3. Target Data for Ranking
        my_hobbies = [str(h).lower().strip() for h in user_data.get('hobbies', [])]
        my_age = user_data.get('age', 20)
        
        candidates = []
        for rank, neighbor_idx in enumerate(indices[0]):
            friend = self.df.iloc[neighbor_idx]
            
            # --- Robust Parsing ---
            try:
                h_raw = friend['Hobbies']
                if isinstance(h_raw, list): f_hobbies_list = h_raw
                elif isinstance(h_raw, str):
                    if h_raw.startswith("[") and h_raw.endswith("]"):
                        f_hobbies_list = ast.literal_eval(h_raw)
                    elif "," in h_raw:
                        f_hobbies_list = h_raw.split(',')
                    else:
                        f_hobbies_list = [h_raw]
                else: f_hobbies_list = []
                
                f_hobbies_clean = [str(h).lower().strip() for h in f_hobbies_list]
            except:
                f_hobbies_clean = []
            
            # --- Soft Matching ---
            shared_hobbies = []
            for mh in my_hobbies:
                for fh in f_hobbies_clean:
                    if mh in fh or fh in mh: 
                        if fh not in shared_hobbies: shared_hobbies.append(fh)

            gnn_score = (1 - distances[0][rank]) * 10.0
            
            # SCORING: Hobbies > Vibe > Age
            hobby_score = len(shared_hobbies) * 20.0
            
            friend_age = friend['Age']
            age_penalty = abs(my_age - friend_age) * 0.5
            
            major_bonus = 1.0 if user_data.get('major') == friend['Major'] else 0.0
            
            final_score = gnn_score + hobby_score + major_bonus - age_penalty
            
            candidates.append({
                'id': int(neighbor_idx),
                'name': str(friend['Name']),
                'major': str(friend['Major']),
                'age': int(friend_age),
                'shared_hobbies': shared_hobbies, 
                'match_score': float(round(final_score, 2)),
                'vibe_match': float(round(gnn_score * 10, 1))
            })

        # 4. Sort and Return
        candidates.sort(key=lambda x: x['match_score'], reverse=True)
        return candidates[:5]

engine = FriendFinderEngine()
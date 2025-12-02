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
# 1. MODEL DEFINITION
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
        
        # --- FIX: Uncomment this so architecture matches saved weights ---
        # We removed the 'if' check to fix TraceError.
        # We rely on Dummy Data in the engine to prevent empty-batch crashes.
        x = self.bn(x) 
        
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return F.normalize(x, p=2, dim=-1)

# ==========================================================
# 2. ENGINE CLASS
# ==========================================================
class StudyBuddyEngine:
    def __init__(self, model_dir="model_files"):
        print("Loading System Artifacts...")
        self.device = torch.device('cpu') 

        # A. Load Artifacts
        try:
            # Construct absolute path to ensure we find the file
            base_path = os.path.dirname(os.path.abspath(__file__))
            full_model_dir = os.path.join(base_path, model_dir)
            
            with open(os.path.join(full_model_dir, "api_artifacts.pkl"), "rb") as f:
                self.artifacts = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find api_artifacts.pkl in {full_model_dir}")
            
        self.df = self.artifacts['df_search']
        self.scaler = self.artifacts['scaler']
        self.metadata = self.artifacts['metadata']
        
        # Load Maps
        self.course_map = self.artifacts['course_map']
        self.slot_map = self.artifacts['slot_map']
        self.hobby_map = self.artifacts.get('hobby_map', {})
        self.major_map = self.artifacts.get('major_map', {})

        # Counts for Identity Matrices
        self.num_majors = len(self.artifacts['major_map'])
        self.num_hobbies = len(self.artifacts['hobby_map'])
        self.num_courses = len(self.artifacts['course_map'])
        self.num_slots = len(self.artifacts['slot_map'])
        
        # B. Load Search Index
        try:
            self.db_vectors = np.load(os.path.join(full_model_dir, "final_user_vectors.npy"))
        except:
            raise FileNotFoundError(f"Could not find final_user_vectors.npy")
        
        self.knn = NearestNeighbors(n_neighbors=50, metric='cosine')
        self.knn.fit(self.db_vectors)

        print(" Loading MPNet & GNN...")
        self.text_encoder = SentenceTransformer('all-mpnet-base-v2')
        
        self.model = SocialGNN(hidden_channels=256, out_channels=64)
        self.model = to_hetero(self.model, self.metadata, aggr='sum')
        
        try:
            state = torch.load(os.path.join(full_model_dir, "gnn_state_dict.pth"), map_location=self.device)
            self.model.load_state_dict(state)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Error loading GNN model: {e}")
        
        print("Study Buddy Engine Ready!")

    def _process_new_user_features(self, user_data):
        # 1. Feature Vector
        full_text = f"{user_data.get('unique_quality', '')}. {user_data.get('story', '')}"
        text_emb = self.text_encoder.encode([full_text])
        
        y_map = {'Freshman': 1, 'Sophomore': 2, 'Junior': 3, 'Senior': 4, 'Graduate': 5, 'PhD': 6}
        y_num = y_map.get(user_data.get('year'), 1)
        # Default age/gpa if missing
        age = user_data.get('age', 20)
        gpa = user_data.get('gpa', 3.0)
        num_feats = self.scaler.transform([[age, gpa, y_num]])
        
        sex_val = 1 if user_data.get('sex') == 'Female' else 0
        sex_feat = np.array([[sex_val]])
        
        raw_features = np.hstack([text_emb, num_feats, sex_feat])
        x_user = torch.tensor(raw_features, dtype=torch.float)

        # 2. Map Inputs to IDs
        course_indices = []
        for c in user_data.get('courses', []):
            if c in self.course_map:
                course_indices.append(self.course_map[c])
        
        slot_indices = []
        for s in user_data.get('free_slots', []):
            if s in self.slot_map:
                slot_indices.append(self.slot_map[s])

        major_idx = self.major_map.get(user_data.get('major'), -1)

        # 3. Build Edges (LongTensor)
        if course_indices:
            edge_enrolled = torch.tensor([
                [0] * len(course_indices), 
                course_indices
            ], dtype=torch.long)
        else:
            edge_enrolled = torch.empty((2, 0), dtype=torch.long)

        if slot_indices:
            edge_free = torch.tensor([
                [0] * len(slot_indices),
                slot_indices
            ], dtype=torch.long)
        else:
            edge_free = torch.empty((2, 0), dtype=torch.long)

        if major_idx != -1:
            edge_studies = torch.tensor([[0], [major_idx]], dtype=torch.long)
        else:
            edge_studies = torch.empty((2, 0), dtype=torch.long)

        # 4. Run Inference
        with torch.no_grad():
            # Pass FULL Identity Matrices to prevent Index Out of Bounds
            x_dict = {
                'user': x_user,
                'major': torch.eye(self.num_majors),
                'hobby': torch.eye(self.num_hobbies),
                'course': torch.eye(self.num_courses),
                'timeslot': torch.eye(self.num_slots)
            }
            
            edge_index_dict = {
                ('user', 'studies', 'major'): edge_studies,
                ('user', 'enrolled_in', 'course'): edge_enrolled,
                ('user', 'is_free_at', 'timeslot'): edge_free,
                ('user', 'likes', 'hobby'): torch.empty((2, 0), dtype=torch.long),
            }
            
            # Fill missing edge types
            for key in self.metadata[1]:
                if key not in edge_index_dict:
                    edge_index_dict[key] = torch.empty((2, 0), dtype=torch.long)

            out_dict = self.model(x_dict, edge_index_dict)
            user_embedding = out_dict['user'].numpy()
            
        return user_embedding

    def recommend_for_new_student(self, user_data):
        query_vector = self._process_new_user_features(user_data)
        distances, indices = self.knn.kneighbors(query_vector)
        
        my_courses = set(user_data.get('courses', []))
        my_slots = set(user_data.get('free_slots', []))
        
        candidates = []
        for rank, neighbor_idx in enumerate(indices[0]):
            friend = self.df.iloc[neighbor_idx]
            try:
                c_str = friend['Courses'] if pd.notna(friend['Courses']) else "[]"
                s_str = friend['Free_Slots'] if pd.notna(friend['Free_Slots']) else "[]"
                
                f_courses = set(ast.literal_eval(c_str)) if isinstance(c_str, str) else set(c_str)
                f_slots = set(ast.literal_eval(s_str)) if isinstance(s_str, str) else set(s_str)
            except:
                f_courses, f_slots = set(), set()
            
            shared_courses = list(my_courses.intersection(f_courses))
            shared_slots = list(my_slots.intersection(f_slots))
            gnn_score = (1 - distances[0][rank])
            
            final_score = (len(shared_courses) * 10.0) + \
                          (min(len(shared_slots) * 0.05, 3.0)) + \
                          (gnn_score * 2.0)
            
            # --- FIX IS HERE: Explicitly convert numpy types to Python types ---
            candidates.append({
                'id': int(neighbor_idx), # Convert numpy.int64 -> int
                'name': str(friend['Name']),
                'major': str(friend['Major']),
                'shared_classes': list(shared_courses),
                'shared_slots_count': int(len(shared_slots)),
                'match_score': float(round(final_score, 2)), # Convert numpy.float32 -> float
                'profile_similarity': float(round(gnn_score * 100, 1)) # Convert numpy.float32 -> float
            })

        candidates.sort(key=lambda x: x['match_score'], reverse=True)
        return candidates[:5]

# Lazy Load
engine = StudyBuddyEngine()
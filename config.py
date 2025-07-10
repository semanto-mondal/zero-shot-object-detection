class Config:
    #class_list = ["dog","cow", "horse", "sheep","zebra", "giraffe","pig", "goat", "donkey","panda"]
    #seen_classes = ["cow", "horse", "sheep","zebra", "giraffe","dog"]
    class_list = ["microwave", "oven", "refrigerator", "toaster", "blender", "dishwasher"]
    seen_classes = ["microwave", "oven", "refrigerator", "toaster"]
    unseen_classes = list(set(class_list) - set(seen_classes))
    embedding_dim = 256          # MATCH this to node_feats dimension
    gcn_hidden_dim = 128
    gcn_out_dim = 256           # This can stay 1024 if you want output embeddings to be 1024-dim
    device = "cuda"  # or "cpu"

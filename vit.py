import torch.nn as nn
import torch

class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.

    Args:
        in_channels (int): Number of color channels for the input images.
        patch_size (int): Size of patches to convert input image into.
        embedding_dim (int): Size of embedding to turn image into.
    """
    
    def __init__(self,
                 in_channels:int,
                 patch_size:int,
                 embedding_dim:int):
        super().__init__()

        self.patch_size = patch_size
        # image into patches
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=self.patch_size ,
                                 stride=patch_size,
                                 padding=0)

        # flatten the patch feature maps into a single dimension
        self.flatten = nn.Flatten(start_dim=2,end_dim=3)


    def forward(self, x):
        # check that inputs are the correct shape
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {self.patch_size }"

 
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)

        return x_flattened.permute(0, 2, 1) # adjust [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]
    



class MultiheadSelfAttentionBlock(nn.Module):

    def __init__(self,
                 embedding_dim:int,
                 num_heads:int=8, # Heads from Table 1 for ViT-Base
                 attn_dropout:float=0): # the paper doesn't uses any dropout in MSABlocks
        super().__init__()

        # Norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # MSA layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=attn_dropout,
                                                    batch_first=True) # batch dimension come first


    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(query=x, 
                                             key=x, 
                                             value=x, 
                                             need_weights=False) # we need just the layer outputs
        return attn_output
    



class MLPBlock(nn.Module):

    def __init__(self,
                 embedding_dim:int=64, # Hidden Size D 
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base
                 dropout:float=0.5): # Dropout 
        super().__init__()

        # Norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # Multilayer perceptron (MLP) 
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=mlp_size),
            nn.ReLU(), # ReLU non-linearity 
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, # needs to take same in_features as out_features of layer above
                      out_features=embedding_dim), # take back to embedding_dim
            nn.Dropout(p=dropout) # Dropout applied after every dense layer
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x
    


class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 embedding_dim : int,
                 num_heads : int = 8,
                 attn_dropout:float=0,
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base
                 dropout:float=0.5, # from Table 1 for ViT-Base
    ):
        super().__init__()

        self.MSA = MultiheadSelfAttentionBlock(embedding_dim=embedding_dim)
        self.MLP = MLPBlock(embedding_dim=embedding_dim)

    
    def forward(self,x):
        x = self.MSA(x) + x  # 1st residual
        x = self.MLP(x) + x # 2nd residual
        return x
    


class ViT(nn.Module):
    def __init__(self, 
                 image_size :int = 224 ,
                 in_channels :int = 3,
                 patch_size :int = 16,
                 embedding_dim :int = 768,
                 num_transformer_layers :int = 12,
                 mlp_size :int = 3072,
                 num_heads :int = 12 ,
                 attn_dropout :float = 0.0,
                 mlp_dropout :float = 0.5,
                 embedding_dropout :float = 0.1,
                 num_classes :int =10):

        super().__init__()  

        assert image_size % patch_size == 0 , f"Image size should be divisible bypatch size"

        # Prepare image 
        self.num_patches =  int(image_size**2//patch_size**2)
    

        self.class_embedding = nn.Parameter(data = torch.randn(1,1,embedding_dim),
                                            requires_grad=True)

        self.position_embedding = nn.Parameter(data= torch.randn(1, self.num_patches+1, embedding_dim),
                                               requires_grad=True)
        
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)
    
        # Transformer Encoder multi layers
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                            num_heads=num_heads,
                                                                            mlp_size=mlp_size,
                                                                            dropout=mlp_dropout) for _ in range(num_transformer_layers)])
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )        


    def forward(self, x):

        # batch size
        batch_size = x.shape[0]

        # expand class token embedding to match the batch size
        class_token = self.class_embedding.expand(batch_size, -1, -1) 

        # patch embedding 
        x = self.patch_embedding(x)

        # Concat class embedding and patch embedding 
        x = torch.cat((class_token, x), dim=1)

        # Add position embedding to patch embedding 
        x = self.position_embedding + x

        # embedding dropout 
        x = self.embedding_dropout(x)

        # transformer encoder layers 
        x = self.transformer_encoder(x)

        # 19. Put 0 index logit through classifier (equation 4)
        x = self.classifier(x[:, 0]) # run on each sample in a batch at 0 index

        return x
    


def vit_classifier(image_size= 224, 
                in_channels= 3, 
                patch_size = 16, 
                embedding_dim = 768, 
                num_transformer_layers = 12, 
                mlp_size = 3072, 
                num_heads = 12, 
                attn_dropout = 0.0, 
                mlp_dropout = 0.5, 
                embedding_dropout = 0.1, 
                num_classes = 10):
    
    model = ViT(image_size= image_size, 
                in_channels= in_channels, 
                patch_size = patch_size, 
                embedding_dim = embedding_dim, 
                num_transformer_layers = num_transformer_layers, 
                mlp_size = mlp_size, 
                num_heads = num_heads, 
                attn_dropout = attn_dropout, 
                mlp_dropout = mlp_dropout, 
                embedding_dropout = embedding_dropout, 
                num_classes = num_classes)
    
    return model
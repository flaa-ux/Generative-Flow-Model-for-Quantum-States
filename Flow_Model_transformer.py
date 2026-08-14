import torch 
from torch import nn 
import torch.nn.functional as F
import numpy as np
import math

device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"

"""
Instantiates the building blocks of a transformer
"""

#0) ============================= Zero Embedder ============================================
"""
Since our data set may contain varying qubit sizes, it is reasonable to have an extra module
that zero extends our vector.
"""
class ZeroExtend (nn.Module) :

    def __init__ (self, target_dim : int):
        """
        Zero extends embeds additional zeroes to the end of every single
        vector
        """
        super().__init__()

        self.target_dim = target_dim 
    
    def forward(self, batched_states : torch.tensor) :
        """
        Args : 
        batched_states : quantum states batched with dimension (B, 2** (n+1)) where n = qubit_count
        """
        remaining_dim = -batched_states.shape[-1] + self.target_dim

        if not remaining_dim : 

            return batched_states 

        #prepare zeroes 
        zero_embedding = torch.zeros(batched_states.shape[0], remaining_dim,
                                     device = device)

        res = torch.cat([batched_states, zero_embedding], dim = 1)

        return res
    

#A) ======================================Image Tokenizer======================================

#1) Patchifier -> splits images into patches (for quantum states, simply breaks into smaller vectors)

class Patchifier (nn.Module) : 

    def __init__ (self, image_size : int, 
                  dim_size : int
                  ):
        """
        Breaks down the input vector into patches. Since we are dealing with
        quantum states, simply slice the input vector from shape (batch, ) to
        an array of tokens (batch, N, ). 
        """
        super().__init__()
        self.image_size = image_size
        self.dim_size = dim_size

        if self.image_size % self.dim_size: 
            
            raise ValueError( "dim_size must be a power of 2")

    def forward (self, input_image) : 
        """
        Slice input image into arrays of tokens t, such it has dimension 
        self.dim_size. It should have shape [batch, N', dim]
        """

        #find the number of smaller vectors we need
        row_count = self.image_size // self.dim_size 

        batch_size = input_image.shape[0]

        #return a view of the vector
        res = input_image.view([batch_size, row_count, self.dim_size])

        return res

#2) Patch Embedder -> Increases the dim
class PatchEmbed (nn.Module): 

    def __init__ (self, initial_dim : int,  
                  target_dim : int) : 
        """
        trains a patch embedder (this can simply be a single layer)
        neuron. Takes in a batch of tokens, and apply a linear transform
        on them, takes in dim (batch_size, row_count, self.dim_size)
        """
        super().__init__()

        self.linear_layer = nn.Linear(initial_dim, target_dim,)

    def forward (self, input_batch) : 

        #flatten the input_batch on the zeroth dimension
        return self.linear_layer(input_batch)
    
#3) UnifiedPatchEmbedder
class UnifiedPatchEmbedder (nn.Module) : 

    def __init__ (self, image_size : int, 
                  initial_patch_channel : int, 
                  target_channel : int, 
                  ) : 
        
        super().__init__()
        self.patcher = Patchifier(image_size, 
                                  initial_patch_channel)
        
        self.tokenizer = PatchEmbed(initial_patch_channel, 
                                    target_channel)
    
    def forward (self, input_image) : 

        patched_vector = self.patcher(input_image)

        tokens = self.tokenizer(patched_vector)

        return tokens
    
#B) ====================================== Conditional Embeddings ======================================

#1) Time conditional - Fourier Embedding -> set as learnable parameter 

class LearnedFourierEmbedder (nn.Module) : 

    """
    Performs fourier embedding on the labels for better details
    scalar -> high dimensional labels
    """
    def __init__ (self, half_dim : int): 
        """
        Instantiate the weights for the fourier embedding

        batch of inputs -> batch of ([cos 2pi omega * input .... sin 2pi omega * input ...])
        """
        super().__init__()
        self.dim = half_dim

        #initialize the frequency weights
        self.omega = nn.Parameter(torch.randn(1, half_dim,))     
                      
        #self.omega creates weight of shape (1, half_dim) to enable broadcasting

    def forward(self, input_batch : torch.tensor, 
                drop = False ) :
        """
        Computes the fourier change
        """
        if drop : 
            res = -1 * torch.ones_like(input_batch,
                                       device = device) * torch.ones_like(self.omega,
                                                                          device = device)
            
            return torch.cat([res, res], dim = 1)
        
        weights = 2 * torch.pi * self.omega * input_batch
        sin_term, cos_term = torch.sin(weights), torch.cos(weights)

        #normalization factor sqrt(2/d)
        res = torch.sqrt(2/ torch.tensor([self.dim]
                                         , device= device)) * torch.cat([cos_term, sin_term], 
                                                                        dim = 1)

        return res 

#2) SRE Label Embedder -> Also a learnable parameter, consisting of a fourier embedder and tokenizer
class SRELabelEmbedder (nn.Module) : 

    def __init__ (self, fourier_half_dim : int, 
                    projection_dim : int) : 
        """
        Performs the following embedding

        Batched scalar SRE [Batch, 1] -> fourier embedding [Batch, 2 * half_dim]
        -> Linear Projection [Batch, target_dim] -> Reshape [Batch, 1, target_dim,]

        """
        super().__init__()
        self.fourier_embedder = LearnedFourierEmbedder(fourier_half_dim)
        self.linear_transform = nn.Linear (2 * fourier_half_dim, projection_dim,
                                           )
        self.projection_dim = projection_dim

    def forward (self, batched_scalar, 
                 drop = False) : 

        batch_size = batched_scalar.shape[0]

        if drop : 
            ones = -1.0 * torch.ones(batch_size, 1, self.projection_dim, 
                                     device= device,
                                     dtype = torch.float32)
            return ones

        #perform fourier embedding
        fourier_res = self.fourier_embedder(batched_scalar)

        #perform a linear projection of some sort
        token = self.linear_transform(fourier_res).view([batch_size, 1, self.projection_dim])

        return token 

#3) Hot label embedder 
class HotLabelEmbedder (nn.Module) : 
    """
    Performs a hot label embedding on distinct classes to get better dimensionality
    for model training
    """
    def __init__ (self, class_count : int) : 
        super().__init__()
        self.flatten = nn.Flatten()

        #class_count must be token dimension 
        self.class_count = class_count 
    
    def forward(self, input_batch, drop_bool = False) : 
        """
        given an input batch of B scalars, return the associated embedding that
        corresponds to the class. Input must be shape (B,1)

        """
        if drop_bool : 
            res = -1.0 * torch.ones_like(input_batch) * torch.ones(self.class_count,
                                                                 device = device)
            
            # return a sequence of tokenized hot label encoding
            return res.view([input_batch.shape[0], 1, self.class_count])
        
        #otherwise creates a hot label encoding for each class 
        input_batch = input_batch[:, 0]
        res = F.one_hot(input_batch, 
                        num_classes= self.class_count)
        # identity_mat = torch.diag(torch.ones(self.class_count)) #shape (class_count, class_count)

        return res.view([input_batch.shape[0], 1, self.class_count])
    
#C) ============================ Time Conditioning Hyperparameters ============================

class TimeConditioningHyperParam (nn.Module) : 

    def __init__ (self, time_dim: int) :
        """
        Creates a function that maps a time embedding of shape (batch, dim) and maps
        it to a vector (gamma, beta) in R^2d. These hyperparameters will be used for
        the adanorm scaling and learning outputs scaling + shifting when the input image is
        feed forwarded through the network
        """ 
        super().__init__()
        self.time_dim = time_dim

        self.linear_stack = nn.Sequential (
            nn.Linear(time_dim, 2 * time_dim), 
            nn.SiLU()
        )

    def forward (self, batched_input_time: torch.tensor) -> tuple:

        #batched_input_time must be fourier embedded time 
        res = self.linear_stack(batched_input_time)

        #slice the result
        gamma, beta = res[:, : self.time_dim].unsqueeze(dim = 1), res[:, self.time_dim :].unsqueeze(dim = 1)

        return gamma, beta

class GatingFunction (nn.Module) : 
    
    def __init__ (self, channel_dim : int) : 
        """
        Constructs a single layer MLP that creates a batch that transforms
        [B, d] -> [B, d]. Performs unsqueezing at dim = 1, such that [B, 1, d] to enable
        broadcasting with viT layer outputs that have shape [B,N, d]
        """
        super().__init__()
        self.channel_dim = channel_dim

        self.nn_stack = nn.Sequential(
            nn.Linear (channel_dim, channel_dim),
            nn.SiLU()
        )
    
    def forward (self, input_batch : torch.tensor) : 
        """
        Input batch will be a time tensor with shape [B, d]. 
        """
        res = self.nn_stack (input_batch)

        #unsqueeze res
        res = torch.unsqueeze(res, dim = 1)         #shape [B, d, 1]

        return res

class AdaNorm (nn.Module) : 
    
    def __init__ (self, channel_dim : int,
                  ) : 
        """
        Performs scaled layer normalizations across the token dimension
        """
        super().__init__()

        #normalize over the token features
        self.layer_norm = nn.LayerNorm(channel_dim)

        self.time_conditioner = TimeConditioningHyperParam(time_dim=channel_dim)
    
    def forward (self, input_tokens : torch.tensor,
                 batched_time : torch.tensor) :
        """
        Returns a scaled value (1 + gamma) * Norm(input) + beta
        """
        gamma, beta = self.time_conditioner(batched_time)

        normal_input = self.layer_norm(input_tokens)

        res = (1 + gamma) * normal_input + beta

        return res

#d) ========================== Transformer Layer building block =============================================

class PositionWiseFeedForward (nn.Module) : 

    def __init__ (self, channel_dim: int) : 
        """
        Performs a double layer transformation with a ReLU activation.
        Expects batched token inputs of shape (B, sequence (N), dim)
        """
        super().__init__()
        self.channel_dim = channel_dim 

        self.mlp = nn.Sequential(
            nn.Linear(channel_dim, 4 * channel_dim),
            nn.ReLU(), 
            nn.Linear(4 * channel_dim, channel_dim)
        )
    
    def forward (self, input_token : torch.tensor) : 

        res = self.mlp(input_token)

        return res

# class HeadAttention (nn.Module) : 

#     def __init__ (self, token_dim : int, 
#                   head_count : int,
#                   ) : 
#         """
#         Computes the attention value for each head using a dot product 
#         likeness function  
#         """
#         assert token_dim % head_count == 0 

#         head_dim = token_dim // head_count
#         super().__init__()

#         #initialize token projectors
#         self.query_projector =  nn.Linear(token_dim, head_dim)
#         self.key_projector = nn.Linear(token_dim, head_dim)
#         self.value_projector = nn.Linear(token_dim, head_dim)

#         self.head_dim = head_dim 
#         self.token_dim = token_dim

#     def forward (self, tokenized_input_x : torch.tensor,
#                     tokenized_input_z : torch.tensor) -> torch.tensor : 
#         """
#         Given two tokenized input batches, project tokens into the dedicated QKV
#         space. Supports cross / self attention depending on what tensor is fed into
#         the argument : tokenized_input_z.

#         We will assume that th
#         """
#         query_tensor = self.query_projector(tokenized_input_x)
#         key_tensor = torch.transpose(self.key_projector(tokenized_input_z), 
#                                      dim0 = 1, dim1 = 2)
#         value_tensor = self.value_projector(tokenized_input_z)

#         #compute the torchmax value with of QK^T [B, N, dh] times [B, dh, M]
#         normalized_attention_dot_prod = query_tensor @ key_tensor * 1/math.sqrt(self.head_dim)

#         #compute the softmax using torch.softmax on dimension 2 (normalized sum for all keys (column) = 1)
#         probability = torch.softmax(normalized_attention_dot_prod,
#                                     dim = 2)
        
#         res = probability @ value_tensor 

#         return res 
    
# class MultiHeadAttention(nn.Module):
#     """
#     Combines multiple HeadAttention modules sequentially.
#     """
#     def __init__(self, token_dim: int, head_count: int):
#         super().__init__()
#         assert token_dim % head_count == 0, "token_dim must be divisible by head_count"
        
#         self.head_count = head_count

#         # IMPORTANT: Use nn.ModuleList, not a standard Python list.
#         # If you use a regular list, PyTorch will not register the parameters
#         # of the HeadAttention modules, and they will not be updated during training.
#         self.heads = nn.ModuleList([
#             HeadAttention(token_dim=token_dim, head_count=head_count) 
#             for _ in range(head_count)
#         ])
        
#         # Final output projection W_O
#         self.linear_projector = nn.Linear(token_dim, token_dim)


#     def forward(self, tokenized_input_x: torch.Tensor, tokenized_input_z: torch.Tensor) -> torch.Tensor:
#         """
#         Feed the inputs to all heads sequentially, concatenate the results, 
#         and apply the final linear projection.
#         """
#         head_outputs = []
        
#         # 1. Loop through each head and compute attention independently
#         for head in self.heads:
#             # Output of a single head has shape: (Batch, Seq_Len, head_dim)
#             h_out = head(tokenized_input_x, tokenized_input_z)
#             head_outputs.append(h_out)
            
#         # 2. Concatenate the outputs from all heads along the last dimension (head_dim)
#         # Shape becomes: (Batch, Seq_Len, token_dim)
#         concatenated_output = torch.cat(head_outputs, dim=-1)
        
#         # 3. Apply the final output projection W_O
#         final_output = self.linear_projector(concatenated_output)
        
#         return final_output

class MultiHeadAttention(nn.Module):
    def __init__(self, token_dim: int, head_count: int):
        super().__init__()
        assert token_dim % head_count == 0, "token_dim must be divisible by head_count"
        
        self.head_count = head_count
        self.head_dim = token_dim // head_count
        
        # Project all heads simultaneously!
        self.q_proj = nn.Linear(token_dim, token_dim)
        self.k_proj = nn.Linear(token_dim, token_dim)
        self.v_proj = nn.Linear(token_dim, token_dim)
        
        # Final output projection
        self.out_proj = nn.Linear(token_dim, token_dim)

    def forward(self, tokenized_input_x: torch.Tensor, tokenized_input_z: torch.Tensor) -> torch.Tensor:
        B, N_x, D = tokenized_input_x.shape
        _, N_z, _ = tokenized_input_z.shape
        
        # 1. Project Q, K, V for ALL heads at once
        q = self.q_proj(tokenized_input_x)
        k = self.k_proj(tokenized_input_z)
        v = self.v_proj(tokenized_input_z)
        
        # 2. Reshape and transpose to isolate heads: (Batch, Heads, Sequence, Head_Dim)
        q = q.view(B, N_x, self.head_count, self.head_dim).transpose(1, 2)
        k = k.view(B, N_z, self.head_count, self.head_dim).transpose(1, 2)
        v = v.view(B, N_z, self.head_count, self.head_dim).transpose(1, 2)
        
        # 3. Compute attention using PyTorch's optimized C++ kernel
        # This replaces the manual softmax(Q @ K.T) @ V
        attn_output = F.scaled_dot_product_attention(q, k, v)
        
        # 4. Reshape back to flat tokens: (Batch, Sequence, Token_Dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N_x, D)
        
        # 5. Final projection
        return self.out_proj(attn_output)
    
class VisionTransformerLayer (nn.Module) : 

    def __init__ (self, channel_dim : int, 
                  head_count : int, 
                  ):
        """
        Implements the vision transformer layer with scaled and shiftable 
        multihead attention values.

        Attention is defined as the softmax of the likeness of the query and key matrices

        """
        super().__init__()

        self.channel_dim = channel_dim 
        self.head_count = head_count 

        # we will have to initialize the hyperparams modules and gating functions
        self.gate_func_self = GatingFunction(channel_dim=channel_dim)
        self.gate_func_cross = GatingFunction(channel_dim= channel_dim)
        self.gate_func_MLP = GatingFunction(channel_dim= channel_dim)

        self.ada_norm_self = AdaNorm(channel_dim= channel_dim)
        self.ada_norm_cross = AdaNorm(channel_dim= channel_dim)
        self.ada_norm_MLP = AdaNorm(channel_dim= channel_dim)

        #initialize the multiattention modules
        self.multihead_self = MultiHeadAttention(
            token_dim= channel_dim,
            head_count= head_count)

        self.multihead_cross = MultiHeadAttention(
            token_dim= channel_dim,
            head_count= head_count)

        self.mlp = PositionWiseFeedForward(channel_dim= channel_dim)
    
    def forward (self, tokenized_input : torch.tensor ,
                 tokenized_labels : torch.tensor, 
                 embedded_time : torch.tensor) :
        """
        The expected workflow for a single layer can be understood as follows :

        normalize tokens :  ada_norm(tokenized_input) with scaling hyperparams
        1. layer norm, perform self multi head attention, feedforward
        2. layer norm, perform cross multi head attention, feedforward
        3. layern norm, MLP layer
        """
        
        #generate the scaling parameter 

        #perform first process (self-attention)
        tokenized_input = self.ada_norm_self(input_tokens = tokenized_input, 
                                        batched_time = embedded_time)
        
        gating_scale = self.gate_func_self(embedded_time)

        tokenized_input = tokenized_input + gating_scale * self.multihead_self(tokenized_input_x = tokenized_input,
                                                                tokenized_input_z = tokenized_input)
        
        #perform second process (prompt-attention)
        tokenized_input = self.ada_norm_cross (input_tokens = tokenized_input,
                                               batched_time = embedded_time)
        
        gating_scale = self.gate_func_cross(embedded_time)

        tokenized_input = tokenized_input + gating_scale * self.multihead_cross(tokenized_input_x = tokenized_input,
                                                                                tokenized_input_z = tokenized_labels)
        
        #perform last MLP (last feedforward)

        tokenized_input = self.ada_norm_MLP(input_tokens = tokenized_input, 
                                            batched_time = embedded_time)

        gating_scale = self.gate_func_MLP(embedded_time)

        tokenized_input= tokenized_input + gating_scale * self.mlp(tokenized_input) 

        return  tokenized_input

#e) ========================== Vision Transformer Block ========================================

class VisionTransformerBlock (nn.Module) : 

    def __init__ (self, depth : int, 
                  channel_dim : int, 
                  head_count : int, 
                  sequence_length : int) : 
        """
        Construct N layers of vision transformers.

        Applies positional embedding within the layer by standard weight additions
        """
        super().__init__()

        #prepare depth layers of transformer layers
        self.vision_layers = nn.ModuleList(
            VisionTransformerLayer(channel_dim=
                                   channel_dim, head_count= head_count) for _ in
                                   range (depth)
        )

        #prepare random weights for positional embedding 
        self.pos_embed = nn.Parameter(torch.randn(sequence_length, channel_dim))

    def forward(self, batched_input_tokens : torch.tensor,
                batched_label_tokens : torch.tensor, 
                embedded_time : torch.tensor) :
        """
        Args :

        batched_input_tokens : B, N, d
        batched_input_labels : B, S, d
        embedded_time : B, d
        """ 
        #each batch should be labeled with a different positional encoding
        batched_input_tokens = batched_input_tokens + self.pos_embed.unsqueeze(dim = 0)

        for layer in self.vision_layers : 
            batched_input_tokens = layer(tokenized_input = batched_input_tokens,
                        tokenized_labels = batched_label_tokens, 
                        embedded_time = embedded_time)
        
        return batched_input_tokens

#f) ========================== Depatchifier Block ============================================
class Depatchifier (nn.Module) : 

    def __init__ (self, state_dim : int, 
                  token_dim : int,
                  sequence_length : int, 
                  ) : 
        """
        Performs layer normalization on token sequences
        """
        super().__init__()

        self.layer_norm = nn.LayerNorm(token_dim)

        #projection dim must be such that proj_dim * sequence_length = state_dim
        proj_dim = state_dim // sequence_length

        self.linear = nn.Linear(token_dim, proj_dim)

        self.flatten = nn.Flatten()


    def forward (self, batched_token_sequences) : 
        """
        depatchifier simply applies a normalization, linear transform,
        and finally dimension flattenin
        """ 
        normalized_tokens = self.layer_norm(batched_token_sequences)

        projected_sequences = self.linear(normalized_tokens)

        #flatten the projected sequences 
        res = self.flatten(projected_sequences)

        return res

#f) ========================== Full Transformer Block ========================================
class VisionTransformerFullModel(nn.Module) : 

    def __init__ (self,
                  state_size : int,
                  state_patch_dim: int, 
                  token_dim : int, 
                  SRE_label_half_dim : int, 
                  multi_head_count : int,
                  transformer_depth : int) : 
        """
        Implements the vision transformer architecture as the neural network 
        of the flow model system. The general workflow can be understood as follows : 

        1) Prepare embedding for input state, SRE label, time
        2) patch state into valid tokens; repeat for SRE labels
        3) time will also be fourier embedded such that it has the same dimension as the token 
        4) Pass through full transformer block
        5) Linear Transform and reshape back to batch of valid vector fields in concatenated representation
        """

        #ensure that the dimension of the token is even 
        assert token_dim % 2 == 0

        super().__init__()
        self.flatten = nn.Flatten()

        # the following must hold : state_patch_dim * sequence_len = state_size
        sequence_len = state_size // state_patch_dim

        #initialize zero extend for unfixed qubit sizes 
        self.zero_extend = ZeroExtend(
            target_dim= state_size
        )

        #intiliaze patching function 
        self.state_patcher = UnifiedPatchEmbedder(
            image_size  = state_size, 
            initial_patch_channel= state_patch_dim, 
            target_channel= token_dim
        )

        self.SRE_embedder = SRELabelEmbedder(
            fourier_half_dim= SRE_label_half_dim, 
            projection_dim= token_dim 
        )

        self.hot_label_embedder = HotLabelEmbedder(
            class_count = token_dim
        )

        self.time_embedder = LearnedFourierEmbedder(
            half_dim = token_dim // 2
        )

        #prepare the transformer block 
        # print("token_dim:",token_dim)
        self.ViT = VisionTransformerBlock(
            depth = transformer_depth,
            channel_dim = token_dim, 
            head_count = multi_head_count, 
            sequence_length = sequence_len
        )

        #prepare the depatchify block
        self.state_depatchify = Depatchifier(
            state_dim = state_size,
            token_dim= token_dim, 
            sequence_length= sequence_len
        )
    
    def forward (self, batched_states : torch.tensor,
                 batched_labels : torch.tensor, 
                 batched_time : torch.tensor, 
                 drop_bool = False, 
                 mixed_qubit = False
                 ):
        # perform zero extension 
        batched_states = self.zero_extend(batched_states)

        # Given a batch of quantum states, perform tokenization
        tokenized_states = self.state_patcher(input_image = 
                                              batched_states)
        
        tokenized_labels = 0

        if mixed_qubit : 

            SRE_labels = batched_labels[:, 0].view([-1, 1])
            qubit_labels = batched_labels[:, 1].view([-1, 1]).long()

            #perform separate embedding and concatenate along sequence dimension
            tokenized_SRE = self.SRE_embedder(batched_scalar = SRE_labels,
                                              drop = drop_bool)
            
            tokenized_qubit = self.hot_label_embedder(qubit_labels,
                                                      drop_bool = drop_bool).long()
            
            tokenized_labels = torch.cat([tokenized_SRE, tokenized_qubit], dim = 1)

        else : 
            tokenized_labels = self.SRE_embedder(batched_scalar = batched_labels,
                                                drop = drop_bool)
        
        # #uncomment to check 
        # print(drop_bool, '\n', tokenized_labels)

        tokenized_time = self.time_embedder(input_batch = batched_time)

        #pass through the vision transformer 
        learned_outputs = self.ViT(batched_input_tokens = tokenized_states, 
                                   batched_label_tokens = tokenized_labels,
                                   embedded_time = tokenized_time)
        
        learned_outputs = self.state_depatchify(
            batched_token_sequences = learned_outputs
        )

        return learned_outputs


if __name__ == "__main__" : 

    # # ================================ Test Zero Embedder ===========================

    # batched_states = torch.randn(20, 64, device = device)

    # zero_embedder = ZeroExtend(target_dim= 64).to(device)

    # out = zero_embedder(batched_states)

    # print(out, out.shape)

    # # ===============================================================================

    #========================= Test Hot Label Embedder ==============================
    random_scalar = np.random.choice(range(2,7), p = 0.2 * np.ones(5))
    random_scalar_2 = np.random.choice(range(2,7), p = 0.2 * np.ones(5))
    random_scalar_3 = np.random.choice(range(2,7), p = 0.2 * np.ones(5))
    
    print(random_scalar, random_scalar_2, random_scalar_3)
    input = torch.stack([torch.tensor([random_scalar]), 
                         torch.tensor([random_scalar_2]),
                         torch.tensor([random_scalar_3])]).to(device)

    print(input.shape)
    hot_embedding = HotLabelEmbedder(15).to(device)

    res = hot_embedding(input, drop_bool = True)

    print (res, '\n', res.shape)
    # ===============================================================================

    # #============== Test Patchifier ==============
    # tensor_1 = torch.tensor([1,2,1,2], device= device, dtype = torch.float32)
    # tensor_2 = torch.tensor([3,2,3,4], device= device, dtype = torch.float32)
    # tensor_3 = torch.tensor([1,2,5,6], device= device, dtype = torch.float32)
    # tensor_4 = torch.tensor([1,7, 8,4], device= device, dtype = torch.float32)

    # batched_tensor = torch.stack([tensor_1, tensor_2, tensor_3,
    #                               tensor_4])
    
    # patcher = Patchifier(tensor_1.shape[0], 
    #                      dim_size = 2).to(device)
    
    # patches = patcher(batched_tensor)

    # tokenizer = PatchEmbed(2, 6).to(device)
    # tokens = tokenizer(patches)

    # print(tokens)

    # tokenizer = UnifiedPatchEmbedder(4, 2, 6).to(device)
    # print(tokenizer(patches))
    # # #=============================================

    # #============== Test Fourier Embedder ==============

    # fourier_embedder = LearnedFourierEmbedder(5).to(device)
    # time_1 = torch.tensor([0.4], device = device)
    # time_2 = torch.tensor([0.2], device = device)
    # time_3 = torch.tensor([0.6], device = device)

    # times = torch.stack([time_1, time_2, time_3])

    # fourier_embedding = fourier_embedder(times)

    # print(fourier_embedding)

    # # ===================================================

    # # ============= Test SRE Label Embedder =============

    # sre_label_embedder = SRELabelEmbedder(fourier_half_dim= 5,
    #                                       projection_dim= 12).to(device)
    
    # sre_scalar_1 = torch.tensor([2.01], device = device)
    # sre_scalar_2 = torch.tensor([1.02], device = device)
    # sre_scalar_3 = torch.tensor([0.51], device = device)

    # sre = torch.stack([sre_scalar_1, sre_scalar_2, sre_scalar_3])

    # sre_embedding_dropped = sre_label_embedder(sre, drop = True)
    # sre_embedding = sre_label_embedder(sre)

    # print(sre_embedding)
    # print(sre_embedding_dropped)

    # # ======================================================


    # #=============== Test Gating Function =====================

    # input_time = torch.randn(4, 5, device= device)
    # print(input_time.shape)

    # gating_func = GatingFunction(5).to(device)

    # res = gating_func(input_time)

    # print(res.shape)

    # #============================================================

    #=============== Test Gating Function =====================

    # input_time = torch.randn(4, 5, device= device)
    # print(input_time.shape)

    # gating_func = TimeConditioningHyperParam(5).to(device)

    # scale, shift = gating_func(input_time)

    # print(scale.shape, shift.shape)

    #============================================================

    # #============ Test feedforward position wise ================
    # model_MLP = PositionWiseFeedForward(10).to(device)
    # input_token = torch.randn(2, 4, 10, device = device)

    # res = model_MLP(input_token)

    # print(res)
    # print(res.shape)

    # #=============================================================

    #=============== Test single head attention ==================
    # head_attention = HeadAttention(10, head_count= 2).to(device)
    # input_token_x = torch.randn(5, 3, 10, device = device)
    # input_token_z = torch.randn(5, 3, 10, device = device)

    # res = head_attention(input_token_x, input_token_z)

    # print (res, '\n', res.shape)


    # =================================================================

    # # ==========================Test the sequential implementation==========================
    # batch_size = 2
    # seq_len = 5
    # token_dim = 64
    # head_count = 32

    # x = torch.randn(batch_size, seq_len, token_dim)
    # z = torch.randn(batch_size, seq_len, token_dim)

    # mha = MultiHeadAttention(token_dim=token_dim, head_count=head_count)
    
    # out = mha(tokenized_input_x=x, tokenized_input_z=z)
    # print(f"Input shape: {x.shape}")
    # print(f"Output shape: {out.shape}") # Should be (2, 5, 64)

    # #===========================================================================================
    
    # #========================== ADA Norm testing ==============================================

    # batched_tokens = torch.randn(20, 4, 10, device = device)
    # batched_time = torch.rand(20, 10, device= device)
    # ada_norm = AdaNorm(channel_dim = 10).to(device)

    # res = ada_norm(batched_tokens,
    #                batched_time)

    # print(res, '\n', res.shape)
    # #===========================================================================================

    #======================== Vision Transformer Layer testing ===========================

    # batched_tokens = torch.randn(20, 4, 64, device = device)
    # batched_labels = torch.randn(20, 6, 64, device = device)
    # batched_time = torch.rand(20, 64, device= device)
    
    # ViTLayer = VisionTransformerLayer(channel_dim= 64,
    #                                   head_count= 16 
    #                                 ).to(device)

    # res = ViTLayer(batched_tokens,
    #                batched_labels,
    #                batched_time)

    # print(res, '\n', res.shape)
    #===========================================================================================

    # #================================= Vision Transformer Block ================================

    # batched_tokens = torch.randn(20, 4, 64, device = device)
    # batched_labels = torch.randn(20, 6, 64, device = device)
    # batched_time = torch.rand(20, 64, device= device)
    
    # ViTBlock = VisionTransformerBlock(depth = 12, 
    #                                   channel_dim= 64,
    #                                   head_count = 8,
    #                                   sequence_length= 4).to(device)

    # res =  ViTBlock(batched_tokens, batched_labels, batched_time )

    # print(res, '\n', res.shape)
    # #===========================================================================================

    # #================================= Depatchifier =============================================

    # batched_output = torch.randn(20, 4, 13, device = device)
    
    # depatched = Depatchifier(state_dim = 64,
    #                          token_dim = 13, 
    #                          sequence_length= 4).to(device)

    # res = depatched(batched_output)

    # print(res, '\n', res.shape)
    # #===========================================================================================

    #================================= Vision Transformer Full Model ================================

    batched_quantum_states = torch.randn(20, 128, device= device)
    batched_SRE = torch.randn (20, 1, device = device)
    batched_qubit = torch.randint (2, 7, (20, 1), device= device).long()

    batched_labels = torch.cat([batched_SRE, batched_qubit], dim = 1)
    batched_time = torch.rand(20, 1, device = device)
    
    ViTModel = VisionTransformerFullModel(state_size = 256,
                                          state_patch_dim=16,
                                          token_dim = 32,
                                          SRE_label_half_dim=16,
                                          multi_head_count=32,
                                          transformer_depth= 8).to(device)

    res_dropped =  ViTModel(batched_states = batched_quantum_states,
                            batched_labels = batched_labels, 
                            batched_time = batched_time, 
                            drop_bool = True,
                            mixed_qubit = True)

    res = ViTModel(batched_states = batched_quantum_states,
                            batched_labels = batched_labels, 
                            batched_time = batched_time, 
                            drop_bool = False,
                            mixed_qubit = True)
    
    print(f"dropped:", res_dropped.shape, '\n', f"not_dropped:", res.shape)
    #===========================================================================================


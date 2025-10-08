from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..structs import TimeSeriesThDF, BufferList

def index_precompute(cin, cout,            # (N,) int64
                     map_river, map_pixel, # (R,) int64  (unordered OK)
                     map_weight):          # (R,) float32/64

    ###
    ### Part 1: Index precomputation
    ###
    N = cin.shape[0]
    device = cin.device
    map_weight = map_weight.to(device=device)

    # ---- 1. sort mapping by river so we can binary-search ----
    sort_idx   = torch.argsort(map_river)
    map_river  = map_river[sort_idx].to(dtype=torch.int64, device=device, copy=False).contiguous()
    map_pixel  = map_pixel[sort_idx].to(dtype=torch.int64, device=device, copy=False)
    map_weight = map_weight[sort_idx]

    # ---- 2. find contiguous blocks per river ----
    left  = torch.searchsorted(map_river, cin, right=False)
    right = torch.searchsorted(map_river, cin, right=True)
    cnt   = right - left                            # how many pixels per row in `vals`
    keep  = cnt > 0                                 # drop rivers with no pixel mapping

    if not keep.all():
        raise NotImplementedError

    prefix = torch.zeros_like(cnt)
    prefix[1:] = torch.cumsum(cnt[:-1], 0)
    tot = int((prefix + cnt).max().item())          # total number of exploded rows

    # ---- 3. explode rows -> one entry per (pixel, river-row) combination ----
    row_id   = torch.repeat_interleave(torch.arange(N, device=device, dtype=torch.int64), cnt)   # (tot,)
    global_i = torch.arange(tot, device=device, dtype=torch.int64)
    rel_i    = global_i - prefix[row_id]                                                # offset within each river
    map_i    = left[row_id] + rel_i                                                     # index in map_pixel/weight

    pixel  = map_pixel[map_i]      # (tot,) int64
    weight = map_weight[map_i]     # (tot,) float32/64
    c_out  = cout[row_id]          # (tot,) int64

    # ---- 4. pack (pixel, c_out) into a 64-bit key and aggregate ----
    key         = (pixel << 32) | (c_out & 0xffffffff)
    key_s, idx  = torch.sort(key)                       # stable sort so unique_consecutive works
    uniq_key, inverse = torch.unique_consecutive(key_s, return_inverse=True)
    M = uniq_key.numel()

    p_in  = (uniq_key >> 32).to(torch.int64)
    c_out_unique = (uniq_key & 0xffffffff).to(torch.int64)
    return weight, idx, row_id, M, inverse, p_in, c_out_unique

def aggregate(vals, weight, idx, row_id, M, inverse, p_in, c_out_unique):
    ###
    ### Part 2: Actual aggregation
    ###
    N, F = vals.shape
    contrib = vals[row_id] * weight.unsqueeze(1)                                        
    contrib_s   = contrib[idx]
    out = torch.zeros(M, F, dtype=vals.dtype, device=inverse.device)
    out.scatter_add_(0, inverse.unsqueeze(1).expand(-1, F), contrib_s)
    return out

def river_to_pixel_gpu_pt(vals,                # (N, F) float32/64, **must** be on CUDA
                          cin, cout,           # (N,) int64
                          map_river, map_pixel,# (R,) int64  (unordered OK)
                          map_weight):
    (weight, idx, row_id, M, 
     inverse, p_in, c_out_unique) = index_precompute(cin, cout,            
                                                     map_river, map_pixel, 
                                                     map_weight)
    out = aggregate(vals, weight, idx, row_id, M, inverse, p_in, c_out_unique)
    return out, c_out_unique, p_in

class CatchmentInterpolator(nn.Module):
    def __init__(self, g, pixel_df, weight_df):
        """
            weight_df: pd.DataFrame with index values in g nodes and columns pixel_idxs and area_sqm_total
            Maybe here we should assume that weight_df only contains the relevant pixels.
            So that when we interpolate the kernel, we interpolate it only for the needed inputs.
        """
        super().__init__()
        input_pixel_idxs = pixel_df._columns
        output_cat_idxs = g.nodes_idx
        
        weight_subset = weight_df.loc[output_cat_idxs.index] # First only select the relevant rows
        # Assert that only the needed input pixels and output catchments are included in the weight_df
        # This is tested by indexation at the next lines anyway
        
        self.register_buffer("dest_idxs", torch.tensor(output_cat_idxs[weight_subset.index].values, dtype=torch.long))
        self.register_buffer("src_idxs",  torch.tensor(input_pixel_idxs[weight_subset["pixel_idx"]].values, dtype=torch.long))
        self.register_buffer("weights",   torch.tensor(weight_subset["area_sqm_total"].values, dtype=torch.float))
        #self.register_buffer("pix_idxs",   torch.tensor(input_pixel_idxs.values, dtype=torch.float))
        
        self.map_inp = input_pixel_idxs
        self.map_out = output_cat_idxs
        self.n_cats = len(self.map_out)
        self.n_pix = len(self.map_inp)
        
    def interpolate_runoff(self, runoff):
        """
            runoff is expected to be arranged according to the nodes_idx of g?
            what is the difference between map_inp and nodes_idx?
        """
        weighted_x = runoff.values[:, self.src_idxs] * self.weights[None, :, None]  # broadcasts over the time dimension
        out_size = runoff.shape[0], self.n_cats, runoff.shape[-1]
        out = torch.zeros(out_size, 
                          dtype=runoff.dtype, device=runoff.device)
        out.index_add_(1, self.dest_idxs, weighted_x)
        return TimeSeriesThDF(out, 
                              columns=self.map_out.index, 
                              index=runoff._index)
    
    def interpolate_kernel(self, irfs_agg, coords):
        """
            Here, coords_pixel are aligned to pix_idxs ordering.
        """
        irfs_agg_pixel, co, pi = river_to_pixel_gpu_pt(irfs_agg,                       # [N, F] float32/64  (GPU)
                                                       coords[:,1], coords[:,0],       # [N] int32/int64
                                                       self.dest_idxs, self.src_idxs,  # [R] int32/int64 (NOT necessarily sorted)
                                                       self.weights)
        coords_pixel = torch.stack([co, pi]).t()
        kernel_size =  (self.n_cats, self.n_pix)
        return irfs_agg_pixel, coords_pixel, kernel_size

    def forward(self, inp):
        if isinstance(inp, TimeSeriesThDF):
            return self.interpolate_runoff(inp)
        elif isinstance(inp, SparseKernel):
            return self.interpolate_kernel(inp.values, inp.coords)
        else:
            raise NotImplementedError()
            
class StagedCatchmentInterpolator(nn.Module):
    def __init__(self, gs, pixel_df, weight_df):
        """
        """
        super().__init__()
        input_pixel_idxs = pixel_df._columns
        output_cat_idxs = gs.nodes_idx
        
        weight_df = weight_df.loc[output_cat_idxs.index] # First only select the relevant rows
        pixel_columns, pixel_idxs, cis = [], [], []
        for g in tqdm(gs):
            weight_subset = weight_df.loc[g.nodes]
            pixel_col = input_pixel_idxs[weight_subset["pixel_idx"].unique()] #weight_subset["pixel_idx"].unique() #
            pixel_subset_df = pixel_df[pixel_col.index]
            
            pixel_idxs.append(torch.from_numpy(pixel_col.values))
            cis.append(CatchmentInterpolator(g, pixel_subset_df, weight_subset))
        
        self.cis = nn.ModuleList(cis)
        self.pix_idxs = BufferList(pixel_idxs)
        self.map_inp = input_pixel_idxs
        self.map_out = output_cat_idxs
        self.n_cats = len(self.map_out)

    def __len__(self):
        return len(self.cis)

    def __iter__(self):
        return iter(self.cis) 

    def __getitem__(self, idx):
        return self.cis[idx]

    def read_pixels(self, runoff, idx):
        return TimeSeriesThDF(runoff.values[:,self.pix_idxs[idx]],
                              columns=self.cis[idx].map_inp.index, 
                              index=runoff._index)

    def interpolate_runoff(self, runoff, idx):
        local_runoff = self.read_pixels(runoff, idx)
        return self[idx].interpolate_runoff(local_runoff)
    
    def yield_all_runoffs(self, runoff, display_progress=False):
        pbar = tqdm if display_progress else lambda y: y
        for idx in pbar(range(len(self))):
            yield self.interpolate_runoff(runoff, idx)

    def interpolate_kernel(self, idx, irfs_agg, coords):
        return self[idx].interpolate_kernel(irfs_agg, coords)
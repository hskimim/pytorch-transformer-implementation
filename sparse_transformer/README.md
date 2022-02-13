# TODO 

1. reduce memory complexity using sparse attention
    
    In paper, they said that sparse attention matrix helps to  reduce the memory complexity relative to original transformer's full attention.
But in my implementation, sparse attention mechanism (strided, fixed) couldn't reduce it. 

2. Experienced diverging during the training in validation loss
    
    The training loss seems to be converged during training but validation doesn't. I think that this issue comes from pre-activation residual block part, 
    not attention part. I'm researching on it.

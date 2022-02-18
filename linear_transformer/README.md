# TODO 

1. For loop dot product causes slowdown for training

2. Experienced diverging during the training in validation loss
    
    The training loss seems to be converged during training but validation doesn't. I think that this issue comes from pre-activation residual block part, 
    not attention part. I'm researching on it.

3. Add inference code with RNN like

    Now, we have to predict and generate N code to predict N+1 word. But linear-transformer supports RNN style inference.
    And It is needed to be worked. 
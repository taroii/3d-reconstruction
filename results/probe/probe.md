# VGGT trainability probe

GPU: NVIDIA RTX 4000 Ada Generation  (19.6 GiB)
torch 2.6.0+cu124, synthetic inputs, no dataset.

## Measured
mode                views               resolution          trainable_params_M  peak_alloc_GiB      aggregator_fwd_s    heads_fwd_s         backward_s          step_s              steps_per_s         aggregator_share    note                
heads_only          2                   518                 281.5               9.47                0.185               0.1088              0.1998              0.4936              2.026               0.375                                   
heads_only          2                   392                 281.5               8.73                0.1087              0.0634              0.1482              0.3203              3.122               0.339                                   
heads_only          4                   518                 281.5               12.37               0.4087              0.1767              0.3186              0.904               1.106               0.452                                   
heads_only          4                   392                 281.5               9.89                0.2117              0.1077              0.2183              0.5377              1.86                0.394                                   
heads_only          8                   518                 281.5               17.14               0.9685              0.3273              0.5577              1.8535              0.54                0.523                                   
heads_only          8                   392                 281.5               13.21               0.4793              0.1927              0.348               1.0199              0.98                0.47                                    
full                2                   518                 1190.6              --                  --                  --                  --                  --                  --                  --                  OOM                 
full                2                   392                 1190.6              --                  --                  --                  --                  --                  --                  --                  OOM                 
full                4                   518                 1190.6              --                  --                  --                  --                  --                  --                  --                  OOM                 
full                4                   392                 1190.6              --                  --                  --                  --                  --                  --                  --                  OOM                 
full                8                   518                 1190.6              --                  --                  --                  --                  --                  --                  --                  OOM                 
full                8                   392                 1190.6              --                  --                  --                  --                  --                  --                  --                  OOM                 

## Projected cost (20,000 optimizer steps per run)
mode              views             resolution        peak_alloc_GiB    steps_per_s       hours_per_run     hours_30_runs     days_30_runs      
heads_only        2                 518               9.47              2.026             2.74              82.3              3.4               
heads_only        2                 392               8.73              3.122             1.78              53.4              2.2               
heads_only        4                 518               12.37             1.106             5.02              150.7             6.3               
heads_only        4                 392               9.89              1.86              2.99              89.6              3.7               
heads_only        8                 518               17.14             0.54              10.29             308.6             12.9              
heads_only        8                 392               13.21             0.98              5.67              170.1             7.1               

## Verdict

Frozen-trunk training fits: at 8 views and 518px it peaks at 17.14 GiB of 19.6 GiB and runs at 0.54 steps/s, training 281.5M parameters.

The frozen aggregator accounts for 52% of each step. Because it is a fixed function of the input under this recipe, caching its token output would cut step time by about that much, at the cost of disk and of fixing the augmentation pipeline.

A full fine-tune did not complete any configuration, which is the expected result: ~1.26B parameters at 16 bytes each for AdamW is ~19 GiB before activations. The frozen-trunk recipe is not a convenience here, it is the only option on this hardware.

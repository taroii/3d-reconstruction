# Does the umbrella Laplacian measure mean curvature?

Analytic surfaces, perspective camera, avg_dis-normalized pointmaps.
Every estimator sees exactly what the training loss sees.

## Zero test: estimator values where true mean curvature is exactly 0
A plane has H = 0 at every slant.  `frac_of_sphere_R1` is the signal
relative to a genuinely curved sphere (|H| = 1) at the same depth;
a mean-curvature estimator should report ~0 for every plane row.

scene                 estimator   true_H      mean        frac_of_sphere_R1
plane_slant0_d5       umbrella    0           8.134e-18   1.511e-13   
plane_slant0_d5       cotangent   0           1.995e-15   0           
plane_slant0_d5       spacing     0           1.752e-13   1.928e-13   
plane_slant0_d5       grad        0           0.004203    0.7967      
plane_slant30_d5      umbrella    0           1.707e-06   0.06673     
plane_slant30_d5      cotangent   0           1.031e-12   6.611e-13   
plane_slant30_d5      spacing     0           0.02448     0.0638      
plane_slant30_d5      grad        0           0.004663    0.8479      
plane_slant60_d5      umbrella    0           2.415e-05   0.2898      
plane_slant60_d5      cotangent   0           1.908e-13   5.521e-14   
plane_slant60_d5      spacing     0           0.03195     0.08822     
plane_slant60_d5      grad        0           0.009959    1.034       
plane_slant75_d5      umbrella    0           5.767e-05   0.5744      
plane_slant75_d5      cotangent   0           6.503e-14   1.943e-14   
plane_slant75_d5      spacing     0           0.02947     0.08002     
plane_slant75_d5      grad        0           0.01418     1.311       
sphere_R1_d5          umbrella    1           8.853e-05   1           
sphere_R1_d5          cotangent   1           0.9992      1           
sphere_R1_d5          spacing     1           0.4751      1           
sphere_R1_d5          grad        1           0.0066      1           

## Slant sweep: a flat plane tilted from 0 to 75 degrees
Truth is 0 for every row.  Any monotone trend is slant leaking into
the signal.

scene                 estimator   slant_deg   mean        median      
plane_slant0_d5       umbrella    0           8.134e-18   3.469e-18   
plane_slant0_d5       cotangent   0           1.995e-15   0           
plane_slant0_d5       spacing     0           1.752e-13   7.471e-14   
plane_slant0_d5       grad        0           0.004203    0.004203    
plane_slant15_d5      umbrella    15          6.58e-07    6.431e-07   
plane_slant15_d5      cotangent   15          1.556e-12   1.244e-12   
plane_slant15_d5      spacing     15          0.01291     0.01292     
plane_slant15_d5      grad        15          0.0043      0.004265    
plane_slant30_d5      umbrella    30          1.707e-06   1.532e-06   
plane_slant30_d5      cotangent   30          1.031e-12   6.61e-13    
plane_slant30_d5      spacing     30          0.02448     0.02472     
plane_slant30_d5      grad        30          0.004663    0.004473    
plane_slant45_d5      umbrella    45          4.437e-06   3.148e-06   
plane_slant45_d5      cotangent   45          5.044e-13   2.972e-13   
plane_slant45_d5      spacing     45          0.03221     0.03317     
plane_slant45_d5      grad        45          0.005659    0.004898    
plane_slant60_d5      umbrella    60          2.415e-05   6.653e-06   
plane_slant60_d5      cotangent   60          1.908e-13   5.521e-14   
plane_slant60_d5      spacing     60          0.03195     0.03418     
plane_slant60_d5      grad        60          0.009959    0.005455    
plane_slant75_d5      umbrella    75          5.767e-05   1.319e-05   
plane_slant75_d5      cotangent   75          6.503e-14   1.943e-14   
plane_slant75_d5      spacing     75          0.02947     0.031       
plane_slant75_d5      grad        75          0.01418     0.006918    

## Depth bias: two identical spheres at 3 m and 9 m in ONE image
Both have |H| = 2, so a curvature estimator owes far/near = 1.
Within ONE image avg_dis is a single constant and cancels out of the
ratio, so the prediction is the full (Z_far/Z_near)^2 = 9. The
across-image figure is shown for contrast and belongs to the next
table, where each frame carries its own scale.

estimator   near_median far_median  far_over_nearpredicted_within_image_Z2predicted_across_images_Z
umbrella    2.198e-05   0.0002144   9.756       9           3           
cotangent   2           1.998       0.9992      9           3           
spacing     0.7557      0.6937      0.9179      9           3           
grad        0.004223    0.01377     3.262       9           3           

## Depth sweep ACROSS images: one sphere alone, 3 m to 12 m
Identical geometry in every row, so a curvature estimator holds
`drift_vs_3m` at 1.00.  avg_dis cancels one of the two powers of Z,
not both, so a linear-in-depth bias survives BETWEEN images as well
as within one.

estimator   dist        true_H      median      drift_vs_3m 
umbrella    3           2           2.778e-05   1           
cotangent   3           2           2           1           
spacing     3           2           0.7564      1           
grad        3           2           0.005291    1           
umbrella    5           2           4.672e-05   1.682       
cotangent   5           2           1.999       0.9998      
spacing     5           2           0.7215      0.9539      
grad        5           2           0.005325    1.006       
umbrella    7           2           6.469e-05   2.329       
cotangent   7           2           1.999       0.9995      
spacing     7           2           0.7036      0.9302      
grad        7           2           0.005292    1           
umbrella    9           2           8.328e-05   2.998       
cotangent   9           2           1.998       0.9991      
spacing     9           2           0.696       0.9202      
grad        9           2           0.005298    1.001       
umbrella    12          2           0.0001102   3.966       
cotangent   12          2           1.997       0.9984      
spacing     12          2           0.6862      0.9072      
grad        12          2           0.005275    0.9969      

## Selection rate by population: what the 'boundary subset' targets
A frame holding a sphere (genuine curvature) and a pure depth step
(true H = 0).  `rate_*` is P(pixel in top decile | population).
`structural_cap` is the most a stencil can reach: the published
first-order control uses backward differences and so can only fire
on one of the two step columns.  `frac_exactly_zero` is the share of
valid pixels where the signal is identically 0, which is why 0.10 is
not the right chance baseline for the umbrella.

estimator   rate_step   structural_caprate_sphere rate_flat   frac_exactly_zerostep_px     
umbrella    1           1           0.819       0           0.4432      508         
cotangent   0           1           0.9397      0           0.4334      508         
spacing     0           1           0.8997      0           0.4432      508         
grad        0.5         0.5         0.1008      0.0863      0           510         

## Rank correlation with the first-order control
How much of the umbrella signal's pixel ranking the gradient control
reproduces.  High rho explains why the two arms perform alike.

scene                 spearman_rhon           
plane_slant60_d5      1           64516       
sphere_R1_d5          0.9995      13061       
step_sphere_R0.5_j2   0.1237      64516       
two_spheres_R0.5_3v9  0.996       9950        

## Recovery of |H| = 1/R on spheres
`measured_over_true` should be ~1 for an estimator of mean curvature,
and constant across R for one that is merely proportional to it.

estimator   radius      dist        true_H      measured    measured_over_true
umbrella    0.2         5           5           0.0001123   2.246e-05   
cotangent   0.2         5           5           4.991       0.9983      
spacing     0.2         5           5           1.698       0.3395      
grad        0.2         5           5           0.005256    0.001051    
umbrella    0.35        5           2.857       6.598e-05   2.309e-05   
cotangent   0.35        5           2.857       2.855       0.9993      
spacing     0.35        5           2.857       1.004       0.3513      
grad        0.35        5           2.857       0.005312    0.001859    
umbrella    0.5         5           2           4.672e-05   2.336e-05   
cotangent   0.5         5           2           1.999       0.9997      
spacing     0.5         5           2           0.7215      0.3607      
grad        0.5         5           2           0.005325    0.002662    
umbrella    0.75        5           1.333       3.115e-05   2.336e-05   
cotangent   0.75        5           1.333       1.333       0.9998      
spacing     0.75        5           1.333       0.4995      0.3747      
grad        0.75        5           1.333       0.005302    0.003977    
umbrella    1           5           1           2.296e-05   2.296e-05   
cotangent   1           5           1           0.9999      0.9999      
spacing     1           5           1           0.3874      0.3874      
grad        1           5           1           0.005275    0.005275    

## Verdict

Slant leaks into the signal. A perfectly FLAT plane, true mean curvature exactly zero, reaches 0.57x the median umbrella signal of a genuinely curved sphere (|H| = 1) at the same depth by 75 degrees of slant, rising monotonically from zero. The cotangent reference reports 1.9e-14 on the same surface. The effect is real but moderate once near-horizon pixels beyond a sensor-plausible depth are excluded; leaving them in inflates it by more than two orders of magnitude and is not a defensible way to report it.

Depth biases the signal, and does so linearly. Two identical spheres at 3 m and 9 m within one image: the umbrella reports the far one at 9.76x the near one, against the 1.00 a curvature estimator owes and the 1.00 the cotangent reference delivers. This matches the Z^2 that theory predicts for a second difference over a pixel grid of spacing Z/f, since within one image avg_dis is a single constant that cancels out of the ratio. Across images each frame carries its own scale, which removes one power of Z, and the same sphere alone in its own frame still draws 3.97x more weight at 12 m than at 3 m -- against a predicted 4.00x. Both predictions hold, which is what makes the Z^2 account of this signal hard to argue with. Identical geometry is supervised more strongly the further away it is.

The signal targets depth discontinuities, not curvature. In a frame holding both, 100% of the pure-step pixels land in the top decile by umbrella signal -- the paper's own boundary evaluation subset -- against 82% of genuinely curved pixels and 0.0% of flat ones. True mean curvature at the step is zero. The cotangent reference selects 0% of them. The umbrella is identically zero on 44% of valid pixels, so its top decile is not a tenth of a continuum but essentially the set of pixels where anything is happening at all.

First- and second-order signals are near-interchangeable where it matters. On every smooth scene tested the umbrella's pixel ranking agrees with the published first-order gradient control at Spearman rho >= 0.996. On the scene that does contain a discontinuity they diverge (rho = 0.124), so the two signals are separable only by the depth edges -- which is to say, only by the thing that is not curvature. This is the mechanical reason the paper's curvature and gradient arms score within seed noise of each other.

The comparison itself is sound: the cotangent reference recovers |H| = 1/R across radii to within 0.2%. At fixed depth the umbrella is itself proportional to |H| to within 3%, so its failure is not an inability to see curvature but an inability to separate curvature from slant and depth.

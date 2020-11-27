# OpInfErrEst

Implementation of operator inference for error estimation to accompany the manuscript [Probabilistic error estimation for non-intrusive reduced models learned from data of systems governed by linear parabolic partial differential equations (arXiv:2005.05890)](https://arxiv.org/abs/2005.05890) by W.I.T. Uy and B. Peherstorfer. The code returns the reduced model operators and operators in the error estimator for intrusive and non-intrusive model reduction as well as relevant error quantities.

The sample script provided <code>main.py</code> considers the heat equation example in the manuscript (Section 4.2) and reproduces Figures 1c, 2.

All the functions for intrusive (traditional) model reduction are contained in <code>IntrusiveROM.py</code> while all the functions for non-intrusive (operator inference) model reduction are contained in <code>OpInf.py</code>. 

The function <code>RunOpInf.py</code> demonstrates that the proposed non-intrusive approach is able to recover the reduced model and error operators in intrusive model reduction. It is also provides reasonable bounds for the constants in the error estimator of the intrusive reduced model. 

To cite this work, please use the following Bibtex entry:

<pre><code>@ARTICLE{UyP2020,
       author = {{Uy}, Wayne Isaac Tan and {Peherstorfer}, Benjamin},
        title = "{Probabilistic error estimation for non-intrusive reduced models learned from data of systems governed by linear parabolic partial differential equations}",
      journal = {arXiv e-prints},
     keywords = {Mathematics - Numerical Analysis, Computer Science - Machine Learning},
         year = 2020,
        month = may,
          eid = {arXiv:2005.05890},
        pages = {arXiv:2005.05890},
archivePrefix = {arXiv},
       eprint = {2005.05890},
 primaryClass = {math.NA},
}
</code></pre>

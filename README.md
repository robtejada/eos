(!!!) IF YOU ENCOUNTER GIT LFS STORAGE ISSUES: Anyone can download a .zip file version stored here: https://drive.google.com/drive/u/1/folders/1V13BQLZ9_VKWoZQp6T5i7OXp8zreIrt7 and the following Zenodo: https://zenodo.org/records/15659606.

If you download the Google Drive zipfile: This zip file contains the entire EOS data and code presented in Tejada Arevalo et al. (2024). I realized that some users have encountered problems with Git LFS in the GitHub repository (https://github.com/Rob685/eos/tree/main), so this is another option for obtaining the EOS data. 

Importantly, this is the [second release](https://zenodo.org/records/14194431) with an updated eos mixtures module called `eos_class.py`. This class handles all the available H-He-Z mixtures, with Z primarily being the water EOS of Haldemann et al (2020; AQUA). 

# H-He EOS quantities for planetary evolution [![DOI](https://zenodo.org/badge/639560032.svg)](https://zenodo.org/doi/10.5281/zenodo.10659248) 
### EOS zip downloadable file (if GIT LFS Storage is limited): [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15659606.svg)](https://doi.org/10.5281/zenodo.15659606)


This repository contains H-He EOSes from [Saumon et al. (1995; SCvH)](https://ui.adsabs.harvard.edu/abs/1995ApJS...99..713S/abstract), [Militzer & Hubbard (2013; MH13)](https://iopscience.iop.org/article/10.1088/0004-637X/774/2/148/meta), [Chabrier et al. (2019; CMS19)](https://iopscience.iop.org/article/10.3847/1538-4357/aaf99f/meta), [Chabrier & Debras (2021; CD21)](https://iopscience.iop.org/article/10.3847/1538-4357/abfc48/meta), and [Mazevet et al. (2022; MLS22)](https://www.aanda.org/articles/aa/abs/2022/08/aa35764-19/aa35764-19.html). The CMS and MLS EOSes are supplemented by the work of [Howard et al. (2023a)](https://www.aanda.org/articles/aa/pdf/2023/04/aa44851-22.pdf) to account for the non-ideal entropy and volume interactions they calculated from MH13 and CD21. Moreover, we calculate H-He-Z mixutures using a water EOS from [Haldemann et al. (2020)](https://www.aanda.org/articles/aa/full_html/2020/11/aa38367-20/aa38367-20.html), iron, and post-perovskite EOSes from Jisheng Zhang (private communication). 

To ease thermodynamic derivative calculations, we have inverted the H-He EOSes to provide several independent axes:


<img width="603" alt="Screenshot 2023-12-30 at 20 01 17" src="https://github.com/Rob685/eos/assets/48569647/5c18c88b-c64a-425a-ac1b-87cb204fc16c">

and thus Github's Large File Storage is required. 

To download and use these tables, 

1. ```git lfs install```
2. ```git clone https://github.com/Rob685/eos.git```

# Importing the EOS tables and functions
Initialization and usage example:


```
# in a folder above the eos directory:
from eos import eos_class

mix_aqua = eos_class.mixtures(hhe_eos='cd', z_eos='aqua', y_prime=True) # y_prime indicates whether the subsequent Y inputs are Y/(X+Y)

logpgrid = np.linspace(6, 14, 100)
logtval = np.full_like(logpgrid, 3.5) # isothermal example
yval = np.full_like(logpgrid, 0.25)
zval = np.full_like(logpgrid, 0.0) # H-He only example

s = mix_aqua.get_s_pt(logpgrid, logtval, yval, zval) # output in erg/g/K units
logrho = mix_aqua.get_logrho_pt(logpgrid, logtval, yval, zval) # output in log10 g/cm^3
```


I will update the tutorials soon. 



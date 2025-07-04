# Dynamic-Transformer

Important Links:
- LaTeX for Report: https://www.overleaf.com/read/xvcyypchrghp#fe71ba
- Google Docs For Project Files: https://docs.google.com/document/d/1rjFtPSFybRemLVBta1tl7ufb-vcAEalQxBYZ3M0Jnmw/edit?tab=t.0

## Project Setup

configs

here we have some yaml files which will be how the paramaters for running training, testing, evaluating and running experiments that do all these steps. Also have flag to be able to run same evaluation multiple times on different random seeds???

data

Here we have the datasets wether they are tiny shakesperae the pile or shortsotries or if it's simpler scripts which cna download these datasets

experiments

if in the config.yaml we find that their is a flag saying that the runtime is an experiemnt we will log and save the results from this experiment in this folder under it's own special folder where we will also keep all the data for that specific experiment.

logs

for any runtime wether experiment or not we save that data into here so that post runtime we cna analyse results. This should not be tracked by git though as really these are temp files that are made to help any live debugging and for dev purposes.

model-weights

Here we have data (weights) and scripts in order to download models and save some of our own models into. Their should also be some code in order to upload and save these models into hugging face so that we can fetch them later.

notebooks

this is just to save some of the notebooks I am using in order to develop this repo and to do quick research.

scripts

bash scripts and that kind of thing 

src
- models
- test
- train

this is the real meat of the project and works in the following way

all the code defining some example archtiectures and our new bayesian surprise based architectures are here in the models subfolder. We will also be defining models here that will use weights from the model-weigths folder.

train will then have different files which will specify different ways of training for example Adam vs SGD or maybe more interestingly we could train using some alternative loss functions, perhaps some load balancing based losses or a gate warmup loss (so that at the begining gates are more likely to open so that they learn more)

NOTE: onece a model is trained the goal is to save this model into the model-weights folder

test: perplexity, and other metircs but ideally this should load back in the model from the model-weights file.

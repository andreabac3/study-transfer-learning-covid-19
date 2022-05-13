from os import makedirs
from typing import List
import wandb

api: wandb.Api = wandb.Api()
prefix = f"ckpt/1.0/"
# runs = api.runs(entity="bacciu-taiello", project="covid-19")
runs: wandb.apis.public.Runs = api.runs("bacciu-taiello/covid-19-train")
for run in runs:
    name = run.name
    print(name)
    makedirs(f"{prefix}/{name}", exist_ok=True)
    ckpts: List[wandb.apis.public.File] = [
        file_run for file_run in run.files() if ".ckpt" in file_run.name
    ]
    assert len(ckpts) == 1, f"More than 1 ckpt in {name}"
    ckpt: wandb.apis.public.File = ckpts[0]
    ckpt.download(f"{prefix}/{name}", replace=True)

# wandb.api.download_file("bacciu-taiello/covid-19-train/1muxxamf/")
# files = wandb.apis.public.Files(run=my_run)
# files = api.Files(my_run)
# print(files)

"""
best_model_file= cerca il file ckpt in files
best_model_ckpt = api.file(best_model_file)
best_model_ckpt.download( root=".", replace=(False))
"""

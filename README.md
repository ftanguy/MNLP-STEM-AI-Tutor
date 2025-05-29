[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/xR4NrJfU)
# CS-552 - Final Submission

Welcome to the final submission for the MNLP project! For this last submission, as you can read in the [project description](https://docs.google.com/document/d/1BUeMoBb2zwg1YvO_OnLcqFiXQd8hxiF8ub-cM2MbX_M/edit?usp=sharing), you have 4 main goals:

1. Finish training the four models detailed in the project description: DPO, MCQA, Quantized-MCQA, RAG-MCQA, optimizing their performance as well as you can, and submit them. (individual work, one model per member)
2. Submit the code you used to train your models, including the training script for each model.
3. Submit the training data used for each model.
4. Write a final report (group work)

Note: Note that for this final submission, the models will be evaluated based on their performance.

## Repo Structure

The repo has 6 folders, 4 of which serve for you to submit all four deliverables:

1. `_templates` contains the LaTeX template for your final report. You MUST use this template.
2. `_test` contains scripts that run automated tests to validate that your submission is correctly formatted.
3. `model_configs` should be populated by you with the 4 model config YAML files, including `dpo_model.yaml`, `mcqa_model.yaml`, `quantized_model.yaml`, and `rag_model.yaml`. Make sure you fill in the important information in each config file, and the information is exactly what is used for evaluating your models. You need to change `<HF_USERNAME_team_member_X>` to your Huggingface Hub username. Make sure that you have submitted your models to the correct Huggingface Hub repositories, adhering to the following name convention:

- `<HF_USERNAME_team_member_DPO>/MNLP_M3_dpo_model`
- `<HF_USERNAME_team_member_MCQA>/MNLP_M3_mcqa_model`
- `<HF_USERNAME_team_member_QUANTIZED>/MNLP_M3_quantized_model`
- `<HF_USERNAME_team_member_RAG>/MNLP_M3_rag_model`

    For the team member responsible for the RAG model, make sure you have submitted the two additional deliverables specific to RAG:

- `<HF_USERNAME_team_member_RAG>/<RAG_DOCUMENT_REPO_NAME>`, replace `<RAG_DOCUMENT_REPO_NAME>` with the actual Huggingface Hub repo name of your submitted RAG documents.
- `<HF_USERNAME_team_member_RAG>/MNLP_M3_document_encoder`

4. `pdf` should be filled by you with your final report PDF (titled `<YOUR-GROUP-NAME>.pdf`). This directory should then have only one PDF.
5. `data` contains `data_repo.json`. In this file, you need to change `<HF_USERNAME_team_member_X>` to your Huggingface Hub username. Make sure that you have submitted the training data for your 4 models to the correct Huggingface Hub repositories, adhering to the following name convention:

- `<HF_USERNAME_team_member_DPO>/MNLP_M3_dpo_dataset`
- `<HF_USERNAME_team_member_MCQA>/MNLP_M3_mcqa_dataset`
- `<HF_USERNAME_team_member_QUANTIZED>/MNLP_M3_quantized_dataset`
- `<HF_USERNAME_team_member_RAG>/MNLP_M3_rag_dataset`

6. The `code` must contain the following:

- Four Bash Training Scripts:
You must provide four executable Bash scripts (.sh files) in the root of the `code` directory. These scripts are essential for reproducing your results and obtaining models equivalent to those you submit.

    - `train_dpo.sh`
    - `train_mcqa.sh`
    - `train_quantized.sh`
    - `train_rag.sh` 

- Four Corresponding Subfolders for Training Code:
For each of the four models, you should create a dedicated subfolder within code/ to house its specific training code. These folders should have the following structure:

    - `train_dpo/`
    - `train_mcqa/`
    - `train_quantized/`
    - `train_rag/ `

    By running these scripts we should be able to reproduce your training process and obtain models that are functionally equivalent to the ones you have developed and submitted.

## Running the tests manually

The autograding tests run automatically with every commit to the repo. If you want to trigger them manually, follow the instructions from the previous milestone.

## Evaluation Suite

For the final submission, same as M2, we provide you with an [evaluation suite](https://github.com/eric11eca/lighteval-epfl-mnlp) to benchmark each of the four models. As we covered in the compute tutorial session, details about how we evaluate your models are listed in these slides: [Evaluation Implementation Slides](https://docs.google.com/presentation/d/1SoVY4u6fDgXQ-F6TdarwaINhQ30NnE9Zb2XFavz3PGU/edit?usp=sharing).

We provide you with a demo MCQA evaluation dataset and a demo DPO evaluation dataset on the Huggingface Hub:

- [MCQA demo dataset](https://huggingface.co/datasets/zechen-nlp/MNLP_STEM_mcqa_demo)
- [DPO demo dataset](https://huggingface.co/datasets/zechen-nlp/MNLP_dpo_demo)

Also, for the RAG part, here is a demo Huggingface repo for the RAG documents and a collection of pretrained huggingface embedding models you can start with:

- [RAG documents demo](https://huggingface.co/datasets/m-ric/huggingface_doc)
- [RAG embedding model collection on Huggingface Hub](https://huggingface.co/models?library=sentence-transformers)

## Submission Via Huggingface Hub

Recall that you have to submit your model weights, RAG documents, and training data via [Huggingface Hub](https://huggingface.co/). Make sure you

- Have a Huggingface account.
- Make all your submissions public on the Huggingface Hub.

Please take a look at the documents on how to [upload your dataset](https://huggingface.co/docs/datasets/en/upload_dataset) (documents which are also a dataset) and [upload your model weights](https://huggingface.co/docs/transformers/en/model_sharing#pushtohubmixin). Note that you also have to **push the tokenizers to the same model repository**.

## Validating Your Submission
**After you push your model weights and RAG documents to the correct Huggingface Hub repositories, make sure to test your models with the official [evaluation suite](https://github.com/eric11eca/lighteval-epfl-mnlp) in a fresh and clean environment (not the same environment you used for development).**

**If you got an error in the clean environment, you are responsible for debugging and correcting it.**

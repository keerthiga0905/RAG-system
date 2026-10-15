---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:115
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: Tell me about The current Placement Records page lists selected
    "Highest P at CIT.
  sentences:
  - '[2. STUDENT HELP DESK]

    Source: https://cit.edu.in/help-desk Email: cithelpdesk@cit.edu.in Phone: 94868
    37757 The page provides a Help Desk form for the name of the student/faculty/parent/visitor,
    email address, phone/mobile number, and a category selection. The form is the
    published route for submitting a help-desk enquiry.'
  - '[3. INTERNSHIP INFORMATION]

    of these internships are converted into job offers at the end of the internship
    programme. It also says some companies offer internships to UG students pursuing
    fast-track programmes and that PG students are provided six-m...'
  - '[4. Current-year salary details:]

    The current Placement Records page lists selected "Highest Package" entries for
    2023 - 24, 2024 - 25, and 2025 - 26. Average and median salaries for these years
    were not found in the visible text of that page.'
- source_sentence: 'Tell me about URL: https://cit.edu.in/ Content inspected: official
    navigat at CIT.'
  sentences:
  - '[11. CIT homepage]

    URL: https://cit.edu.in/ Content inspected: official navigation and links to placement
    pages; homepage visible text includes a "PLACED STUDENTS" counter, but the number
    rendered in the extracted text was incomplete ("8 0") and is not used as a reliable
    statistic.'
  - '[1. Placement-year label conflict in the 50-page Placement Report:]

    The report''s introduction says it covers placement year 2021 - 2022. However,
    multiple tables and the narrative on later pages explicitly say "Placement Statistics
    2022 - 2023" and "Overview of Placement 2022 - 2023." Both labels are preserved
    in this file.'
  - '[9. PLACEMENT REPORTS, BROCHURES, AND RELATED DOCUMENTS]

    1. Placement Report (PDF; intro identifies placement year 2021 - 2022; includes
    tables and narrative that also refer to 2022 - 2023) URL: https://cit.edu.in/uploads/epdf_documents/d5c3b7e01e813...'
- source_sentence: what is the email and phone number for cit help desk?
  sentences:
  - '[General Overview]

    COIMBATORE INSTITUTE OF TECHNOLOGY (CIT) PLACEMENT AND CAREER DEVELOPMENT DATASET
    Suggested filename: 06_placements_and_career.txt Access date: 2026-10-15 Scope
    note: This file covers the official CIT pages and linked documents inspected for
    this collection. It does NOT claim that every placement-related page or every
    linked file across the entire CIT website has been inspected.'
  - '[2. STUDENT HELP DESK]

    Source: https://cit.edu.in/help-desk Email: cithelpdesk@cit.edu.in Phone: 94868
    37757 The page provides a Help Desk form for the name of the student/faculty/parent/visitor,
    email address, phone/mobile number, and a category selection. The form is the
    published route for submitting a help-desk enquiry.'
  - '[9. STUDENT SERVICES AND ADMINISTRATIVE OFFICES]

    Student information source: https://cit.edu.in/index.php/students/information-for-students
    Principal: Dr. A. Rajeswari Phone: +91-422-2573187 Fax: +91-422-2575020 Email:
    principal@cit.edu.in Administra...'
- source_sentence: 'Tell me about - Eligibility: BC/MBC/DNC Students who secure the
    highest ma at CIT.'
  sentences:
  - '[5. The Commissioner of Technical Education, Chennai - Member - Government of
    Tamil Nadu Nominee]

    6. Dr. J. K. Tripathi, Joint Secretary, University Grants Commission, Bahadurshah
    Zafar Marg, New Delhi - 110 002 - Member - Nominee of University Grants Commission
    7. Dr. R. Thenmozhi, Professor in Civil Engineering, Government College of Technology,
    Coimbatore - 641 013 - Member - Nominee of Anna University'
  - '[7. State Government Anna Award for Backward / Most Backward / DNC Students from
    Tamil Nadu:]

    - Eligibility: BC/MBC/DNC Students who secure the highest marks in the HSC Examination
    at district level. - Sanctioning Authority: District Backward Classes and Minority
    Welfare Officer of the district concerned.'
  - '[5. TUITION AND PROGRAMME-RELATED FEES]

    A. Academic Year 2026-27 (Published Structure): - B.E. / B.Tech (Aided) - Counselling:
    * I Year: Rs. 33,350 * Lateral Entry: Rs. 33,350 * II Year: Rs. 31,735 * III Year:
    Rs. 31,735 * IV Year: Rs. 35,035 - B.E. ...'
- source_sentence: 'Tell me about URL: https://cit.edu.in/information-technology Use:
    Example at CIT.'
  sentences:
  - '[4. LIBRARY FACILITIES, HOURS AND ONLINE RESOURCES]

    Current library navigation/source: https://cit.edu.in/library/about Library overview:
    The library was established in 1956 and is adjacent to the main building. The
    current page gives the area as 13,...'
  - '[8. State Government Gandhi Memorial Award students from Tamil Nadu:]

    - Eligibility: SC/ST Students who secure the highest marks in the district in
    the H.S.C Examination. - Sanctioning Authority: Director of Adi Dravidar Welfare,
    Chennai-5.


    [9. Tamil Pudhalvan Scheme:]

    - Eligibility: Male students who studied from 6th to 12th standard in Government
    Schools and Government Aided Schools in Tamil Medium in the State.'
  - '[22. Information Technology department page]

    URL: https://cit.edu.in/information-technology Use: Example department page reviewed
    for department contact publication. Missing: No separate department-office phone/email
    identified in the inspected page; this is not a full audit of all department pages.
    END OF FILE'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy@1
- cosine_accuracy@3
- cosine_accuracy@5
- cosine_accuracy@10
- cosine_precision@1
- cosine_precision@3
- cosine_precision@5
- cosine_precision@10
- cosine_recall@1
- cosine_recall@3
- cosine_recall@5
- cosine_recall@10
- cosine_ndcg@10
- cosine_mrr@10
- cosine_map@100
model-index:
- name: SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2
  results:
  - task:
      type: information-retrieval
      name: Information Retrieval
    dataset:
      name: cit val retrieval
      type: cit-val-retrieval
    metrics:
    - type: cosine_accuracy@1
      value: 0.46153846153846156
      name: Cosine Accuracy@1
    - type: cosine_accuracy@3
      value: 0.5384615384615384
      name: Cosine Accuracy@3
    - type: cosine_accuracy@5
      value: 0.6153846153846154
      name: Cosine Accuracy@5
    - type: cosine_accuracy@10
      value: 0.6153846153846154
      name: Cosine Accuracy@10
    - type: cosine_precision@1
      value: 0.46153846153846156
      name: Cosine Precision@1
    - type: cosine_precision@3
      value: 0.17948717948717946
      name: Cosine Precision@3
    - type: cosine_precision@5
      value: 0.12307692307692307
      name: Cosine Precision@5
    - type: cosine_precision@10
      value: 0.061538461538461535
      name: Cosine Precision@10
    - type: cosine_recall@1
      value: 0.46153846153846156
      name: Cosine Recall@1
    - type: cosine_recall@3
      value: 0.5384615384615384
      name: Cosine Recall@3
    - type: cosine_recall@5
      value: 0.6153846153846154
      name: Cosine Recall@5
    - type: cosine_recall@10
      value: 0.6153846153846154
      name: Cosine Recall@10
    - type: cosine_ndcg@10
      value: 0.5432004855111423
      name: Cosine Ndcg@10
    - type: cosine_mrr@10
      value: 0.5192307692307693
      name: Cosine Mrr@10
    - type: cosine_map@100
      value: 0.5289635578097116
      name: Cosine Map@100
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps inputs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'BertModel'})
  (1): Pooling({'embedding_dimension': 384, 'pooling_mode': 'mean', 'include_prompt': True})
  (2): Normalize({'module_input_name': 'sentence_embedding', 'module_output_name': 'sentence_embedding'})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
queries = [
    'Tell me about URL: https://cit.edu.in/information-technology Use: Example at CIT.',
]
documents = [
    '[22. Information Technology department page]\nURL: https://cit.edu.in/information-technology Use: Example department page reviewed for department contact publication. Missing: No separate department-office phone/email identified in the inspected page; this is not a full audit of all department pages. END OF FILE',
    '[4. LIBRARY FACILITIES, HOURS AND ONLINE RESOURCES]\nCurrent library navigation/source: https://cit.edu.in/library/about Library overview: The library was established in 1956 and is adjacent to the main building. The current page gives the area as 13,...',
    '[8. State Government Gandhi Memorial Award students from Tamil Nadu:]\n- Eligibility: SC/ST Students who secure the highest marks in the district in the H.S.C Examination. - Sanctioning Authority: Director of Adi Dravidar Welfare, Chennai-5.\n\n[9. Tamil Pudhalvan Scheme:]\n- Eligibility: Male students who studied from 6th to 12th standard in Government Schools and Government Aided Schools in Tamil Medium in the State.',
]
query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(documents)
print(query_embeddings.shape, document_embeddings.shape)
# [1, 384] [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(query_embeddings, document_embeddings)
print(similarities)
# tensor([[0.6556, 0.2716, 0.0725]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Information Retrieval

* Dataset: `cit-val-retrieval`
* Evaluated with [<code>InformationRetrievalEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.sentence_transformer.evaluation.InformationRetrievalEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| cosine_accuracy@1   | 0.4615     |
| cosine_accuracy@3   | 0.5385     |
| cosine_accuracy@5   | 0.6154     |
| cosine_accuracy@10  | 0.6154     |
| cosine_precision@1  | 0.4615     |
| cosine_precision@3  | 0.1795     |
| cosine_precision@5  | 0.1231     |
| cosine_precision@10 | 0.0615     |
| cosine_recall@1     | 0.4615     |
| cosine_recall@3     | 0.5385     |
| cosine_recall@5     | 0.6154     |
| cosine_recall@10    | 0.6154     |
| **cosine_ndcg@10**  | **0.5432** |
| cosine_mrr@10       | 0.5192     |
| cosine_map@100      | 0.529      |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 115 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>sentence_2</code>
* Approximate statistics based on the first 100 samples:
  |          | sentence_0                                                                         | sentence_1                                                                           | sentence_2                                                                          |
  |:---------|:-----------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type     | string                                                                             | string                                                                               | string                                                                              |
  | modality | text                                                                               | text                                                                                 | text                                                                                |
  | details  | <ul><li>min: 13 tokens</li><li>mean: 25.42 tokens</li><li>max: 44 tokens</li></ul> | <ul><li>min: 59 tokens</li><li>mean: 218.17 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 48 tokens</li><li>mean: 75.32 tokens</li><li>max: 134 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                    | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | sentence_2                                                                                                                                                                                                                                                                    |
  |:--------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Tell me about Source: https://cit.edu.in/students/scholarship - Fee Conces at CIT.</code>               | <code>[4. FINANCIAL ASSISTANCE AND FEE CONCESSIONS]<br>Source: https://cit.edu.in/students/scholarship - Fee Concession under Tamil Nadu Education Rule 92 MER: "Full fee concession will be given to the students belonging to the Scheduled Caste / Scheduled Tribe and converts to Christianity regardless of their annual income for Govt. Aided courses."<br><br>[5. TUITION AND PROGRAMME-RELATED FEES]<br>Source: https://cit.edu.in/admission/tuition-fees</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | <code>[2. INSTITUTIONAL & ALUMNI-ADMINISTERED SCHOLARSHIPS]<br>Source: https://cit.edu.in/students/scholarship The institution lists several institutional endowment and alumni-administered charitable trusts/scholarships: - CIT Endowment Fund Scholarship - 196...</code> |
  | <code>What is the income eligibility limit for Tamil Nadu State Government BC/MBC scholarships at CIT?</code> | <code>[2. INSTITUTIONAL & ALUMNI-ADMINISTERED SCHOLARSHIPS]<br>Source: https://cit.edu.in/students/scholarship The institution lists several institutional endowment and alumni-administered charitable trusts/scholarships: - CIT Endowment Fund Scholarship - 1967 Batch Financial Assistance to Students - 1972 Batch Financial Assistance to Students - 1973 Batch Endowment Scholarship 2024-25 - CIT 1984 - I Alumni Charitable Trust - 1980-84 - II Batch Alumni Charitable Trust - CIT86 Alumni Charitable Trust - 1991 Batch Alumni Charitable Trust - 1986-89 Batch B.Sc. Applied Sciences Alumni Charitable Trust - Friends of CIT 2000 Batch Alumni Charitable Trust - 2017 Batch Siragugal Trust (Siragugal Scholarship) Private Scholarship Referenced: - Tamil Nadu Educational Trust Scholarship Note on amounts/specific eligibility: Detailed cut-offs and financial disbursal figures for individual alumni batches are published through respective trust circulars and were not listed in tabular form on the main scholarsh...</code> | <code>[1. GENERAL SCHOLARSHIP POLICY & APPLICATION PROCEDURES]<br>Source: https://cit.edu.in/students/scholarship Application Procedure: - Students wishing to apply for a scholarship must submit a duly completed application to the Office of the Principal. - A...</code> |
  | <code>Tell me about Sources: https://cit.edu.in/naac/naac ; Grade sheet https:// at CIT.</code>               | <code>[6.1 NAAC]<br>Sources: https://cit.edu.in/naac/naac ; Grade sheet https://cit.edu.in/uploads/epdf_documents/6e98965d7fe8ba1e4b4f8e4279279523.pdf ; Certificate https://cit.edu.in/uploads/epdf_documents/a485ef1de7fe5647fbd77fefd0932c34.pdf - Grade sheet: NAAC Institutional Assessment and Accreditation, Accreditation Cycle 2; Coimbatore Institute of Technology, Coimbatore, Tamil Nadu, 641014; Track ID TNCOGN100891; AISHE-ID C-36969; Visit dates 15-05-2025 to 16-05-2025. - Institutional CGPA 3.09 (3094/1000); Grade A. - Criterion-wise CGPA: Curricular Aspects 3.5; Teaching-learning and Evaluation 3.35; Research, Innovations and Extension 2.98; Infrastructure and Learning Resources 2.45; Student Support and Progression 2.97; Governance, Leadership and Management 2.69; Institutional Values and Best Practices 3.06. - Certificate PDF: scanned; text extraction is garbled. Legible fragments indicate CGPA 3.09 on a four-point scale, grade A, and a validity/date line, but the exact validity and issue...</code> | <code>[6.2 NBA (programme-level accreditation)]<br>Sources: https://cit.edu.in/others/nba ; https://cit.edu.in/uploads/epdf_documents/37189efdc2fede14a11805c173e4aa4c.pdf - NBA letter dated 01-08-2025 (File No. 33-109/2010-NBA as extracted): UG engineering pr...</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `num_train_epochs`: 4
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 8
- `num_train_epochs`: 4
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 8
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `dataloader_multiprocessing_context`: None
- `dataloader_in_order`: True
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: None
- `fsdp_config`: None
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}
- `warmup_ratio`: None

</details>

### Training Logs
| Epoch | Step | cit-val-retrieval_cosine_ndcg@10 |
|:-----:|:----:|:--------------------------------:|
| -1    | -1   | 0.4533                           |
| 1.0   | 15   | 0.4990                           |
| 2.0   | 30   | 0.5148                           |
| 3.0   | 45   | 0.5432                           |


### Training Time
- **Training**: 4.0 minutes
- **Evaluation**: 35.9 seconds
- **Total**: 4.6 minutes

### Framework Versions
- Python: 3.12.10
- Sentence Transformers: 6.1.0
- Transformers: 5.19.0
- PyTorch: 2.14.1+cpu
- Accelerate: 1.15.0
- Datasets: 5.1.0
- Tokenizers: 0.23.3

## Additional Resources

- [Training and Finetuning Embedding Models with Sentence Transformers](https://huggingface.co/blog/train-sentence-transformers): the end-to-end guide for training or finetuning Sentence Transformer models.
- [Introduction to Matryoshka Embedding Models](https://huggingface.co/blog/matryoshka): variable-size embeddings that can be truncated with minimal quality loss.
- [Binary and Scalar Embedding Quantization for Significantly Faster & Cheaper Retrieval](https://huggingface.co/blog/embedding-quantization): post-training compression of embedding vectors.
- [Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers): use text, image, audio, and video models through the same API.
- [Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-multimodal-sentence-transformers): train multimodal embedding models, with a Visual Document Retrieval walkthrough.

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{oord2019representationlearningcontrastivepredictive,
      title={Representation Learning with Contrastive Predictive Coding},
      author={Aaron van den Oord and Yazhe Li and Oriol Vinyals},
      year={2019},
      eprint={1807.03748},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1807.03748},
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->